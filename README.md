# NVIDIA GPU power modeling

**SXM 모듈의 V100·A100·H100**에서 **충분히 활용한 조건의 실측 pJ/FLOP·pJ/bit 최소점과 주변 측정점**을 찾기 위한 CUDA/NVML 실험 도구다. FP16 Tensor, L1, L2, HBM의 working set·thread/block·stride·SM/memory clock을 바꾸고, 수초 동안 전력과 처리량을 함께 측정한다.

SXM은 GPU 모듈의 장착 형태이고 HBM은 측정할 메모리 계층이다. 실제 메모리 용량·SKU·SM 수를 이름만으로 확정하지 않는다. 이 저장소에는 실측 GPU 숫자가 들어 있지 않다. 전체 단가, 승인된 전후 idle 증가분, 같은 process에서 짝지은 active-reference 대비를 별도로 보고한다. `idle`은 운영상 기준이며 순수 누설 전력이 아니다. 지원되는 memory power scope도 전체 GPU scope와 구분한다. cache/DRAM counter로 검증하기 전에는 목표 계층과 물리 회로 에너지를 동일시하지 않는다.

[실험 설계 HTML](docs/experiment-design.html)은 treatment/reference 도식, 약 90 MHz clock coverage, 실제 plan·summary JSON의 로컬 뷰어를 제공한다. 서버 업로드 없이 사용할 수 있으며 문서 도식에는 실측 전력곡선이 없다.

## 제공 기능

| 기능 | 구현 |
|---|---|
| Tensor | FP16 입력·FP32 누산 WMMA 반복, 독립 accumulator sweep, cuBLAS dense GEMM 비교 |
| L1·L2·HBM | `.ca`/`.cg` load, working set·grid·thread·stride·주소 offset·read/copy sweep |
| L2 locality | 의존 pointer chase의 SM별 cycle/access와 offset 변화; near/far 확정은 별도 evidence 필요 |
| 전력 측정 | capability 기반 NVML 평균/현재/누적에너지, 지원되는 memory scope, raw timestamps·오류 |
| 시간·기준 | 같은 context·버퍼·clock policy의 전후 idle 및 AB/BA paired active reference, arm별 warmup·완료 epoch의 정렬 적분 |
| 클럭·DVFS | 지원 pair 안의 약 90 MHz graphics grid, 정확한 1110 MHz·advertised default·현재 정책 reference coverage, 요청/실제 클럭·복원 |
| 분석 | GPU UUID별 전체·idle 증가분·paired-reference 단가 최소와 이산 근접 측정점·bootstrap 구간; 주파수별 활용 조건과 전체 최고 성능 제약을 분리 |
| 모델 | 명시적 활동률 특징, rank/condition 검사, 다른 GPU·클럭 층 분리, mixed holdout 검증 |
| 검증 | Nsight Compute counter를 자동 판정하고 분석에 반영; `pass`/`fail`/`inconclusive` 및 근거 보존 |

**A100 GA100의 `nvmlDeviceGetPowerUsage`는 현재 전력이고 H100은 약 1초 평균이다.** H100 이름만 보고 HBM 별도 센서가 지원된다고 가정하지 않는다. 지원 scope와 실패 상태를 실제 장치에서 확인한다. [공식 자료](docs/sources.md)

## 설치와 빌드

Linux, Python 3.10 이상, CMake 3.22 이상, NVIDIA driver와 CUDA Toolkit이 필요하다. 세 세대를 같은 코드로 비교하려면 **CUDA 12.x**를 사용한다. CUDA 13.0은 V100/Volta의 offline compilation과 library support를 제거했다.

NCU도 세대 지원을 맞춰야 한다. **V100·A100·H100 공통 profiling에는 Nsight Compute 2025.2.x처럼 GV100을 지원하는 버전을 사용한다.** Nsight Compute 2025.3부터 Volta 지원이 제거되어 최신 NCU만 설치하면 V100 검증이 실행되지 않는다. `profile`/`validate-run --ncu /설치경로/ncu`로 실제 사용할 executable을 지정하고 driver 요구사항을 확인한다. [공식 버전 지원 자료](docs/sources.md)

```bash
python -m pip install -e .
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="70;80;90"
cmake --build build -j
python -m powermodeling --help
```

CUDA compiler를 별도 경로로 지정해야 하면 CMake에 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`를 추가한다. Python 분석만 사용할 때는 CUDA 빌드가 필요하지 않다. GPU 측정에는 `build/powerbench`가 필요하다.

## 첫 실험

먼저 장치를 조회하고 기본 DVFS 상태의 smoke sweep으로 실행·센서·결과 형식을 확인한다. smoke도 평균 센서 때문에 수초씩 실행하며, 포화 조건을 전수 탐색하는 용도는 아니다.

```bash
python -m powermodeling discover --device 0 --bench build/powerbench
python -m powermodeling plan --config configs/smoke.json --device 0 --bench build/powerbench --output smoke-plan.json
python -m powermodeling run --plan smoke-plan.json --device 0 --bench build/powerbench --output results/smoke
python -m powermodeling analyze --input results/smoke --output results/smoke-report
```

고정 클럭의 처리량·효율 sweep은 다음과 같다. `plan`이 지원 clock pair를 탐색하고 실행 수와 최소 예상 시간을 보여 준다. 실제 시간에는 메모리 준비와 clock settling 등이 추가된다.

```bash
python -m powermodeling plan --config configs/saturation.json --device 0 --bench build/powerbench --output saturation-plan.json
python -m powermodeling run --plan saturation-plan.json --device 0 --bench build/powerbench --output results/saturation --apply-clocks --clock-method applications
python -m powermodeling analyze --input results/saturation --output results/saturation-report
```

고정 클럭 변경은 해당 GPU를 독점적으로 사용하고 필요한 권한이 있을 때 수행한다. `applications` 방식은 기존 application clock pair를 읽고 복원한다. 지원되지 않거나 권한이 없으면 실패가 표시된다. GPU driver가 요청 클럭을 그대로 유지한다는 보장은 없으며 분석은 actual clock·temperature·throttling을 확인한다.

`locked` 방식에서는 이전 lock 정책을 일반적으로 읽어 복원할 수 없으므로 기존 범위를 `--locked-restore` JSON으로 제공하거나, 실행 전 unlocked 상태가 확인된 소유 GPU에 `--clock-reset-on-exit`을 명시한다. 이 옵션은 실행 뒤 lock을 기본 상태로 해제한다. 기존 정책이 있는 공유 GPU에서 추측하여 사용하지 않는다. 도구는 power limit·persistence·MIG 상태를 자동 변경하지 않는다.

완료된 실험을 이어 실행하려면 같은 plan/output에 `--resume`을 추가한다. 실패한 측정은 raw 로그를 남기며 성공한 결과로 처리되지 않는다. `--limit 3`은 일부 trial의 실행 점검에 사용할 수 있지만 repeat나 sweep가 불완전하면 효율 최적값을 확정할 수 없다.

## sweep 구성

| 설정 파일 | 목적 |
|---|---|
| [configs/smoke.json](configs/smoke.json) | 센서·CUDA·기본 실행을 확인하는 작은 DVFS sweep |
| [configs/saturation.json](configs/saturation.json) | warp/block·working set·accumulator·GEMM 크기와 고정 클럭 탐색 |
| [configs/dvfs.json](configs/dvfs.json) | memory×SM clock 도메인의 bandwidth plateau와 효율 탐색 |
| [configs/locality.json](configs/locality.json) | L2 latency/stride/주소 offset/실행 SM 진단; 물리 near/far labels는 자동 부여하지 않음 |

JSON의 `clock_pairs`로 실측 장치가 지원하는 `(graphics_mhz, memory_mhz)`를 명시할 수 있다. **`configs/dvfs.json`은 `graphics_step_mhz: 90`으로 모든 광고된 memory domain에서 약 90 MHz graphics grid를 만든다.** 이전 quantile 선택은 이 간격을 보장하지 않았다. `saturation.json`과 `locality.json`의 quantile은 geometry/latency 탐색 단계이며 90 MHz frequency sweep로 표시하지 않는다. 지원 범위 끝점, exact 1110 MHz(지원 domain), advertised default 고정 pair와 무설정 current-policy reference를 포함하고, 미지원/미조회 사유와 실제 간격은 `plan.clock_sweep_coverage`에 남긴다. 1110 미지원일 때 근사 주파수로 대체하지 않는다. current-policy reference는 이전 applications/locked 정책이 있을 수 있어 factory default라고 확정하지 않는다. `sm_count`, `l2_bytes`, `total_memory_bytes`를 쓰는 숫자 표현식은 장치의 조회값으로 해석된다. `blocks = sm_count × 2`는 작업량 지정이며 정확히 각 SM에 2 blocks를 배치하는 명령이 아니다.

DVFS 설정의 geometry는 예제 시작점이다. saturation 결과에서 확인한 block/thread·working set을 `configs/dvfs.json`에 반영한 뒤 주파수 도메인을 검토한다. 전체 격자는 수시간 걸릴 수 있으므로 plan의 trial 수와 예상 시간을 확인한다.

HBM stride 실험은 stride가 커질 때 전체 할당 크기도 늘리고 순환 접근의 가능한 sector footprint를 검사한다. worker는 한 launch가 유한한 iteration 동안 실제로 방문할 수 있는 footprint와 전체 stride cycle의 footprint를 구분한다. 짧은 launch가 같은 작은 주소 집합을 반복하면 큰 할당이어도 cache 실험이 될 수 있으므로 DRAM counter가 필요하다. write/copy는 thread 간 주소 소유권이 겹치지 않도록 유효 크기를 조정하고 실제 사용 크기를 결과에 기록한다.

기본 custom Tensor·L1·L2·HBM 에너지 실험은 초기 target warmup 3초, 전후 idle 각각 6초, reference/treatment 각 arm warmup 3초·측정 12초, 50 ms polling, 기본 4회 반복이다. 각 arm은 최소 10초 이상 측정하며, 완전히 균형 잡힌 AB/BA 순서는 짝수 4회 이상 반복을 권장한다. odd 반복은 순서 imbalance를 진단에 남기며 paired 최적점에는 동일한 유효 AB/BA 수를 요구한다. 기본 cuBLAS GEMM은 대응 geometry가 불명확하여 unpaired이며 전체·idle 기준으로 읽는다. 분석은 각 active arm 양 끝 2초 및 idle 양 끝 1초를 제외한다. 50 ms polling이 센서의 50 ms 갱신을 뜻하지 않는다. config에서 실험 시간을 늘릴 수 있다.

## Treatment·idle·active reference

Treatment는 측정하려는 대상 작업이다. `paired_reference: true`인 custom workload는 같은 worker process·context·할당·clock policy에서 issue-loop `control`을 AB/BA 순서로 짝지어 실행한다. grid/thread·loop·SM filter·batch 설정과 actual clocks·온도·cap·간섭을 검사하며, 각 arm의 에너지와 완료 count를 독립적으로 정렬한다. control도 정수 연산·제어·launch·store 전력을 쓰므로 두 arm의 차이는 operational contrast다.

전후 idle는 active 시점에 보간하고 drift·actual clock·온도를 검사한다. baseline가 실패해도 treatment 전체 에너지의 품질 판정은 보존한다. 미승인 차감값과 음의 대비는 진단값으로 남기지만 승인된 최적점 후보로 사용하지 않는다. 전체 에너지, 승인된 idle 증가분, 승인된 paired 대비를 서로 대체하지 않으며 순수 static/dynamic·회로 에너지로 이름 붙이지 않는다. 이전 plan/결과와 새 paired protocol을 같은 repeat로 합치지 않는다.

## 결과 읽기

각 trial의 raw JSON과 실행 plan이 결과 폴더에 저장된다. 분석 폴더의 `trials.csv`와 `summary.json`에서 측정값과 탈락 이유를 확인한다.

| 값 | 해석 |
|---|---|
| `board_power_w` | 측정 구간의 NVML GPU scope 평균 전력 |
| `idle_power_w` | 전후 idle 구간으로 보간한 기준 전력 |
| `incremental_power_w` | 전체 GPU 전력의 기준 대비 증가분; 순수 block dynamic이 아님 |
| `total_pj_per_flop`, `total_pj_per_logical_bit` | 전체 GPU 단가. FMA=2 FLOP, logical bit=요청 byte×8 |
| `operational_idle_increment_pj_per_flop`, `operational_idle_increment_pj_per_logical_bit` | 전후 idle 대비 증가분 단가; 승인 여부와 함께 읽음 |
| `paired_active_reference_pj_per_flop`, `paired_active_reference_pj_per_logical_bit` | 같은 process의 짝지은 control 대비 단가; 물리 component isolation이 아님 |
| `baseline_valid`, `baseline_issues`, `operational_idle_increment_eligible` | 전후 idle 품질과 실제 상태 일치에 따라 증가분 최적점 사용을 승인 |
| `paired_active_reference_eligible`, `paired_active_reference_issues` | arm별 geometry·시간 정렬·actual clock·온도·cap·간섭·protocol 검사 |
| `paired_reference_order_counts` | group의 AB/BA 유효 반복 수. 두 순서의 수가 같아야 paired 최적점 승인 |
| 기존 `pj_per_op`, `pj_per_logical_byte` | legacy alias; `metric_aliases`와 실제 FLOP/byte 정의를 함께 읽음 |
| `memory_rail_*` | 지원되는 memory scope의 관측값; 미지원이면 null |
| `tensor_peak_tflops_at_achieved_clock` | 실제 SM 개수·실측 MHz로 계산한 dense FP16 issue ceiling |
| `tensor_utilization_vs_dense_clock_peak` | 측정 TFLOP/s / 해당 클럭의 dense peak |
| `idle_fraction_of_measured_power` | 기준 idle / 측정 부하 전력 |
| `idle_fraction_of_power_limit` | 기준 idle / 설정 power limit |
| `valid`, `issues`, `warnings` | 구간·센서·클럭·온도·간섭에 대한 판정과 이유 |
| `target_verified` | 연결한 NCU(Nsight Compute) evidence를 counter·단위·대상·클럭 조건으로 다시 평가하여 통과했는지 |
| `baseline_clock_matched`, `baseline_temperature_matched` | idle와 active의 actual clock/temperature가 분석 허용범위 안에서 일치하는지 |
| `dynamic_attribution_eligible` | 품질·counter·기준 상태 요건을 통과했는지; 순수 dynamic 분리의 증명은 아님 |
| `profiler_suitability_status` | NCU 적절성 판정: `pass` / `fail` / `inconclusive` |
| `count_energy_time_alignment_exact` | work count와 에너지 적분이 같은 완료된 측정 구간을 사용했는지 |
| `energy_per_work_kind` | 같은 구간의 실제 count 기반 단가인지, 과거 whole-run rate를 이용한 추정인지 |
| `verified_selection_eligible` | 품질·NCU 통과·같은 구간의 정확한 work count가 최적값 선택에 충분한지 |
| `uncontrolled_clock_exploratory_best` | 기본 DVFS 결과의 탐색용 최적값; 고정 클럭 비교와 별도 |
| `within_clock_best`, `cross_clock_best` | 같은 clock/전체 clock 관측 최고 처리량의 95% 이상 조건에서 최저 승인된 증가분 단가 |
| `within_clock_best_total_energy`, `cross_clock_best_total_energy` | 95% 처리량 조건에서 최소 전체 에너지 단가의 설정 |
| `pareto_frontiers` | 더 높은 처리량과 더 낮은 전력으로 동시에 개선할 수 없는 관측 설정 |
| `active_control_associations` | 별도로 실행한 control과의 설명용 연결; same-process paired arm과 다르며 자동 component 차감에 사용하지 않음 |
| `empirical_gpu_energy_optima` | GPU UUID·작업·access·고정 memory MHz·objective별 실제 단가 최솟값. 각 주파수 전체 geometry peak의 기본 95% 이상 요구 |
| `empirical_gpu_overall_energy_optima` | measured graphics·memory domain을 함께 비교한 각 GPU의 실제 최솟값 |
| `near_optimum_support_points`, `uncertainty_overlap_support_points` | 기본 최소 단가 5% 이내 / 95% 구간이 겹치는 실측 support points; 미측정 gap이나 연속 최적 구간을 보장하지 않음 |
| `observed_frequency_pairs_without_eligible_candidate` | 측정은 했으나 검증·활용·baseline 요건으로 최적점 후보를 만들지 못한 frequency pair |
| `verified_target_*` | NCU 통과·정확한 시간 정렬 조건 중 **전체 유효 고정 클럭 sweep 최고 처리량의 95% 이상**을 달성한 결과의 최저 단가; 없으면 winner 없음 |
| `verified_target_coverage` | 전체/검증된 최고 처리량·비율·95% 통과 조건 수·미검증 또는 실패한 peak group을 보고 |

`valid=true`는 기록의 품질 기준을 통과했다는 뜻이며, `target_verified=true`나 물리 블록 isolation의 증명이 아니다. worker는 약 1초 간격의 완료 batch 수와 실제 SM admission 수를 기록하고, 분석은 양 끝을 제외한 완료 구간에서 work count와 에너지를 함께 계산한다. 이 기록이 없는 과거 결과는 지속 처리량이 일정하다는 가정의 추정치로 남기고 검증된 최적값에는 사용하지 않는다. idle와 active의 실제 클럭이나 온도가 다르면 증가분에는 activation·주파수 상태·누설 변화가 섞일 수 있으며 경고가 남는다. 3–4회처럼 적은 반복의 bootstrap 범위는 거칠다. 수치 차이가 작으면 반복과 최소점 주변 지원 주파수 측정을 늘리고 온도·센서·counter evidence를 확인한다. V100·A100·H100의 최적 pJ/bit·pJ/FLOP 주파수는 각 UUID의 결과에서 독립적으로 선택하며 1110 MHz를 최적점으로 미리 지정하지 않는다. physical pJ/bit는 동일 energy-window의 계층 traffic provenance가 없어 현재 withheld이고 NCU replay bytes만으로 단가를 계산하지 않는다.

## NCU를 통한 적절성 판단

권장 순서는 **전력 sweep → 전체 조건의 NCU 검증 → 분석**이다. `validate-run`은 같은 condition의 반복 중 하나를 별도로 profile하고 모든 반복에 판정을 연결한다. 처리한 조건과 남은 조건을 manifest에 남기며 전체 energy trial을 보존한다. 고정 클럭 조건에는 전력 실행과 같은 클럭 적용 옵션을 사용한다.

```bash
python -m powermodeling validate-run --input results/saturation --plan saturation-plan.json --output results/validated --profiles-dir profiles --apply-clocks --clock-method applications
python -m powermodeling analyze --input results/validated --output results/validated-report
```

`--limit-conditions N`은 일부 조건의 실행 점검용이다. 검증 coverage가 제한된 결과에서 전체 sweep의 verified 최적값을 확정하지 않는다. 개별 조건을 확인하거나 기존 evidence를 다시 평가하려면 `plan.json`의 `trial_id`를 사용한다.

```bash
python -m powermodeling profile --plan saturation-plan.json --trial-id TRIAL_ID --device 0 --bench build/powerbench --output profiles --apply-clocks --clock-method applications
python -m powermodeling evaluate-profile --evidence profiles/TRIAL_ID.evidence.json --output profiles/TRIAL_ID.assessment.json
python -m powermodeling attach-verification --input results/saturation --evidence profiles/TRIAL_ID.evidence.json --output results/verified
python -m powermodeling analyze --input results/verified --output results/verified-report
```

`null/null`인 무설정 current-policy trial은 `--apply-clocks` 없이 profile한다. DVFS sweep의 고정 클럭 trial은 전력 실행과 같은 `--apply-clocks` 및 clock method를 사용한다. `locked` 방식의 복원 옵션은 `run`과 같다. `evaluate-profile`은 offline counter 판정이며, 측정 trial의 UUID·binary·effective parameter·클럭 일치까지 확인하는 최종 판정은 연결과 분석 시 수행한다. `--policy policy.json`으로 프로젝트 판정 기준을 바꿀 수 있으며, 평가에 사용한 기준도 결과에 보존된다. counter 이름·지원 범위는 세대 및 Nsight Compute 버전에 따라 다르므로 장치의 metric 목록을 먼저 조회한다.

| 목표 | 자동 판단에 사용하는 근거 | 해석 |
|---|---|---|
| FP16 Tensor / GEMM | Tensor pipe 활동, 실제 SM clock, 의도한 workload의 launch | Tensor 경로 사용 여부; worker의 finite sample·checksum은 sanity 검사이며 수치 정확도 증명과 구분 |
| L1 | L1 hit와 요청 sector, 하위 L2·DRAM traffic | L1이 주된 공급원인지 |
| L2 | L2 hit, 요청 sector와 DRAM traffic | L2가 주된 공급원인지; near/far는 추가 지도 필요 |
| HBM | DRAM read/write byte, L2 요청 sector, L2 hit | 실제 DRAM 이동이 충분한지; logical byte와 physical byte를 구분 |

기본 policy는 L1/L2 read hit ≥95%, L2/HBM read의 L1 hit ≤5%, cache의 하위 traffic ≤logical byte의 10%를 사용한다. HBM read는 L2 hit ≤20%, 요청 방향 DRAM byte ≥logical byte의 75%, DRAM/L2 byte 비율 0.75–1.25를 요구한다. sector inflation 허용 범위는 0.90–8.25, profile actual clock 오차·drift 허용은 3%다. 모든 목표에서 local load/store sector가 0인지 확인한다. 이는 초기 프로젝트 기준이며 실제 장치의 replay 오차·쓰기가 kernel 종료 뒤 writeback되는 현상에 맞춰 검토해야 한다. L2 write/copy는 read hit 정책만으로 적절성을 증명하지 못하므로 현재 `inconclusive`로 남긴다.

판정은 `pass`(통과), `fail`(관측 counter가 조건을 위반), `inconclusive`(필수 counter·clock evidence 누락 또는 불명확) 중 하나다. sector는 32 bytes로 변환하고 metric 단위를 확인한다. 사용자가 JSON의 기존 `memory_target_verified` 또는 `tensor_instructions_verified`를 `true`로 바꾸어도 counter 검증을 대신하지 못한다. 연결할 때와 분석할 때 evidence를 다시 평가한다.

profile 대상이 매우 짧으면 active 구간에 NVML memory-clock sample이 없어 `inconclusive`가 될 수 있다. 이때 verified 표시를 수동으로 바꾸지 말고 같은 workload의 iteration/batch 설정을 검토하여 plan을 다시 생성한 뒤 전력과 profile을 같은 조건으로 재실행한다. counter·클럭 누락과 명확한 workload 실패는 별도 이유로 기록된다.

실패·미확정 조건의 raw 에너지 측정은 보존하고 `verified_target_*` 최적값 선정에서 제외한다. 95% 기준의 분모는 미검증·target 실패 후보를 포함한 **전체 유효 고정 클럭 sweep의 최고 처리량**이다. 검증된 후보들만으로 최고값을 낮추지 않는다. 해당 기준에 도달하는 검증 후보가 없으면 winner를 비워 두고 `verified_target_coverage`에 이유를 남긴다. 연결 대상이 아닌 trial도 새 결과 디렉터리에 그대로 보존한다. `pass`는 목표 경로의 적절성 판단이며 대역폭 포화·높은 처리량의 최소 에너지·순수 물리 회로 에너지 분리는 각각 별도 판단이다.

profiling은 전력 측정이 아니다. CUDA profiler start/stop 구간에 실제 대상 workload만 넣고 setup·initialization·warmup은 제외한다. cuBLAS의 내부 kernel 이름을 추측하지 않는다. 생성 명령은 `--profile-from-start off --cache-control none --clock-control none --replay-mode application --print-units base`를 사용하며, `--log-file`로 NCU CSV를 worker JSON과 분리한다. cache flushing을 껐다고 residency가 보장되지는 않으므로 실제 hit와 DRAM bytes를 확인한다. [공식 자료](docs/sources.md)

L2 Fabric counter는 `--extra-metrics`로 추가할 수 있다. `local-heavy`/`remote-heavy`/`mixed` 분류는 독립적으로 검증한 SM·주소·fabric 지도를 요구하며 기본값은 `unclassified`다. counter 통과만으로 해당 지도를 자동 생성하지 않는다.

산점도와 클럭별 비교 이미지를 함께 만들려면 `python -m pip install -e ".[plots]"` 후 analyze에 `--plots`를 추가한다.

## 모델 fitting

한 GPU, 한 고정 클럭 층에서 측정한 활동률과 `incremental_power_w`를 명시한 JSON 배열을 준비한다. 각 row에는 선택한 feature가 모두 있어야 하고 미측정 값을 0으로 채우지 않는다. `split: "validation"` 또는 `"test"`인 row는 fitting에서 제외하여 검증에 쓴다.

| row 항목 | 필요한 내용 |
|---|---|
| `gpu_uuid`, `config` | 실제 UUID와 양수 `graphics_clock_mhz`, `memory_clock_mhz` |
| `model_features`, `incremental_power_w` | 선택한 각 활동률과 같은 측정 구간의 기준 대비 전력; feature를 row 최상위에 둘 수도 있음 |
| `feature_units` | Tensor의 `"TFLOP/s"`, byte rate의 `"GB/s"`(10⁹ byte/s) |
| `feature_provenance` | 각 feature의 `source`; byte feature는 `traffic_kind: "logical"` 또는 `"physical"`도 필수 |
| `power_provenance` | power의 측정·idle subtraction 정의를 담은 문자열 또는 `source` 객체 |
| `trial_id` / `measurement_id` | mixed holdout의 독립성을 판단하는 실제 측정 ID; calibration과 중복되지 않아야 함 |
| `split` | calibration과 별도로 수집한 `validation`/`test` row 구분 |

예를 들어 physical HBM rate의 source에는 사용한 DRAM read/write counter와 rate의 시간 기준을 적는다. 별도 NCU kernel busy time의 rate를 지속 전력 실행의 wall time rate로 그대로 취급하지 않는다. 단위·logical/physical 정의가 다른 row는 같은 모델에 섞지 않는다. 분석 결과 row를 직접 fitting할 때는 새로 평가한 NCU 통과와 정확한 work/energy 시간 정렬도 요구한다.

```bash
python -m powermodeling fit --input feature-rows.json --features tensor_tflops,l1_gbps,l2_gbps,hbm_gbps --output model.json
```

입력·식별 요건을 충족하지 못한 fitting은 `status: "rejected"`와 이유를 JSON에 보존하고 exit code 2를 반환한다.

현재 harness는 Tensor와 memory 활동률을 함께 측정하는 mixed-workload calibration을 자동 생성하지 않는다. 해당 workload를 별도로 실행하고 실제 활동률 counters로 feature rows를 준비해야 한다. 예측 가능한 모델의 완성 여부는 이 데이터와 독립적인 holdout 검증에 달려 있다.

Tensor feature의 단위는 TFLOP/s, byte feature의 단위는 GB/s이다. fitting 계수의 단위는 각각 W/(TFLOP/s), W/(GB/s)이다. 같은 숫자는 전자의 경우 pJ/FLOP, 후자의 경우 nJ/byte로 변환된다. 모델은 intercept, residual, rank와 condition, 검증 오차를 보고한다. mixed 예측은 통과한 독립 holdout feature들의 convex hull, 즉 실제 검증한 혼합 조건을 가중 평균해서 만들 수 있는 범위로 제한한다. holdout 하나만 통과하면 그 혼합 vector만 검증된 것이며 임의의 다른 혼합으로 확대하지 않는다.

Tensor의 클럭별 ceiling은 `SM 개수 × 실제 SM MHz × FLOP/SM/cycle × 10^-6` TFLOP/s로 계산한다. 공통 dense FP16·FP32 누산의 FLOP/SM/cycle은 V100 1,024, A100 2,048, H100 4,096을 사용한다. clock-specific issue ceiling이며, 공통 WMMA가 이 수치를 모두 달성한다는 보장은 없다.

400 W/312 TFLOPS만으로 idle 몫을 구할 수는 없다. TDP는 상한 사양이며 실제 부하 전력과 다르고, 312 TFLOPS는 특정 A100 SKU의 dense FP16 Tensor peak다. 측정한 idle, 실제 power, 실제 clock의 peak ceiling을 사용하여 idle 비율과 throughput utilization을 평가한다.

## 검증 및 상세 설계

```bash
python -m unittest discover -s tests -v
```

CPU 테스트는 데이터 분석·모델 식별·plan·NVML mock·클럭 복원을 검증한다. CUDA 컴파일, actual GPU 실행, cache attribution과 측정 정확도는 V100/A100/H100 장비에서 확인해야 한다.

- [실험 설계: static/dynamic 기준, DVFS, hierarchy, near/far, fairness](docs/experiment-design.ko.md)
- [전체 구현 자가점검: 발견 사항·수정·검증·남은 실측](docs/self-audit.ko.md)
- [NVIDIA 공식 출처 및 검증이 필요한 주장](docs/sources.md)
