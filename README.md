# NVIDIA GPU power modeling

V100·A100·H100에서 **높은 지속 처리량을 유지하면서 가장 작은 연산·전송당 에너지**를 찾기 위한 CUDA/NVML 실험 도구다. FP16 Tensor, L1, L2, HBM의 working set·thread/block·stride·SM/memory clock을 바꾸고, 수초 동안 전력과 처리량을 함께 측정한다.

이 저장소에는 실측 GPU 숫자가 들어 있지 않다. `idle`은 실험 기준 전력이며 순수 누설 전력이 아니다. 보드 전체 전력에서 기준을 뺀 pJ/FLOP·pJ/logical-byte와, 지원되는 memory power scope를 구분해서 보고한다. cache/DRAM counter로 검증하기 전에는 목표 계층과 물리 회로 에너지를 동일시하지 않는다.

## 제공 기능

| 기능 | 구현 |
|---|---|
| Tensor | FP16 입력·FP32 누산 WMMA 반복, 독립 accumulator sweep, cuBLAS dense GEMM 비교 |
| L1·L2·HBM | `.ca`/`.cg` load, working set·grid·thread·stride·주소 offset·read/copy sweep |
| L2 locality | 의존 pointer chase의 SM별 cycle/access와 offset 변화; near/far 확정은 별도 evidence 필요 |
| 전력 측정 | capability 기반 NVML 평균/현재/누적에너지, 지원되는 memory scope, raw timestamps·오류 |
| 시간·기준 | CUDA 컨텍스트를 유지한 전후 idle, warmup, multi-second active 구간, trimmed 적분 |
| 클럭·DVFS | 지원 clock pair 탐색, 요청·실제 클럭 비교, 권한 실패와 throttling 기록, 원래 정책 복원 |
| 분석 | repeat 중앙값·bootstrap 구간, 최대 처리량의 95% 이상에서 최소 전체/증가분 에너지, Pareto 경계, idle 비율 |
| 모델 | 명시적 활동률 특징, rank/condition 검사, 다른 GPU·클럭 층 분리, mixed holdout 검증 |
| 검증 | 별도 Nsight Compute 실행과 검토된 evidence 연결; profiler replay를 전력 결과와 분리 |

**A100 GA100의 `nvmlDeviceGetPowerUsage`는 현재 전력이고 H100은 약 1초 평균이다.** H100 이름만 보고 HBM 별도 센서가 지원된다고 가정하지 않는다. 지원 scope와 실패 상태를 실제 장치에서 확인한다. [공식 자료](docs/sources.md)

## 설치와 빌드

Linux, Python 3.10 이상, CMake 3.22 이상, NVIDIA driver와 CUDA Toolkit이 필요하다. 세 세대를 같은 코드로 비교하려면 **CUDA 12.x**를 사용한다. CUDA 13.0은 V100/Volta의 offline compilation과 library support를 제거했다.

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

JSON의 `clock_pairs`로 실측 장치가 지원하는 `(graphics_mhz, memory_mhz)`를 명시하거나, `clock_sweep`으로 지원 목록의 상대 단계를 선택할 수 있다. `sm_count`, `l2_bytes`, `total_memory_bytes`를 쓰는 숫자 표현식은 장치의 조회값으로 해석된다. `blocks = sm_count × 2`는 작업량 지정이며 정확히 각 SM에 2 blocks를 배치하는 명령이 아니다.

DVFS 설정의 geometry는 예제 시작점이다. saturation 결과에서 확인한 block/thread·working set을 `configs/dvfs.json`에 반영한 뒤 주파수 도메인을 검토한다. 전체 격자는 수시간 걸릴 수 있으므로 plan의 trial 수와 예상 시간을 확인한다.

HBM stride 실험은 stride가 커질 때 전체 할당 크기도 늘려, 순환 접근이 방문할 수 있는 sector footprint가 최소 L2의 4배인지 검사한다. 이 검사는 가능한 주소 집합의 크기이며 실제 DRAM traffic을 보장하지 않는다.

기본 에너지 실험은 측정 12초, warmup 3초, 전후 idle 각각 6초, 50 ms polling, 3회 반복이다. 분석은 active 양 끝 2초 및 idle 양 끝 1초를 제외한다. 50 ms polling이 센서의 50 ms 갱신을 뜻하지 않는다. config에서 실험 시간을 늘릴 수 있다.

## 결과 읽기

각 trial의 raw JSON과 실행 plan이 결과 폴더에 저장된다. 분석 폴더의 `trials.csv`와 `summary.json`에서 측정값과 탈락 이유를 확인한다.

| 값 | 해석 |
|---|---|
| `board_power_w` | 측정 구간의 NVML GPU scope 평균 전력 |
| `idle_power_w` | 전후 idle 구간으로 보간한 기준 전력 |
| `incremental_power_w` | 전체 GPU 전력의 기준 대비 증가분; 순수 block dynamic이 아님 |
| `pj_per_op`, `total_pj_per_op` | 기준 대비 / 전체 pJ/FLOP; Tensor에서 FMA를 2 FLOP으로 계산 |
| `pj_per_logical_byte`, `total_pj_per_logical_byte` | 기준 대비 / 전체 요청 바이트당 에너지; 물리 traffic과 구분 |
| `memory_rail_*` | 지원되는 memory scope의 관측값; 미지원이면 null |
| `tensor_peak_tflops_at_achieved_clock` | 실제 SM 개수·실측 MHz로 계산한 dense FP16 issue ceiling |
| `tensor_utilization_vs_dense_clock_peak` | 측정 TFLOP/s / 해당 클럭의 dense peak |
| `idle_fraction_of_measured_power` | 기준 idle / 측정 부하 전력 |
| `idle_fraction_of_power_limit` | 기준 idle / 설정 power limit |
| `valid`, `issues`, `warnings` | 구간·센서·클럭·온도·간섭에 대한 판정과 이유 |
| `target_verified` | 별도 counter evidence가 검토되어 연결되었는지 |
| `baseline_clock_matched`, `baseline_temperature_matched` | idle와 active의 actual clock/temperature가 분석 허용범위 안에서 일치하는지 |
| `dynamic_attribution_eligible` | 품질·counter·기준 상태 요건을 통과했는지; 순수 dynamic 분리의 증명은 아님 |
| `within_clock_best`, `cross_clock_best` | 관찰된 최고 처리량의 95% 이상 조건에서 최저 증가분 단가 |
| `within_clock_best_total_energy`, `cross_clock_best_total_energy` | 95% 처리량 조건에서 최소 전체 에너지 단가의 설정 |
| `pareto_frontiers` | 더 높은 처리량과 더 낮은 전력으로 동시에 개선할 수 없는 관측 설정 |
| `active_control_associations` | UUID·클럭·grid/thread·SM filter가 같은 control과의 설명용 연결; 자동 전력 차감은 하지 않음 |
| `verified_target_*` | counter evidence가 연결된 결과만의 선택 |

`valid=true`는 기록의 품질 기준을 통과했다는 뜻이며, `target_verified=true`나 물리 블록 isolation의 증명이 아니다. idle와 active의 실제 클럭이나 온도가 다르면 증가분에는 activation·주파수 상태·누설 변화가 섞일 수 있으며 경고가 남는다. 반복 3회의 bootstrap 범위는 거칠다. 수치 차이가 작으면 반복을 늘리고 온도·센서·counter evidence를 확인한다.

## cache·Tensor 검증

전력 sweep 이후 원하는 `trial_id`를 `plan.json`에서 선택해 별도 Nsight Compute 실행을 한다.

```bash
python -m powermodeling profile --plan saturation-plan.json --trial-id TRIAL_ID --device 0 --bench build/powerbench --output profiles
```

profiling은 전력 측정이 아니다. 자동 적용하지 않는 clock 설정을 동일하게 맞추고, profile의 실제 clock·L1/L2 hit·DRAM bytes·Tensor 명령을 확인한다. 필요하면 `ncu --query-metrics --query-metrics-mode all`로 현재 장치의 L2 fabric counter를 조회하여 추가한다. default cache flushing과 clock control은 검증하려는 상태를 바꿀 수 있으므로 생성 명령은 `--cache-control none --clock-control none --replay-mode application`을 쓴다. 짧은 검증 실행은 `--fixed-batches 1 --warmup-batches 1 --batch-launches 1`로 결정적 batch 수를 사용한다. custom kernel은 warmup 한 번을 건너뛰고 측정 launch 하나를 profile한다. warmup 한 번이 cache 전체를 데우기에 충분한지는 counter로 확인해야 한다. 특히 pointer chase에서는 부족할 수 있다.

생성된 evidence JSON은 미검증으로 시작한다. counter 결과와 actual clock을 확인한 뒤 `profile_clocks_verified: true`와 `verification_notes`를 기록한다. 메모리는 `memory_target_verified`, Tensor/GEMM은 `tensor_instructions_verified`를 명시한다. near/far 후보를 명시하려면 `locality_mapping_evidence`도 필요하다. 자동으로 footprint나 warmup 횟수만 보고 검증된 것으로 처리하지 않는다.

```bash
python -m powermodeling attach-verification --input results/saturation --evidence reviewed-manifest.json --output results/verified
python -m powermodeling analyze --input results/verified --output results/verified-report
```

evidence는 GPU UUID와 condition이 일치하는 측정에만 연결할 수 있다. counter 이름·지원 범위는 세대 및 Nsight Compute 버전에 따라 다르다.

산점도와 클럭별 비교 이미지를 함께 만들려면 `python -m pip install -e ".[plots]"` 후 analyze에 `--plots`를 추가한다.

## 모델 fitting

한 GPU, 한 클럭 층에서 측정한 활동률과 `incremental_power_w`를 명시한 JSON 배열을 준비한다. 각 row에는 선택한 feature가 모두 있어야 하고, 미측정 값은 0으로 자동 치환되지 않는다. `split: "validation"` 또는 `"test"`인 row는 fitting에서 제외하여 검증에 쓴다.

```bash
python -m powermodeling fit --input feature-rows.json --features tensor_tflops,l1_gbps,l2_gbps,hbm_gbps --output model.json
```

현재 harness는 Tensor와 memory 활동률을 함께 측정하는 mixed-workload calibration을 자동 생성하지 않는다. 해당 workload를 별도로 실행하고 실제 활동률 counters로 feature rows를 준비해야 한다. 예측 가능한 모델의 완성 여부는 이 데이터와 독립적인 holdout 검증에 달려 있다.

Tensor feature의 단위는 TFLOP/s, byte feature의 단위는 GB/s이다. fitting 계수의 단위는 각각 W/(TFLOP/s), W/(GB/s)이다. 같은 숫자는 전자의 경우 pJ/FLOP, 후자의 경우 nJ/byte로 변환되며, logical byte인지 counter physical byte인지 반드시 데이터의 정의에 기록한다. 모델은 intercept, residual, rank와 condition, 검증 오차를 보고한다. 독립적인 mixed-workload holdout이 통과하기 전에는 네 계수를 합쳐 일반 workload를 예측하지 않는다.

Tensor의 클럭별 ceiling은 `SM 개수 × 실제 SM MHz × FLOP/SM/cycle × 10^-6` TFLOP/s로 계산한다. 공통 dense FP16·FP32 누산의 FLOP/SM/cycle은 V100 1,024, A100 2,048, H100 4,096을 사용한다. clock-specific issue ceiling이며, 공통 WMMA가 이 수치를 모두 달성한다는 보장은 없다.

400 W/312 TFLOPS만으로 idle 몫을 구할 수는 없다. TDP는 상한 사양이며 실제 부하 전력과 다르고, 312 TFLOPS는 특정 A100 SKU의 dense FP16 Tensor peak다. 측정한 idle, 실제 power, 실제 clock의 peak ceiling을 사용하여 idle 비율과 throughput utilization을 평가한다.

## 검증 및 상세 설계

```bash
python -m unittest discover -s tests -v
```

CPU 테스트는 데이터 분석·모델 식별·plan·NVML mock·클럭 복원을 검증한다. CUDA 컴파일, actual GPU 실행, cache attribution과 측정 정확도는 V100/A100/H100 장비에서 확인해야 한다.

- [실험 설계: static/dynamic 기준, DVFS, hierarchy, near/far, fairness](docs/experiment-design.ko.md)
- [NVIDIA 공식 출처 및 검증이 필요한 주장](docs/sources.md)
