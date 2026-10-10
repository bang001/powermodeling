# 전체 구현 자가점검

검토 기준일: 2026-10-10 (Asia/Seoul), 이전 전체 검토는 2026-10-05. 대상은 FP16 Tensor·L1·L2·HBM workload, 전력 수집, 클럭 제어, NCU 판정, 결과 분석, 모델 fitting과 CLI이다.

**NCU counter로 목표 경로의 적절성을 평가하고 최적값 선정에 반영하도록 수정했다.** 수동 verified 표시, 전체 실행의 처리량과 잘라낸 전력 구간의 혼합, 기본 DVFS 결과의 고정 클럭 비교 등 결과를 과신하게 만들 수 있는 경로를 함께 점검했다. 실제 V100/A100/H100 장비는 이 검토 환경에 없어 hardware 측정·NCU 실행의 적절성까지 통과했다고 주장하지 않는다.

## SXM·Treatment·주파수·최적점 재점검

| 요청 항목 | 재점검 결과 | 구현·문서 반영 | 남은 실장비 확인 |
|---|---|---|---|
| 모두 SXM | SXM과 HBM은 form factor와 memory 기술로 서로 다른 분류; 이전 family/compute capability만으로 SKU를 충분히 제한하지 않았음 | SXM 선언/식별 근거와 GPU family 검증을 device에 보존. 이름상 PCIe·다른 GPU family는 거부하며 모호한 이름은 명시적 확인 필요 | 실제 V100/A100/H100 모듈 이름·용량·SM 수·power limit |
| treatment와 idle | 이전 구현은 이미 전후 idle 보간·drift 검사를 했으므로 단일 idle snapshot의 무조건 차감은 아니었음. 그러나 same-process paired active reference는 없었음 | custom Tensor/L1/L2/HBM에 arm별 warmup·지속 측정·완료 epoch, 같은 context/버퍼/clock policy, seeded AB/BA pairing 추가. 전체·idle 증가분·paired 대비의 승인 범위를 분리 | control과 target actual clock·온도·cap·간섭·SM admission·repeat order |
| 약 90 MHz | 이전 graphics quantile은 약 90 MHz interval을 보장하지 않았음 | saturation·DVFS·locality 모두 모든 광고 memory domain에서 `graphics_step_mhz:90`, 지원값 매핑·끝점·실제 간격을 기록 | 실제 장치의 지원 list, 요청/실제 MHz, clock-domain plateau |
| default·1110 | 기존 relative quantile만으로 해당 기준점 포함이 보장되지 않았음 | exact 1110 지원 pair만 포함, 지원 불가는 not-applicable 사유. advertised-default fixed pair와 무설정 incoming-policy reference를 분리; default 미확정 energy-sweep plan은 실행 전에 차단; 기록된 native 지원 목록에서 필수 90 MHz grid를 재계산하고 geometry·treatment-design별 실제 clock trial coverage 재검사 | default API 지원·clock method, 이전 lock 정책 여부. null pair를 factory default로 단정하지 않음 |
| GPU별 최적 pJ/bit·pJ/FLOP | global-near-peak만으로 선택하면 각 주파수에서 활용된 에너지 최적점을 놓칠 수 있었음 | 각 UUID의 clock pair별 최소 2 검증 resource geometry 비교. HBM이 아닌 workload는 전체 유효 관측 peak의 95%. HBM은 에너지 구간 logical byte/s가 해당 memory clock 이론 bandwidth의 80% 이상. 단일 geometry는 별도 exploratory 진단. total/idle/paired objective 분리 | 모든 후보 NCU·품질·활용 승인, 최소점 주변 추가 측정·반복 불확실성 |
| visualization 문서 | 설계 내용을 한 번에 읽을 독립 HTML이 없었음 | [experiment-design.html](experiment-design.html)에 계층/단위, paired timeline, supported-grid 도식, 판정 순서, 로컬 plan/summary 뷰어 추가 | HTML에서 표시되는 숫자는 사용자 결과 provenance를 확인해야 하며 문서에 측정 숫자는 없음 |

baseline가 실패해도 treatment 자체의 유효한 전체 에너지는 보존한다. 차감 objective에는 별도의 상태 일치 요건을 적용하고, 음의 관측 contrast는 부호를 보존한 진단값으로 남기되 최적값을 만들지 않는다. 기본 config는 4회 반복이며 paired 최적점에는 AB/BA 유효 반복 수가 같아야 한다. odd 반복은 imbalance 진단을 보존하고 paired 최적점으로 승인하지 않는다. issue-loop control은 다른 명령·memory·occupancy 경로를 사용하므로 이 대비를 순수 component dynamic이라고 주장하지 않는다.

## 중요한 발견과 수정

높음은 에너지 단가·목표 계층·재현성·GPU 정책에 영향을 주는 결함이며, 중간은 진단·범위 해석을 약화하는 결함이다.

| 우선순위 | 이전 문제 | 반영한 수정 | 검증 및 남은 조건 |
|---|---|---|---|
| 높음 | 수동 `*_verified: true`를 신뢰하여 잘못된 계층을 verified 최적값으로 선택할 수 있었음 | raw NCU rows·단위·대상·actual clocks로 재평가; `pass`/`fail`/`inconclusive`와 개별 근거 저장 | 반례 counter·필수 metric 누락·수동 bool 조작 테스트; 실제 장치별 counter 지원 확인 필요 |
| 높음 | 전체 실행의 rate를 trimmed 전력과 조합하면 전이 구간이 다른 에너지 단가를 만들 수 있었음 | 완료 epoch의 정확한 work count와 같은 시작/끝의 에너지 적분 사용; 과거 결과는 추정치로 표시 | 중간 처리량 변화·epoch 누락/겹침/총합 불일치 테스트; sensor 평균창은 여전히 실측 확인 필요 |
| 높음 | 큰 HBM 할당과 전체 stride cycle만 보고 실제 짧은 launch의 footprint를 과대평가할 수 있었음 | finite-launch footprint·전체 cycle footprint·정확값/상한 구분; cache에 머물 가능성 표시 | 독립 주소 열거 30,720 finite-footprint 조건 확인; 실제 DRAM counter 통과가 필요 |
| 높음 | write/copy에서 wrap한 주소를 여러 thread가 동시에 쓸 수 있었음 | `blocks×threads×stride`의 온전한 tile로 유효 footprint 조정; 요청/실제 크기 기록 | 독립 주소 열거 9,543 ownership 조건 확인; 장비에서 sanitizer 확인 필요 |
| 높음 | setup/warmup·cuBLAS initialization이 profile에 섞이거나 실패한 profile의 오래된 evidence를 재사용할 수 있었음 | CUDA profiler 구간에서 대상만 계측, 별도 CSV·실패 파일 보존, stale evidence overwrite 거부 | 생성 명령·region·CSV·profile 실패/복원·중복 ID 테스트; 실제 launch는 장비 확인 |
| 높음 | 최신 NCU가 V100까지 지원한다고 가정하면 삼자 비교의 검증이 실행되지 않음 | NCU 2025.3의 Volta 지원 제거와 2025.2.x의 GV100/A100/H100 공통 지원 명시; `--ncu`로 버전 선택 | NVIDIA 공식 버전 자료 확인; 설치 executable·driver 요구사항은 장비에서 확인 |
| 높음 | clock setter가 일부 변경 후 실패하거나 복원 실패해도 정책이 남을 가능성 | 변경 시도를 기록하고 실패 경로에서도 복원; 복원 evidence와 실패를 명시 | 일부 setter 실패·복원 실패 mock 테스트; 지원/권한은 장비에 따라 다름 |
| 높음 | resume 시 다른 clock method·복원 정책·sampling interval로 같은 결과를 이어 쓸 수 있었음 | saved plan·GPU UUID·binary·condition·실행 clock policy·polling interval을 대조 | resume policy 충돌 테스트; 동일 드라이버/온도 재현성까지 보장하지는 않음 |
| 높음 | process inventory 누락·sampler 실패·malformed worker event로 간섭 또는 raw 실패 근거가 감춰질 수 있었음 | compute/graphics/MPS inventory·active/pending MIG 확인; parse/phase/UUID/sampler/interrupt 실패의 stdout·events·samples를 raw에 보존 | `test_telemetry.py`·`test_clocks.py`·`test_runner_audit.py`의 실패·간섭·중단 반례 테스트; blocked native NVML call는 강제 중단하지 않고 bounded stop/lock 실패로 표시 |
| 높음 | model 입력의 단위·실측 근거가 없고 좁은 holdout 하나를 넓은 mixed 예측으로 일반화할 수 있었음 | 단위·feature/power provenance·GPU/고정 클럭 검증과 독립 mixed holdout 범위 제한 | 누락 provenance·단위 혼합·중복 holdout·calibration 범위 밖 예측 테스트; 실제 mixed calibration은 별도 필요 |
| 중간 | seed·실제 DVFS 상태가 다른 결과를 repeat로 합치거나 기본 DVFS를 공정한 고정 클럭 최소값으로 해석할 수 있었음 | seed·binary·clock policy 보존, uncontrolled clock의 탐색용 결과 분리 | grouping·uncontrolled selection 테스트 |
| 중간 | NVML field 응답의 ID/scope와 sensor timestamp 시계를 충분히 구분하지 못함 | `fieldId`/`scopeId` 검증, `timestamp_clock=unix_epoch_microseconds`와 host monotonic/realtime query bracket 분리 | field identity 불일치·timestamp schema 테스트 |

## 전체 구현 범위별 판정

| 구성 | 코드에서 점검한 기준 | 현재 해석과 한계 |
|---|---|---|
| FP16 Tensor WMMA | FP16 입력·FP32 누산, WMMA `m16n16k16` register reuse, 독립 accumulator, `FMA=2 FLOP`, output sanity | 공통 WMMA API 경로의 에너지. raw `mma.m16n8k16`이나 shared operand 경로가 아니고 SASS shape와 같다고 단정하지 않음. H100 WGMMA/TMA를 대표하지 않음 |
| cuBLAS GEMM | datatype·누산, `2MNK`, library version, deterministic profile region | Tensor와 메모리 경로가 함께 동작; 순수 Tensor rail 에너지로 이름 붙이지 않음 |
| L1 | `.ca`, block별 working set, global-load hit·하위 traffic | shared/L1 carveout와 residency에 따라 실제 용량 변화; NCU 확인 필요 |
| L2 | `.cg`, L2 hit와 DRAM 이동, sector→byte 변환 | SM·interconnect·L2 비용 포함; hit만으로 near/far 판정하지 않음 |
| HBM | `.cg`, finite footprint, read/write/copy 실제 크기, DRAM byte | controller·L2·SM 공급 비용 포함; 압축과 coalescing은 counter로 확인 |
| GPC·SM·warp·block | warp=32, SM admission count, `%smid`, block/thread sweep | SM filter는 배치된 block의 admission 제어이며 물리 SM enable/disable·GPC 고정 API가 아님 |
| L2 locality | 의존 pointer chase의 SM/offset latency·독립 fabric 지도 | 후보 지도만 생성; bandwidth saturation과 near/far energy 검증은 별도 |
| 클럭·DVFS | 지원 pair·actual clock·throttling·변경 실패·복원·resume policy | MHz가 같아도 전압·공정·power cap·온도가 같다는 보장은 없음 |
| NVML 센서 | UUID 대응, capability probe, null/error, power/energy/scope timestamp | A100 GA100 현재 전력과 H100 평균 전력 차이 기록; rail 범위는 실측 지원 확인 |
| 전후 idle·paired 기준 | 전후 보간, baseline와 전체 품질 분리, 같은 process AB/BA arm·actual clock/온도/cap·조회 가능한 pstate/enforced cap·geometry·epoch 확인 | idle 증가분과 paired operational contrast를 분리; control이 완전한 counterfactual이 아니며 순수 static/dynamic 분리의 증명은 아님 |
| 시간·적분·처리량 | monotonic phase, 완료 epoch, exact count, matched energy window | CUDA event elapsed는 launch gap 포함; power sensor의 지연·평균창은 남음 |
| NCU 적절성 | 필수 metric·단위·valid range·target launch·clocks·identity·local-memory traffic | 판정 threshold는 프로젝트 정책. profile memory clock은 샘플 전부가 양의 유한값이고 해당 필드 오류가 없어야 하며, graphics `SM_HZ` 대체는 모든 kernel의 유효값이 필요하다. 불완전하면 `inconclusive`. `raw_samples`/`valid_samples`와 `kernel_samples`/`valid_kernel_samples`를 남긴다. L2 write/copy residency는 미확정 |
| 최적값 분석 | UUID별 clock pair 최소 2 검증 resource geometry·HBM logical 80% 또는 그 외 관측 peak 95%·검증·정렬·total/idle/paired 단가, discrete support point·CI·Pareto | 주파수별 efficiency와 전체 clock 최고 성능 제약을 분리. HBM 95%는 추가 탈락 조건이 아니다. reference 미승인/단일 geometry는 승인 winner 없음. 2개 비교가 실제 포화 증명은 아님 |
| 모델 | explicit measured feature·단위/provenance·rank/condition·holdout 상태·온도 폭·`execution_scope` | `fitted`는 calibration 성공. holdout `fail`은 예측을 거부. 온도 5°C 폭과 요청 clock context는 별도. 혼합 calibration 없는 계수 단순 합산 금지 |
| CLI·provenance | validate-run/profile/evaluate/attach/analyze, 원본 보존, UUID·binary·condition | batch validation checkpoint·coverage 기록; 실패 evidence도 남김; binary hash·toolchain·PID로 결과 추적 |
| 빌드·CI·문서 | Python 회귀 테스트와 CUDA 12의 `sm_70;sm_80;sm_90` 컴파일 | 컴파일/CPU 테스트가 실제 센서·cache 적절성 검증을 대신하지 않음 |

## 2026-10-10 Grok·Codex 교차 검토와 구현 변경

Grok과 Codex는 같은 합성 반례 네 가지를 각각 실행한 뒤, 승인 범위와 호환성에 대해 두 차례 토론했다. Codex는 `powermodeling/model.py`, `validation.py`, `cli.py`와 회귀 테스트를 수정했고, Grok은 실험 정의와 변경 내용을 문서에 반영했다. 아래 필드와 동작은 현재 구현에 반영되어 있다. 실제 GPU 측정값이나 하드웨어 검증 결과는 포함하지 않는다.

수정 전에는 단일 활동 holdout 상대오차 0.91도 `predict_power`를 통과했다. 허용치는 0.10이었고, 오차 비교는 mixed holdout이 있을 때만 `additive_validated`에 적용됐다. 30–80°C의 검증된 행 6개는 한 모델에 fit됐다. fit은 GPU·요청 clock·software stratum을 검사하지만 그 값을 모델에 저장하지 않아, 다른 GPU와 다른 요청 clock으로 `predict_power`가 같은 계수를 반환했다. profile memory clock 샘플 11개 중 10개가 실패해도 남은 1개의 중앙값으로 `pass`와 `suitable_verified=true`가 나왔다. 이 값은 모두 합성 입력이며 측정치가 아니다.

토론 후 구현한 동작은 다음과 같다.

- `status="fitted"`는 OLS calibration 성공만 뜻한다. 계수는 남긴다.
- `holdout_validation_status`는 `not_provided`, `pass`, `fail`이다. `fail`이면 단일 활동 예측과 mixed 예측을 모두 거부하고 `fit` CLI는 exit 2다. 실패 진단도 남는다. mixed 실패의 `additive_validated=false`와 convex hull 제한은 유지한다.
- fitting 성공과 예측 승인을 분리했다. Holdout 실패로 진단용 계수까지 폐기하지 않고, 실패한 모델의 예측 사용을 차단한다.
- `temperature_stratum.status`는 `qualified`, `unverified`, `rejected`다. `minimum_c`, `maximum_c`, `median_c`, `maximum_span_c=5`를 저장한다. 분석 행은 중앙값만이 아니라 measure `temperature_c_min`/`temperature_c_max`를 폭에 넣는다. 알려진 온도의 전체 폭이 5°C를 넘거나 일부 행만 온도가 없으면 rejected다. 온도가 모두 없으면 warning과 `unverified`다. 정확한 온도 일치나, 5°C보다 넓은 sweep를 위한 새 모델은 만들지 않는다.
- `execution_scope`는 `gpu_uuid`, `requested_clock_pairs`, `cross_clock_model`, `benchmark_sha256`, `measurement_stratum`, `treatment_design_stratum`이다. clock은 요청값이다.
- `predict_power(model, features, *, context=None, allow_unbound_context=False, allow_extrapolation=False)`는 기본적으로 GPU와 요청 clock context를 요구한다. `features` 안에 그 context가 있으면 그것도 읽는다. cross-clock 모델은 fit 때 MHz 단위의 clock feature를 config 요청값과 대조하고, 예측 때는 그 feature를 쓴다. 온도가 알려진 모델은 현재 온도를 요구하고, calibration·holdout과 예측 온도를 합친 최대−최소가 5°C 이내여야 한다. 저장한 software/power stratum은 context에 제공된 경우 비교하며, GPU·clock 확인만으로 생략된 환경 정보까지 검증했다고 해석하지 않는다.
- context를 생략하면 warning만 남기자는 안은 채택하지 않았다. `allow_unbound_context=True`는 누락 context의 과거 진단 계산만 허용한다. 명시적 mismatch, holdout 실패, mixed hull 거부는 우회하지 않는다. scope가 없는 legacy 모델도 기본은 다시 fit하거나 이 opt-in이 필요하다.
- `profile_actual_clocks`는 NVML의 `raw_samples`/`valid_samples`, kernel의 `kernel_samples`/`valid_kernel_samples`, energy 구간의 `energy_raw_samples`/`energy_valid_samples`와 실제 선택한 `source`를 보존한다. memory clock evidence가 불완전하거나 graphics `SM_HZ` 대체가 kernel 전부에 유효하지 않으면 `inconclusive`다. 모든 kernel에서 `SM_HZ`가 미지원이면 완전한 NVML graphics 관측을 대안으로 사용할 수 있다.

구현 교차 검토에서 Grok은 클럭 `errors`의 빈 payload가 성공으로 취급되고 list 형식에서는 예외가 발생하는 반례를 추가로 찾았다. 오류 키가 존재하면 `{}`·`null`·`false`·`0`도 실패 근거로 보고, 오류 컨테이너가 dict가 아니면 예외 대신 `inconclusive`로 처리하도록 보완했다. Codex는 문자열 `"false"`의 truthiness가 unbound 옵션을 켤 수 있음을 확인해, `allow_unbound_context`와 `allow_extrapolation`이 실제 boolean만 받도록 수정했다. 두 경우 모두 회귀 테스트의 실패를 확인한 뒤 수정했다.

clock feature와 요청 clock이 다른 행을 전체 fitting 실패로 올릴지는 재검토했다. 기존 provenance 검사와 같이 해당 행을 제외하고 `skipped_rows`에 사유를 보존하는 정책을 유지한다. 반환된 calibration 행 수와 skip 사유를 함께 읽어야 한다. `holdout_validation_status=pass`는 상대오차 관문의 통과만 뜻하고, mixed 독립성·calibration 범위·convex hull은 별도 검사다. Cross-clock 모델의 단일 활동 예측은 주파수 feature의 calibration 범위 안에서 보간할 수 있으며, mixed 활동에는 기존 holdout hull 제한이 적용된다. scope가 없는 legacy artifact의 unbound 진단에는 원래 GPU·클럭을 확인할 근거가 없으므로 장치 적용 검증으로 해석하지 않는다.

Tensor shape는 토론 중 정정했다. 한 답변은 `m16n8k16`을 Turing 이후 WMMA shape라고 했다. 이는 raw `mma`와 `wmma`, `k8`과 `k16`을 섞은 것이다. 2026-10-10에 읽은 PTX ISA 9.4 Target ISA Notes는 `.f16` `mma` `.m16n8k8`을 `sm_75` 이상, `.f16` `mma` `.m16n8k16`을 `sm_80` 이상으로 둔다. floating-point `wmma`는 `sm_70` 이상이고 FP16 shape는 `.m16n16k16`, `.m8n32k16`, `.m32n8k16`이다. 공통 커널은 WMMA `m16n16k16` register reuse로 유지한다. 그 shape와 SASS `mma`가 같다는 주장은 하지 않는다. 새 raw MMA나 shared-resident Tensor subsystem은 이번 범위가 아니다.

HBM 선정은 logical byte/s ÷ 이론 bandwidth ≥ 0.80을 유지한다. 다른 replay의 DRAM/logical ≥ 0.75와 이 0.80을 곱한 0.60은 같은 구간의 DRAM 전송률도 아니고 하한도 아니다. DRAM byte/s 80% gate는 추가하지 않는다.

새 반례는 [모델 승인·적용 범위 테스트](../tests/test_model_scope.py)와 [profiler 클럭 coverage 테스트](../tests/test_profile_clock_coverage.py)에 보존했다. 기존 수학 예측 테스트도 일치하는 GPU·요청 clock context를 제공하도록 갱신해, context 누락이 원래의 feature/hull 검사 실패를 가리지 않게 했다. 새 동작은 수정 전 실패를 확인한 뒤 구현했다.

## 재현 가능한 확인

```bash
python -m unittest discover -s tests -v
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="70;80;90"
cmake --build build -j
python -m powermodeling --help
```

2026-10-10 현재 작업 트리에서 `python3 -m unittest discover -s tests`는 **480개 실행, 466개 통과, 14개 skip**으로 끝났다. 이번에 추가한 모델 범위 21개와 클럭 coverage 8개 테스트를 포함한다. Python compileall과 diff-check도 통과했다. Grok은 초기 구현의 관련 90개 테스트와 마지막 오류·옵션 경계 테스트를 별도로 실행했고, 최종 교차 검토에서 남은 Critical/Important가 없다고 판정했다. 이는 생성된 데이터와 NVML/subprocess mock의 CPU 검증이다. 현재 환경에는 `nvcc`, `ncu`, `nvidia-smi`가 없어 새 CUDA 빌드·GPU 실행·센서 정확도는 검증하지 않았다.

2026-10-05 이전 검토에는 clock coverage·resource geometry·baseline 상태 검사와 CLI 사전 차단을 포함한 204개 CPU 테스트, Python compileall 및 diff-check 통과가 기록되어 있다. 이전 CUDA 변경의 다중 아키텍처 빌드 결과는 [PR #1](https://github.com/bang001/powermodeling/pull/1)의 해당 commit CI를 기준으로 확인한다. NCU timeout·중단 시 소유한 process group만 종료하고 부분 출력·실패 evidence·clock 복원 결과를 보존하는 경로도 포함한다. 독립 주소 열거에서 write/copy 소유권 9,543 조건과 finite footprint 30,720 조건을 확인했으며, 이는 주소 수학의 검증이다. CUDA 컴파일 가능성이나 이전 CI는 이번 환경의 실제 GPU 측정 검증을 대신하지 않는다. 테스트 이름·조건은 `tests/`에서 확인할 수 있다.

## HTML 문서 확인 범위

[experiment-design.html](experiment-design.html)은 외부 script·stylesheet·font·image를 요청하지 않는 독립 파일이다. HTML 구조·중복 ID·anchor 연결과 embedded JavaScript 문법을 확인했다. 최신 planner/analyzer가 만든 **synthetic QA fixture**로 local JSON 읽기, 검증된 다중 geometry의 eligible support points 6개, default 미확정 실행 차단, 단일 geometry 탐색 진단, 이전 geometry 근거 미확인 진단, objective 선택, 탭/키보드 이동, invalid·oversize JSON, 안전한 text 삽입을 점검했다. 이 fixture는 GPU 실측 결과가 아니다.

정적 SVG 도식 4개를 개별 렌더하여 도형 배치와 경로를 점검했다. 검토 환경의 한국어 font 지원은 제한되어 있었다. 브라우저 binary가 없는 검토 환경에서는 전체 HTML의 desktop/mobile browser 렌더를 검증하지 못했으므로 이를 완료했다고 주장하지 않는다. 현재 문서의 측정값 없는 초기 화면, 개념도와 결과를 불러온 화면을 구분한다. 로컬 파일 읽기는 브라우저 안에서만 데이터를 처리하고 서버에 전송하지 않는다.

## 실제 GPU에서 남은 확인

| 확인할 것 | 통과 판단의 근거 | 통과 전 내릴 수 없는 결론 |
|---|---|---|
| 세 GPU의 전력 API·rail | discovery의 실제 지원 scope·오류·sensor window, 독립 계측 교차 검증 | H100에서 항상 HBM rail을 읽을 수 있다는 결론 |
| profiler 버전 | V100은 GV100 지원 NCU(예: 2025.2.x), 실제 metric 목록·driver 요구사항 | 최신 NCU가 세 GPU를 모두 profile할 수 있다는 결론 |
| CUDA 수치·메모리 정확성 | Tensor/GEMM의 기준 결과 비교와 memory sanitizer; finite sample/checksum sanity와 구분 | 계산값이 올바르거나 data race가 없다는 실장비 검증 주장 |
| 각 목표 workload의 NCU | 동일 조건의 필수 counter와 실제 클럭·worker identity | L1/L2/HBM label에 해당하는 측정이라는 결론 |
| 고처리량 포화 | geometry·clock sweep의 반복 plateau와 physical traffic | utilization 100% 또는 이론 peak 도달 주장 |
| baseline 상태 | idle/active/reference actual clock·온도·cap·전후 drift·AB/BA 재현성 | 증가분 또는 paired 전력 차이를 순수 dynamic으로 분리했다는 결론 |
| near/far | 반복 SM/주소 latency와 fabric 지도, 같은 조건의 고처리량 재실행 | SM 번호나 주소 절반이 물리 partition이라는 결론 |
| concurrent mixed workload | 독립 활동 변화의 counter feature, held-out mixed 오차·지원 범위 | 네 microbenchmark 계수의 합으로 일반 앱 전력 예측 |
| idle 비율 | 실측 idle W·부하 W·actual clock와 SKU 사양 구분 | 400 W/312 TFLOPS 사양만으로 idle W 역산 |

NCU 적절성 통과 → 높은 지속 처리량 확인 → 최소 에너지 비교 → 모델 calibration/holdout 검증 순서로 결론을 좁힌다. 이 단계들은 서로 대체하지 않는다. 실행 명령은 [README](../README.md), 설계 가정과 시각화는 [experiment-design.ko.md](experiment-design.ko.md)·[experiment-design.html](experiment-design.html), 공식 출처는 [sources.md](sources.md)에 있다.
