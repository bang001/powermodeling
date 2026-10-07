# NVIDIA GPU power modeling

**SXM 모듈의 V100·A100·H100**에서 **충분히 활용한 조건의 실측 pJ/FLOP·pJ/bit·pJ/SFU instruction 최소점과 주변 측정점**을 찾기 위한 CUDA/NVML 실험 도구다. FP16 Tensor, L1, L2, HBM과 **register-resident SFU 기본 명령**의 작업량·thread/block·SM/memory clock을 바꾸고, 수초 동안 전력과 처리량을 함께 측정한다. 비선형 기본 실험은 global 입력 버퍼 없이 EX2·LG2·RCP·RSQ·SQRT·native TANH를 반복한다. 메모리 실험에는 stride·주소 offset sweep도 제공한다.

SXM은 GPU 모듈의 장착 형태이고 HBM은 측정할 메모리 계층이다. 실제 메모리 용량·SKU·SM 수를 이름만으로 확정하지 않는다. 이 저장소에는 실측 GPU 숫자가 들어 있지 않다. 전체 단가, 승인된 전후 idle 증가분, 같은 process에서 짝지은 active-reference 대비를 별도로 보고한다. `idle`은 운영상 기준이며 순수 누설 전력이 아니다. 지원되는 memory power scope도 전체 GPU scope와 구분한다. cache/DRAM counter로 검증하기 전에는 목표 계층과 물리 회로 에너지를 동일시하지 않는다.

[실험 설계 HTML](docs/experiment-design.html)은 treatment/reference 도식, clock coverage, 실제 plan·summary JSON의 로컬 뷰어를 제공한다. 서버 업로드 없이 사용할 수 있으며 문서 도식에는 실측 전력곡선이 없다.

## 제공 기능

| 기능 | 구현 |
|---|---|
| Tensor | FP16 입력·FP32 누산 WMMA 반복, 독립 accumulator sweep, cuBLAS dense GEMM 비교 |
| L1·L2·HBM | `.ca`/`.cg` load, working set·grid·thread·stride·주소 offset·read/copy sweep |
| L2 locality | 의존 pointer chase의 SM별 cycle/access와 offset 변화; near/far 확정은 별도 evidence 필요 |
| Register SFU | EX2·LG2·RCP·RSQ·SQRT·native TANH의 register 반복, SFU를 뺀 control 대비 차분 pJ/scalar instruction |
| 전력 측정 | capability 기반 NVML 평균/현재/누적에너지, 지원되는 memory scope, raw timestamps·오류 |
| 시간·기준 | 같은 context·버퍼·clock policy의 전후 idle 및 AB/BA paired active reference, arm별 warmup·완료 epoch의 정렬 적분 |
| 클럭·DVFS | 900 MHz 이상 지원 pair의 60/90/120 등 가변 간격 graphics grid, 정확한 1110 MHz·advertised default·현재 정책 reference coverage, 요청/실제 클럭·복원 |
| 분석 | GPU UUID별 전체·idle 증가분·paired-reference 단가 최소와 이산 근접 측정점·bootstrap 구간; 주파수별 활용 조건과 전체 최고 성능 제약을 분리 |
| 모델 | 명시적 활동률 특징, rank/condition 검사, 다른 GPU·클럭 층 분리, mixed holdout 검증 |
| 검증 | Nsight Compute counter를 자동 판정하고 분석에 반영; `pass`/`fail`/`inconclusive` 및 근거 보존 |

**A100 GA100의 `nvmlDeviceGetPowerUsage`는 현재 전력이고 H100은 약 1초 평균이다.** H100 이름만 보고 HBM 별도 센서가 지원된다고 가정하지 않는다. 지원 scope와 실패 상태를 실제 장치에서 확인한다. [공식 자료](docs/sources.md)

## 설치와 빌드

Linux, Python 3.10 이상, CMake 3.22 이상, NVIDIA driver와 CUDA Toolkit이 필요하다. **A100은 CUDA 13.0의 `sm_80` 빌드를 지원한다.** V100·A100·H100을 같은 Toolkit으로 비교하려면 CUDA 12.x를 사용한다. CUDA 13.0은 V100/Volta의 offline compilation과 library support를 제거했다.

NCU도 세대 지원을 맞춰야 한다. **V100·A100·H100 공통 profiling에는 Nsight Compute 2025.2.x처럼 GV100을 지원하는 버전을 사용한다.** Nsight Compute 2025.3부터 Volta 지원이 제거되어 최신 NCU만 설치하면 V100 검증이 실행되지 않는다. `profile`/`validate-run --ncu /설치경로/ncu`로 실제 사용할 executable을 지정하고 driver 요구사항을 확인한다. [공식 버전 지원 자료](docs/sources.md)

```bash
python -m pip install -e .
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="70;80;90"
cmake --build build -j
python -m powermodeling --help
```

**A100 + CUDA 13.0**은 CUDA 13.0을 지원하는 Linux R580 이상 driver와 별도 빌드 디렉터리를 사용한다. Toolkit 설치 경로가 다르면 두 경로를 함께 바꾼다.

```bash
cmake -S . -B build-a100-cuda13 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-a100-cuda13 -j
python -m powermodeling discover --device 0 --bench build-a100-cuda13/powerbench
```

아키텍처를 생략한 새 빌드의 기본값은 CUDA 12에서 `70;80;90`, CUDA 13에서 `80;90`이다. `-DCMAKE_CUDA_ARCHITECTURES`와 `CUDAARCHS` 환경변수의 지정값을 우선한다. A100 + CUDA 13의 NCU 검증에는 CUDA 13을 지원하는 Nsight Compute 2025.3 이상을 사용한다. 아래 실행 예시의 `--bench build/powerbench`도 선택한 실행 파일 경로로 바꾼다. 비선형 함수 지침에는 두 빌드 경로를 모두 제공한다. CUDA/cuBLAS 버전이 다른 측정은 별도 분석 층으로 기록한다.

CUDA 12 compiler를 별도 경로로 지정하려면 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`를 추가한다. Python 분석만 사용할 때는 CUDA 빌드가 필요하지 않다.

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

## L1·L2·HBM 이동에너지: coalesced read

기본 이동에너지 실험은 [configs/memory-read.json](configs/memory-read.json)을 사용한다. L1·L2·HBM read만 실행하며 **`stride_words: 1`, `offset_bytes: 0`**을 명시한다. `saturation.json`·`dvfs.json`도 memory 기본값을 명시한다. 기존 CUDA 기본 stride 역시 1이었지만, 이전 검증은 stride가 큰 접근을 coalesced 에너지 후보에서 별도로 제외하지 못했다.

Stride는 **4-byte word 개수**다. 새 config 이름 `stride_words`와 CLI `--stride-words`가 단위를 드러낸다. 기존 `stride_elements`·`--stride-elements`도 같은 word 단위로 지원하며, 두 이름을 동시에 지정하면 오류다. 호환성을 위해 plan의 정규화된 parameter와 raw의 `stride_elements`는 유지하고 raw에 `memory_word_bytes`, `stride_words`, `lane_stride_bytes`도 기록한다.

| Word stride | 인접 lane 주소 간격 | Full warp의 요청 sectors | 유효 payload / sector bytes |
|---:|---:|---:|---:|
| 1 | 4 B | 4 × 32 B | 100% |
| 2 | 8 B | 8 × 32 B | 50% |
| 4 | 16 B | 16 × 32 B | 25% |
| 8 | 32 B | 32 × 32 B | 12.5% |

표는 정렬된 full warp가 서로 다른 주소의 4 B를 하나씩 읽는 경우다. Stride 1이어도 offset 4 B나 잘못 정렬된 L1 slice는 sector 낭비를 만든다. Energy read plan은 stride 1·32 B 정렬된 offset/region·최소 128 B region을 검사한다. L1은 CTA별 slice와 시작 주소도 확인한다. 128 B cache line이 4 sectors라는 사실이 매 접근을 항상 128 B 전송으로 만드는 것은 아니다.

```bash
export POWERBENCH=build/powerbench  # A100/CUDA13: build-a100-cuda13/powerbench
python -m powermodeling plan --config configs/memory-read.json \
  --bench "$POWERBENCH" --device 0 --output results/memory-read-plan.json
python -m powermodeling run --plan results/memory-read-plan.json \
  --bench "$POWERBENCH" --device 0 --output results/memory-read \
  --apply-clocks --clock-method applications
python -m powermodeling validate-run --plan results/memory-read-plan.json \
  --input results/memory-read --output results/memory-read-validated \
  --profiles-dir results/memory-read-profiles --bench "$POWERBENCH" --device 0 \
  --apply-clocks --clock-method applications
python -m powermodeling analyze --input results/memory-read-validated \
  --plan results/memory-read-plan.json --output results/memory-read-report --plots
```

**목표 계층 경로와 coalescing을 따로 검증한다.** 기존 sector inflation 상한 8.25는 의도적인 stride 진단의 경로 검사에 남긴다. 에너지 후보에는 별도로 정렬된 연속 접근과 해당 replay의 read-sector/logical-read 비율이 기대값 1에 가까운지 확인한다. 기본 허용 오차는 ±5%이며 기대 효율은 100%다. L1은 L1 read sectors, L2/HBM은 L2 read sectors를 사용한다. 이 L2 관측을 DRAM 전송 효율이라고 부르지 않는다. 근거가 없으면 미확정으로 남기고 검증된 에너지 후보에서 제외한다.

`cache-sector-diagnostics.json`·`locality.json`의 stride/offset/SM 실험은 `experiment_role: "diagnostic"`으로 유지하며 최적 에너지 후보에서 제외한다. Raw 단가·logical bandwidth·NCU sector/DRAM traffic은 그대로 보존한다. 기존 stride 4 결과에 4를 곱해 BW를 실측치로 만들거나 pJ를 4로 나누어 보정하지 않는다. 기존 결과는 최신 분석기로 다시 평가하고, coalesced 비교값은 새 binary·plan으로 재측정한다. [Sector 효율 검토와 비교 방법](docs/cache-sector-review.ko.md)

## 비선형 실험: SFU 마이크로벤치

**`configs/nonlinear*.json`은 register-resident SFU 실험이다.** 반복 루프에서 global/shared/local load/store를 제거하고, 동일한 bounded register loop에서 SFU 명령을 뺀 control과 AB/BA로 비교한다. L1·L2·HBM working set·stride·행 너비는 sweep하지 않는다. 기존 `sfu-register*.json`도 같은 설정으로 유지한다.

| 연산 / workload | 의미 | V100 | A100/H100 |
|---|---|---|---|
| EX2 / `sfu_ex2` | 2ˣ | 지원 | 지원 |
| LG2 / `sfu_lg2` | log₂x | 지원 | 지원 |
| RCP / `sfu_rcp` | 1/x | 지원 | 지원 |
| RSQ / `sfu_rsqrt` | 1/√x | 지원 | 지원 |
| SQRT / `sfu_sqrt` | √x | 지원 | 지원 |
| TANH / `sfu_tanh` | native tanh(x) | 미지원, skip 사유 기록 | 지원 |

모두 FP32 **근사 명령**이다. EXP(eˣ)는 EX2(2ˣ)와 다르고, RMSNorm·Softmax·SiLU 전체에는 다른 산술·reduction이 필요하므로 기본 SFU 실험에 포함하지 않는다.

Q=`sfu_lanes`는 register 작업 lane 수이며 블록 수는 `ceil(Q/threads)`다. Q 3개 × threads 2개 × 독립 chains 3개로 처리량을 점검한다. 각 chain의 다음 입력을 register에서 유효 범위로 재구성하며 control도 같은 재구성 작업을 한다. 마지막 검증용 lane당 4B store 한 번은 반복 후에만 수행한다. `nonlinear-amortization.json`으로 반복 길이에 따른 초기화·최종 store·launch 비용의 영향을 확인한다. Q만 늘렸다고 포화가 입증되는 것은 아니다.

### 실행과 검증

```bash
export POWERBENCH=build/powerbench  # A100/CUDA13: build-a100-cuda13/powerbench
python -m pip install -e ".[plots]"
python tools/check_sfu_sass.py --binary "$POWERBENCH" --output results/sfu-sass.json
python -m powermodeling plan --config configs/nonlinear-smoke.json \
  --bench "$POWERBENCH" --device 0 --sfu-sass-evidence results/sfu-sass.json \
  --output results/nonlinear-smoke-plan.json
python -m powermodeling run --plan results/nonlinear-smoke-plan.json \
  --bench "$POWERBENCH" --device 0 --output results/nonlinear-smoke
python -m powermodeling analyze --input results/nonlinear-smoke \
  --output results/nonlinear-smoke-report --plots

# 고정 클럭 sweep 및 별도 NCU 검증
python -m powermodeling plan --config configs/nonlinear.json \
  --bench "$POWERBENCH" --device 0 --sfu-sass-evidence results/sfu-sass.json \
  --output results/nonlinear-plan.json
python -m powermodeling run --plan results/nonlinear-plan.json \
  --bench "$POWERBENCH" --device 0 --output results/nonlinear \
  --apply-clocks --clock-method applications
python -m powermodeling validate-run --plan results/nonlinear-plan.json \
  --input results/nonlinear --output results/nonlinear-validated \
  --profiles-dir results/nonlinear-profiles --bench "$POWERBENCH" --device 0 \
  --apply-clocks --clock-method applications
python -m powermodeling analyze --input results/nonlinear-validated \
  --plan results/nonlinear-plan.json --output results/nonlinear-report --plots
```

SASS 검사는 실제 native 명령, 반복당 명령 수, hot loop의 메모리 접근·spill 여부를 확인하며 증거를 실행 파일 SHA256에 연결한다. 실행 파일을 다시 빌드했다면 certificate와 plan도 새로 만든다. NCU replay는 전력 측정과 분리하며 누락된 검증은 `inconclusive`로 남긴다. 지원되지 않는 clock 적용 방식의 대안과 복원 방법은 위 클럭 지침을 따른다.

Smoke는 V100 20 trials/최소15분, A100/H100 24 trials/최소18분이다. 전체 sweep은 **clock 조건 하나당 V100 최소4.5시간, A100/H100 최소5.4시간**이며 NCU·준비·overrun 시간은 추가된다. graphics sweep은 900 MHz 이상·기본90 MHz 간격, 지원되는 exact1110 MHz·advertised factory default·현재 정책 reference를 포함한다. Memory clock은 treatment/control의 환경 조건으로 기록하고 domain별로 비교한다. SFU 커널이 메모리 계층을 읽는 실험이라는 뜻은 아니다.

### 단가와 평가

주 결과 `sfu_reference_delta_pj_per_instruction`은 다음과 같다.

```text
SFU 실행 수 N = 완료 launches × Q × chains × iterations
SFU 차분 pJ/instruction = (P_treatment − P_control) / (N / treatment 시간) × 10¹²
```

여기서 scalar 명령 적용 1회를 element 1회로 정의하면 pJ/element와 같은 수치다. 초기 Q로만 나누지 않는다. **Idle을 뺀 값이나 전체 GPU 에너지가 주 결과가 아니다.** 전체 단가와 idle 증가분은 진단으로 보존한다. Register/issue·실행 시간 차이의 영향이 남으므로 이 차분을 물리 SFU 전원선만의 에너지로 단정하지 않는다.

`evaluation.html`에서 primitive·Q·chains·클럭별 차분과 반복 CI, 처리량, control 전력, factory default·1110 MHz 대비를 확인한다. Q·chains·반복 수가 다른 단가를 하나의 median으로 합치지 않는다. 후보는 정확한 count·수치·SASS/NCU·paired 상태·반복 정밀도·Q scaling 근거를 검사한다. 처리량 기본95% 기준은 비교 가능한 sweep에서 **관측한 최고 처리량**에 대한 유지율이며, 이론적 SFU 최대 성능의 95%를 요구하는 뜻은 아니다. 음수·0을 포함하는 CI는 진단으로 표시하고 에너지 최적점으로 승인하지 않는다. 전체 선정 순서와 시각화는 [평가 설계](docs/evaluation-design.ko.md), 커널·control·실행 지침은 [비선형 SFU 실험 설계](docs/nonlinear-experiments.ko.md)를 참조한다.

이전 global 입출력 기반 EXP·TANH·RMSNorm·Softmax·SiLU 설정은 `configs/legacy/nonlinear-streaming*.json`으로 옮겼다. 이 workload를 새로 실행하려면 config의 각 experiment `parameters`에 `nonlinear_mode: "streaming"`, 직접 CLI에는 `--nonlinear-mode streaming`을 명시한다. 기존 raw 결과는 계속 분석할 수 있으며 SFU 결과와 단위·조건을 분리한다. [이전 streaming 실험 지침](docs/legacy/nonlinear-streaming.ko.md)

## sweep 구성

| 설정 파일 | 목적 |
|---|---|
| [configs/smoke.json](configs/smoke.json) | 센서·CUDA·기본 실행을 확인하는 작은 DVFS sweep |
| [configs/saturation.json](configs/saturation.json) | warp/block·working set·accumulator·GEMM 크기와 고정 클럭 탐색 |
| [configs/dvfs.json](configs/dvfs.json) | memory×SM clock 도메인의 bandwidth plateau와 효율 탐색 |
| [configs/memory-read.json](configs/memory-read.json) | L1·L2·HBM의 stride 1 coalesced read 에너지·clock sweep |
| [configs/memory-read-smoke.json](configs/memory-read-smoke.json) | 같은 read 경로의 실행·센서 점검 |
| [configs/cache-sector-diagnostics.json](configs/cache-sector-diagnostics.json) | Sector 정렬·stride 효율 진단; 에너지 최적점에서 제외 |
| [configs/locality.json](configs/locality.json) | L2 latency/stride/주소 offset/실행 SM 진단; 에너지 최적점에서 제외하고 물리 near/far labels는 자동 부여하지 않음 |
| [configs/nonlinear-smoke.json](configs/nonlinear-smoke.json) | Native SFU 5/6종과 대응 register control의 실행·수치·센서 점검 |
| [configs/nonlinear.json](configs/nonlinear.json) | SFU 기본 명령의 Q·threads·chains·clock별 차분 pJ/instruction |
| [configs/nonlinear-amortization.json](configs/nonlinear-amortization.json) | SFU 반복 길이에 따른 초기화·최종 store·launch 비용 영향 점검 |
| `configs/sfu-register*.json` | 대응하는 `nonlinear*.json`과 같은 SFU 설정; 기존 경로 호환 |
| [configs/component-diagnostics.json](configs/component-diagnostics.json) | Tensor·L1·L2·HBM 단가가 높을 때 iterations/batching·Tensor dependency·L1 footprint를 분리하는 추가 진단 |

JSON의 `clock_pairs`로 검증된 특정 pair를 진단할 수 있다. **`saturation.json`·`dvfs.json`·`locality.json`·`nonlinear.json`의 energy sweep은 기본 `graphics_min_mhz: 900`, `graphics_step_mhz: 90`을 사용한다.** 간격은 60·90·120 MHz 등 양의 정수로 변경할 수 있다. 각 memory domain에서 900 MHz 이상의 지원 값에만 grid를 매핑하고 해당 평가 범위의 끝점, exact 1110 MHz(지원 domain), advertised factory-default 고정 pair와 incoming-policy reference를 포함한다. 일반 200–300 MHz sweep는 생성하지 않는다. 필수 default pair가 하한 아래이면 그 anchor만 예외로 포함한다. 1110 미지원은 사유를 기록하며 근사값으로 대체하지 않는다. incoming policy를 factory-default DVFS라고 단정하지 않는다. 지원/평가 범위·제외된 낮은 native 값·실제 간격은 `plan.clock_sweep_coverage`에 남긴다.

`study_design: "energy_sweep"`은 graphics 간격·memory domain·1110·advertised default·current-policy reference의 요구사항 coverage를 검사한다. default pair를 조회하거나 지원 근거로 확인하지 못한 계획은 `requirements_status: "incomplete"`, `execution_allowed: false`이며 runner가 clock 변경·커널 실행 전에 차단한다. 계획 파일과 미확정 사유는 검토용으로 남는다. Runner는 기록된 native 지원 clock 목록에서 선언한 하한·간격의 grid·평가 끝점·default·1110 조건을 다시 계산하고, 각 geometry와 treatment design 층의 실제 trial 목록에 그 clock pair가 모두 있는지 검사한다. 선택된 coverage와 trial 목록에서 같은 조건을 함께 제거해도 native 목록에 근거한 필수 grid 검사로 드러난다. `smoke.json`의 null clock은 `diagnostic` 예외로 허용하고 전체 효율 sweep로 승인하지 않는다. `sm_count`, `l2_bytes`, `total_memory_bytes`를 쓰는 숫자 표현식은 장치의 조회값으로 해석된다. `blocks = sm_count × 2`는 작업량 지정이며 정확히 각 SM에 2 blocks를 배치하는 명령이 아니다.

DVFS 설정은 SM 수의 2·4·8배 blocks × 128·256 threads, GEMM shape를 함께 비교한다. saturation 설정은 Tensor accumulator 수도 바꾼다. seed·working set·stride·주소 offset만 바꾸어 resource geometry 개수를 부풀리지 않는다. 각 frequency pair에서 NCU·품질·정확한 시간 정렬을 통과한 최소 2개의 resource geometry가 실제로 비교되어야 승인된 효율 최적점을 만들 수 있다. geometry 비교 수는 자원 활용에 대한 최소 근거이며 실제 plateau의 증명은 아니다. saturation 결과와 NCU에서 확인한 geometry·working set으로 범위를 늘리거나 정밀하게 탐색한다. 전체 격자는 수시간 걸릴 수 있으므로 plan의 trial 수와 예상 시간을 확인한다.

HBM stride 실험은 stride가 커질 때 전체 할당 크기도 늘리고 순환 접근의 가능한 sector footprint를 검사한다. worker는 한 launch가 유한한 iteration 동안 실제로 방문할 수 있는 footprint와 전체 stride cycle의 footprint를 구분한다. 짧은 launch가 같은 작은 주소 집합을 반복하면 큰 할당이어도 cache 실험이 될 수 있으므로 DRAM counter가 필요하다. write/copy는 thread 간 주소 소유권이 겹치지 않도록 유효 크기를 조정하고 실제 사용 크기를 결과에 기록한다.

기본 custom Tensor·L1·L2·HBM 에너지 실험은 초기 target warmup 3초, 전후 idle 각각 6초, reference/treatment 각 arm warmup 3초·측정 12초, 50 ms polling, 기본 4회 반복이다. 각 arm은 최소 10초 이상 측정하며, 완전히 균형 잡힌 AB/BA 순서는 짝수 4회 이상 반복을 권장한다. odd 반복은 순서 imbalance를 진단에 남기며 paired 최적점에는 동일한 유효 AB/BA 수를 요구한다. 기본 cuBLAS GEMM은 대응 geometry가 불명확하여 unpaired이며 전체·idle 기준으로 읽는다. 분석은 각 active arm 양 끝 2초 및 idle 양 끝 1초를 제외한다. 50 ms polling이 센서의 50 ms 갱신을 뜻하지 않는다. config에서 실험 시간을 늘릴 수 있다.

## Treatment·idle·active reference

Treatment는 측정하려는 대상 작업이다. `paired_reference: true`인 custom workload는 같은 worker process·context·할당·clock policy에서 `control`을 AB/BA 순서로 짝지어 실행한다. SFU는 대상 명령을 제거한 register recurrence control, Tensor·memory는 기존 issue-loop control을 사용한다. grid/thread·loop·SM filter·batch 설정과 actual clocks·온도·cap·간섭을 검사하며, 각 arm의 에너지와 완료 count를 독립적으로 정렬한다. control도 정수 연산·제어·launch·store 전력을 쓰므로 두 arm의 차이는 operational contrast다.

전후 idle는 active 시점에 보간하고 drift·actual clock·온도를 검사한다. baseline가 실패해도 treatment 전체 에너지의 품질 판정은 보존한다. 미승인 차감값과 음의 대비는 진단값으로 남기지만 승인된 최적점 후보로 사용하지 않는다. 전체 에너지, 승인된 idle 증가분, 승인된 paired 대비를 서로 대체하지 않으며 순수 static/dynamic·회로 에너지로 이름 붙이지 않는다. 이전 plan/결과와 새 paired protocol을 같은 repeat로 합치지 않는다.

## 결과 읽기

`analyze`는 [컴포넌트별 평가·시각화 설계](docs/evaluation-design.ko.md)에 따라 계획 대비 coverage, 측정 품질, NCU/수치 근거, resource plateau, 처리량을 유지하는 에너지 후보와 default·1110 대비 개선을 함께 평가한다. `evaluation.html`은 GPU·함수·입력·objective·memory·clock 필터와 CI/anchor/품질/근거 표를 제공하고 SVG를 저장할 수 있다. `evaluation.json`·`evaluation.csv`도 생성하며 `--plots`는 조건을 분리한 PNG와 SVG를 낸다. 원본 run의 `plan.json`은 자동 사용하고, NCU 검증 후 디렉터리에는 원 energy plan을 `--plan`으로 지정한다. plan이 없거나 누락·plateau 부족이 있으면 후보는 잠정으로 남긴다.

```bash
python -m powermodeling analyze --input results/validated \
  --plan saturation-plan.json --output results/validated-report --plots
```

각 trial의 raw JSON과 실행 plan이 결과 폴더에 저장된다. 분석 폴더의 `trials.csv`와 `summary.json`에서 측정값과 탈락 이유를 확인한다.

| 값 | 해석 |
|---|---|
| `board_power_w` | 측정 구간의 NVML GPU scope 평균 전력 |
| `idle_power_w` | 전후 idle 구간으로 보간한 기준 전력 |
| `incremental_power_w` | 전체 GPU 전력의 기준 대비 증가분; 순수 block dynamic이 아님 |
| `total_pj_per_flop`, `total_pj_per_logical_bit` | 전체 GPU 단가. FMA=2 FLOP, logical bit=요청 byte×8 |
| `operational_idle_increment_pj_per_flop`, `operational_idle_increment_pj_per_logical_bit` | 전후 idle 대비 증가분 단가; 승인 여부와 함께 읽음 |
| `paired_active_reference_pj_per_flop`, `paired_active_reference_pj_per_logical_bit` | 같은 process의 짝지은 control 대비 단가; 물리 component isolation이 아님 |
| `total_pj_per_element`, `operational_idle_increment_pj_per_element`, `paired_active_reference_pj_per_element` | 비선형 함수의 전체·idle 증가분·paired 대비 단가; 분모는 완료 출력 원소 수 |
| 각 `*_pj_per_row`, `throughput_elements_s`, `throughput_rows_s` | RMSNorm·Softmax의 행 단가와 원소/행 처리량; 행 너비별로 비교 |
| `baseline_state_matched`, `baseline_state_issues` | 조회 가능한 pstate/enforced power cap의 일치·불일치 근거 |
| `baseline_valid`, `baseline_issues`, `operational_idle_increment_eligible` | 전후 idle 품질과 실제 상태 일치에 따라 증가분 최적점 사용을 승인 |
| `paired_active_reference_eligible`, `paired_active_reference_issues` | arm별 geometry·시간 정렬·actual clock·온도·cap·간섭·protocol, 조회 가능한 pstate/enforced cap 검사 |
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
| `empirical_gpu_energy_optima` | GPU UUID·작업·access·고정 memory MHz·objective별 승인 단가 최솟값. 각 frequency pair의 최소 2 검증 resource geometry 비교·관측 peak의 기본 95% 이상 요구 |
| `empirical_gpu_overall_energy_optima` | 같은 승인 조건에서 measured graphics·memory domain을 함께 비교한 각 GPU의 실제 최솟값 |
| `exploratory_single_geometry_energy_optima`, `exploratory_single_geometry_overall_energy_optima` | 검증된 geometry 한 종류뿐인 fixed/overall 탐색 결과; 승인된 효율 최적점과 분리 |
| `own_clock_verified_distinct_geometry_count`, `own_clock_observed_distinct_geometry_count` | 해당 clock pair의 검증된/전체 유효 resource geometry 비교 수. `own_clock_distinct_geometry_count`는 검증 수 alias |
| `geometry_evidence_status`, `saturation_proven` | 승인 또는 단일 geometry 진단의 근거. 최소 2 비교도 실제 hardware saturation 증명이 아니므로 `saturation_proven: false` |
| `near_optimum_support_points`, `uncertainty_overlap_support_points` | 기본 최소 단가 5% 이내 / 95% 구간이 겹치는 실측 support points; 미측정 gap이나 연속 최적 구간을 보장하지 않음 |
| `observed_frequency_pairs_without_eligible_candidate` | 측정은 했으나 검증·활용·baseline 요건으로 최적점 후보를 만들지 못한 frequency pair |
| `verified_target_*` | NCU 통과·정확한 시간 정렬 조건 중 **전체 유효 고정 클럭 sweep 최고 처리량의 95% 이상**을 달성한 결과의 최저 단가; 없으면 winner 없음 |
| `verified_target_coverage` | 전체/검증된 최고 처리량·비율·95% 통과 조건 수·미검증 또는 실패한 peak group을 보고 |

`valid=true`는 기록의 품질 기준을 통과했다는 뜻이며, `target_verified=true`나 물리 블록 isolation의 증명이 아니다. worker는 약 1초 간격의 완료 batch 수와 실제 SM admission 수를 기록하고, 분석은 양 끝을 제외한 완료 구간에서 work count와 에너지를 함께 계산한다. 이 기록이 없는 과거 결과는 지속 처리량이 일정하다는 가정의 추정치로 남기고 검증된 최적값에는 사용하지 않는다. idle와 active의 실제 클럭이나 온도가 다르면 증가분에는 activation·주파수 상태·누설 변화가 섞일 수 있으며 경고가 남는다. 3–4회처럼 적은 반복의 bootstrap 범위는 거칠다. 수치 차이가 작으면 반복과 최소점 주변 지원 주파수 측정을 늘리고 온도·센서·counter evidence를 확인한다. V100·A100·H100의 최적 pJ/bit·pJ/FLOP 주파수는 각 UUID의 결과에서 독립적으로 선택하며 1110 MHz를 최적점으로 미리 지정하지 않는다. physical pJ/bit는 동일 energy-window의 계층 traffic provenance가 없어 현재 withheld이고 NCU replay bytes만으로 단가를 계산하지 않는다.

## 단순 메모리 read 커널

L1·L2·HBM의 `access=read`는 **단일 stream의 32-bit load + uint32 덧셈 누산**을 사용한다. 이전 네 stream의 XOR 누산과 여러 주소 관리를 줄였으며 L1 `.ca`, L2/HBM `.cg`를 유지한다. 읽은 값을 전혀 사용하지 않으면 컴파일러가 중간 load를 제거하므로 덧셈 하나는 남긴다. 기본 iterations는 4096으로, 이전 read의 1024 × 4 loads와 같은 요청량이다. 직접 지정한 iterations는 변환하지 않는다.

[변경 내용·count·재실험 절차](docs/memory-read-v2.ko.md)를 참고한다. [메모리 read 전용 점검 설정](configs/memory-read-smoke.json)은 L1/L2/HBM 합계 12 trials·최소 9분이며, `--stage hbm_read`로 HBM만 실행하면 4 trials·최소 3분이다. NCU·준비 시간은 별도다. 새 binary로 plan과 profile을 다시 만들고, 이전 결과와는 implementation version·binary hash로 구분한다. 실제 에너지 개선은 GPU 재측정으로 확인해야 한다.

## 기존 실험보다 에너지 단가가 높을 때

먼저 같은 컴포넌트·단위·에너지 기준·실제 클럭을 비교한다. 전체 GPU 단가와 idle 증가분은 서로 다르고, 메모리의 legacy `pj_per_logical_byte`는 pJ/bit의 8배다. 같은 기준이라면 높은 단가는 전력 증가 또는 낮은 지속 처리량에서 생길 수 있다. 경로 검증 `pass`만으로 bandwidth 포화를 확인하지 않는다.

기존 raw를 새 출력 폴더로 `analyze`하면 `measurement_diagnostics`가 같은 구간의 J/count 재구성, 전체/idle/reference 비율, 누적 에너지·전력 적분 crosscheck, timing과 occupancy 근거를 제공한다. `evaluation.html`의 **Energy accounting and comparison checks**에서 각 기준을 나란히 확인할 수 있다. NCU evidence에는 sector·DRAM traffic amplification을 추가했다. 에너지 숫자에 임의의 보정 배율을 적용하지 않는다.

[원인 점검·재분석·추가 진단 지침](docs/high-energy-investigation.ko.md)에 비교 항목과 `component-diagnostics.json`의 stage별 실행 방법을 설명한다. 전체 진단은 clock 조건 하나에서 최소 약 2.29시간이며, 필요한 stage만 실행할 수 있다. 기존 raw는 원래 binary의 evidence를 유지하고 새 binary로 수행한 진단은 별도 결과로 저장한다.

[L1/L2 sector·정렬 검토](docs/cache-sector-review.ko.md)는 128-byte cache line과 32-byte 최소 접근 단위를 구분하고, offset 0/4/32/128 B와 stride 1/2/4/8/32의 비교 방법을 설명한다. `configs/cache-sector-diagnostics.json`으로 L1/L2만 실행할 수 있으며, clock 조건 하나에서 alignment 최소 24분, stride 최소 30분이다. NCU replay 시간은 별도다.

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
| EXP·TANH·RMSNorm·Softmax·SiLU | 대상 커널·SFU instruction 활동·spill·duration, CPU 기준 표본 검증, 원소/행 카운트와 조건 일치 | 전체 함수 구현 경로의 근거; SFU의 물리 에너지나 처리량 포화 증명과 구분 |

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
- [비선형 SFU 마이크로벤치: register loop·차분 pJ/instruction·control·SASS 검증](docs/nonlinear-experiments.ko.md)
- [이전 streaming 전체 함수 실험: EXP·TANH·RMSNorm·Softmax·SiLU](docs/legacy/nonlinear-streaming.ko.md)
- [컴포넌트별 평가·시각화: coverage·반복 정밀도·plateau·에너지 후보](docs/evaluation-design.ko.md)
- [높은 Tensor·L1·L2·HBM 단가: 기준·계산·처리량 점검과 추가 진단](docs/high-energy-investigation.ko.md)
- [전체 구현 자가점검: 발견 사항·수정·검증·남은 실측](docs/self-audit.ko.md)
- [NVIDIA 공식 출처 및 검증이 필요한 주장](docs/sources.md)
