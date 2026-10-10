# NVIDIA GPU power modeling

**SXM 모듈의 V100·A100·H100**에서 **충분한 처리량을 유지하는 pJ/FLOP·pJ/logical-bit·pJ/SFU instruction 최소점과 주변 측정점**을 찾기 위한 CUDA/NVML 실험 도구다. HBM은 **같은 에너지 구간의 logical byte/s가, 실측 memory clock과 버스 폭으로 계산한 이론 bandwidth의 80% 이상**인 후보에서 최소 단가를 고른다. 이 80%는 DRAM bus 사용률이 아니다. 다른 계층·연산은 관측 peak·plateau의 상대 평가를 사용한다. 어느 판정도 물리적 포화나 전역 최소점을 보장하지 않는다. FP16 Tensor, L1, L2, HBM과 **register-resident SFU 기본 명령**의 작업량·thread/block·SM/memory clock을 바꾸고, 수초 동안 전력과 처리량을 함께 측정한다. 비선형 기본 실험은 global 입력 버퍼 없이 EX2·LG2·RCP·RSQ·SQRT·native TANH를 반복한다. 메모리 실험에는 stride·주소 offset sweep도 제공한다.

SXM은 GPU 모듈의 장착 형태이고 HBM은 측정할 메모리 계층이다. 실제 메모리 용량·SKU·SM 수를 이름만으로 확정하지 않는다. 이 저장소에는 실측 GPU 숫자가 들어 있지 않다. 전체 단가, 승인된 전후 idle 증가분, 같은 process에서 짝지은 active-reference 대비를 별도로 보고한다. `idle`은 운영상 기준이며 순수 누설 전력이 아니다. 지원되는 memory power scope도 전체 GPU scope와 구분한다. cache/DRAM counter가 목표 경로를 확인한 뒤에도 측정 에너지를 해당 물리 회로만의 에너지로 동일시하지 않는다.

[실험 설계 HTML](docs/experiment-design.html)은 treatment/reference 도식, clock coverage, 실제 plan·summary JSON의 로컬 뷰어를 제공한다. 서버 업로드 없이 사용할 수 있으며 문서 도식에는 실측 전력곡선이 없다.

바로가기: [설계 원칙과 정합성](#설계-원칙과-구현-정합성) · [설치와 빌드](#설치와-빌드) · [sweep 구성](#sweep-구성) · [결과 읽기](#결과-읽기) · [H100 메모리 센서](#h100-hbm-메모리-전력에너지).

## 제공 기능

| 기능 | 구현 |
|---|---|
| Tensor | FP16 입력·FP32 누산 WMMA `m16n16k16` register reuse, 독립 accumulator sweep, cuBLAS dense GEMM 비교. raw PTX `mma.m16n8k16`이나 shared-resident operand 경로가 아님 |
| L1·L2·HBM | L1 `.ca`, L2/HBM `.cg` load, HBM `.ca`/`.cs` 비교 진단; working set·grid·thread·stride·주소 offset·read/copy sweep |
| L2 locality | 의존 pointer chase의 SM별 cycle/access와 offset 변화; near/far 확정은 별도 evidence 필요 |
| Register SFU | EX2·LG2·RCP·RSQ·SQRT·native TANH의 register 반복, SFU를 뺀 control 대비 차분 pJ/scalar instruction |
| 전력 측정 | capability 기반 NVML 평균/현재/누적에너지, 지원되는 memory scope, raw timestamps·오류 |
| 시간·기준 | 같은 context·버퍼·clock policy의 전후 idle 및 AB/BA paired active reference, arm별 warmup·완료 epoch의 정렬 적분 |
| 클럭·DVFS | 900 MHz 이상 지원 pair의 60/90/120 등 가변 간격 graphics grid, 정확한 1110 MHz·advertised default·현재 정책 reference coverage, 요청/실제 클럭·복원 |
| 분석 | GPU UUID별 전체·idle 증가분·paired-reference 단가 최소와 이산 근접 측정점·bootstrap 구간; 주파수별 활용 조건과 전체 최고 성능 제약을 분리 |
| 모델 | 명시적 활동률 특징, rank/condition 검사, GPU·요청 클럭·온도 폭·실행 scope 분리. `fitted`는 calibration 성공이고 holdout 승인과 별도 |
| 검증 | Nsight Compute counter를 자동 판정하고 분석에 반영; `pass`/`fail`/`inconclusive` 및 근거 보존 |

**A100 GA100의 `nvmlDeviceGetPowerUsage`는 현재 전력이고 H100은 약 1초 평균이다.** H100 이름만 보고 HBM 별도 센서가 지원된다고 가정하지 않는다. 지원 scope와 실패 상태를 실제 장치에서 확인한다. [공식 자료](docs/sources.md)

## 설치와 빌드

저장소 루트에서 아래 명령으로 실험 도구를 자동 설치하고 빌드할 수 있다. 전체 설치는 **Linux x86_64, Python 3.10 이상(`venv` 포함)**이 필요하며 `sudo` 없이 저장소 안에 설치한다. `pypi.org`, `files.pythonhosted.org`, `conda.anaconda.org`에 HTTPS로 접근할 수 있어야 한다. 다운로드는 약 1.1 GB이며 설치·패키지 캐시·빌드용으로 **8 GB 이상의 여유 공간**을 준비한다.

```bash
python3 tools/setup_experiment.py --build --jobs 2
source .tools/env.sh
python -m powermodeling --help
"$NCU" --version
"$POWERBENCH" --help
```

[설치 스크립트](tools/setup_experiment.py)는 V100·A100·H100 공통 도구로 **CUDA 12.9, cuBLAS, profiler API, Nsight Compute 2025.2.1, GCC/G++ 13, cuobjdump, nvdisasm**을 설치한다. Python 가상환경에는 이 프로젝트와 `numpy`, `nvidia-ml-py`, `matplotlib`, CMake와 Ninja를 설치한다. `--build`를 생략하면 도구만 설치하고, 빌드는 `--build`로 다시 실행한다. 같은 명령을 재실행하면 기존 설치와 다운로드 캐시를 재사용한다.

| 경로 / 변수 | 용도 |
|---|---|
| `.venv/`, `.venv/bin/python` | 프로젝트·분석·plot·테스트에 사용하는 Python 환경 |
| `.venv/bin/cmake`, `.venv/bin/ninja` | 빌드 도구 |
| `.tools/cuda12/` | CUDA 12.9·cuBLAS·NCU·host compiler 설치 prefix |
| `.tools/cuda12/bin/nvcc`, `cuobjdump`, `nvdisasm` | CUDA compiler·SASS 도구; 세 executable 모두 같은 `bin/` 안에 설치 |
| `.tools/cuda12/bin/ncu`, `$NCU` | 설치한 Nsight Compute executable; 실행 시 `$NCU`로 지정 |
| `build-cuda12/powerbench`, `$POWERBENCH` | `--build`로 생성한 `70;80;90` CUDA 실행 파일 |
| `.tools/env.sh` | `.venv` 활성화 및 `PATH`, `CUDACXX`, `CUDAToolkit_ROOT`, `CXX`, `CUDAHOSTCXX`, `NCU`, `POWERBENCH` 설정 |
| `.tools/cache/`, `.tools/mamba/` | 다운로드·패키지 캐시와 설치 관리자 상태 |

새 Bash 세션마다 `source .tools/env.sh`를 실행한다. `NCU`와 `POWERBENCH`는 절대 경로로 설정되므로 시스템의 다른 NCU나 `build/powerbench`와 섞이지 않는다. **NVIDIA driver 설치·변경과 GPU counter 권한 설정은 스크립트가 수행하지 않는다.** 실제 측정에는 호환 driver가 있는 GPU 호스트와 필요한 counter/clock 권한이 별도로 필요하다. 설치·빌드와 버전 조회만으로 GPU 실행이나 측정 정확도가 검증되는 것은 아니다.

GPU 도구 없이 Python 분석·plot·CPU 테스트만 준비하려면 다음을 실행한다. 이 모드는 CUDA/NCU 경로를 설정하지 않으며 `--build`와 함께 사용할 수 없다.

```bash
python3 tools/setup_experiment.py --python-only
source .tools/env.sh
python -m unittest discover -s tests -v
```

수동 설치에서는 Linux, Python 3.10 이상, CMake 3.22 이상, NVIDIA driver와 CUDA Toolkit을 준비한다. **A100은 CUDA 13.0의 `sm_80` 빌드를 지원한다.** V100·A100·H100을 같은 Toolkit으로 비교하려면 CUDA 12.x를 사용한다. CUDA 13.0은 V100/Volta의 offline compilation과 library support를 제거했다.

NCU도 Toolkit·GPU 세대에 맞춘다. **자동 설치의 CUDA 12.9 + Nsight Compute 2025.2.1은 V100·A100·H100 공통 경로다.** Nsight Compute 2025.3부터 Volta 지원이 제거되었다. A100/H100의 수동 CUDA 13 경로에는 CUDA 13을 지원하는 **별도 Nsight Compute 2025.3 이상**을 설치하고, 자동 설치의 NCU 2025.2를 재사용하지 않는다. `profile`/`validate-run --ncu /설치경로/ncu`로 실제 executable을 지정하고 driver 요구사항을 확인한다. [공식 버전 지원 자료](docs/sources.md)

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

아키텍처를 생략한 새 빌드의 기본값은 CUDA 12에서 `70;80;90`, CUDA 13에서 `80;90`이다. `-DCMAKE_CUDA_ARCHITECTURES`와 `CUDAARCHS` 환경변수의 지정값을 우선한다. 수동 CUDA 13 실행에서는 `POWERBENCH=build-a100-cuda13/powerbench`, `NCU=/별도-NCU-2025.3-이상-설치경로/ncu`로 두 실행 파일을 함께 바꾼다. CUDA/cuBLAS 버전이 다른 측정은 별도 분석 층으로 기록한다.

CUDA 12 compiler를 별도 경로로 지정하려면 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`를 추가한다. Python 분석만 사용할 때는 CUDA 빌드가 필요하지 않다.

## 첫 실험

먼저 장치를 조회하고 기본 DVFS 상태의 smoke sweep으로 실행·센서·결과 형식을 확인한다. smoke도 평균 센서 때문에 수초씩 실행하며, 포화 조건을 전수 탐색하는 용도는 아니다.

아래 실행 예시는 자동 설치의 경로를 사용한다. 수동 설치를 사용하면 `source .tools/env.sh`를 생략하고 선택한 `POWERBENCH`와 `NCU`를 직접 `export`한다. 같은 plan의 전력 실행과 NCU 검증에는 같은 benchmark 실행 파일을 사용한다.

```bash
source .tools/env.sh
python -m powermodeling discover --device 0 --bench "$POWERBENCH"
python -m powermodeling plan --config configs/smoke.json --device 0 --bench "$POWERBENCH" --output smoke-plan.json
python -m powermodeling run --plan smoke-plan.json --device 0 --bench "$POWERBENCH" --output results/smoke
python -m powermodeling analyze --input results/smoke --output results/smoke-report
```

고정 클럭의 처리량·효율 sweep은 다음과 같다. `plan`이 지원 clock pair를 탐색하고 실행 수와 최소 예상 시간을 보여 준다. 실제 시간에는 메모리 준비와 clock settling 등이 추가된다.

```bash
python -m powermodeling plan --config configs/saturation.json --device 0 --bench "$POWERBENCH" --output saturation-plan.json
python -m powermodeling run --plan saturation-plan.json --device 0 --bench "$POWERBENCH" --output results/saturation --apply-clocks --clock-method applications
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
source .tools/env.sh  # 자동 설치의 CUDA 12.9 / NCU 2025.2.1 경로
python -m powermodeling plan --config configs/memory-read.json \
  --bench "$POWERBENCH" --device 0 --output results/memory-read-plan.json
python -m powermodeling run --plan results/memory-read-plan.json \
  --bench "$POWERBENCH" --device 0 --output results/memory-read \
  --apply-clocks --clock-method applications
python -m powermodeling validate-run --plan results/memory-read-plan.json \
  --input results/memory-read --output results/memory-read-validated \
  --profiles-dir results/memory-read-profiles --bench "$POWERBENCH" --ncu "$NCU" --device 0 \
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
source .tools/env.sh  # 수동 CUDA 13은 설치 절의 POWERBENCH·NCU 경로 사용
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
  --profiles-dir results/nonlinear-profiles --bench "$POWERBENCH" --ncu "$NCU" --device 0 \
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
| `board_power_w` | NVML device/관련 회로의 구간 평균 전력; 유효 누적 energy 차분을 우선하고 `nvmlDeviceGetPowerUsage` 적분으로 대체. 별도 field API의 GPU scope나 고립된 core rail과 동일하다고 가정하지 않음 |
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
| `hbm_memory_power` | HBM 실험의 별도 메모리 센서 W·J·pJ/logical bit, 출처·품질·반복 수·신뢰구간 |
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
| `within_clock_best`, `cross_clock_best` | 최저 승인된 증가분 단가. HBM은 해당 memory clock 이론 BW의 80% 이상; 다른 workload는 같은 clock/전체 clock 관측 peak의 95% 이상 |
| `within_clock_best_total_energy`, `cross_clock_best_total_energy` | 위 workload별 처리량 조건에서 최소 전체 에너지 단가의 설정 |
| `pareto_frontiers` | 더 높은 처리량과 더 낮은 전력으로 동시에 개선할 수 없는 관측 설정 |
| `active_control_associations` | 별도로 실행한 control과의 설명용 연결; same-process paired arm과 다르며 자동 component 차감에 사용하지 않음 |
| `empirical_gpu_energy_optima` | GPU UUID·작업·access·고정 memory MHz·objective별 승인 단가 최솟값. 각 frequency pair의 최소 2 검증 resource geometry 비교; HBM은 이론 BW 80%, 다른 workload는 관측 peak 95% 기준 |
| `empirical_gpu_overall_energy_optima` | 같은 승인 조건에서 measured graphics·memory domain을 함께 비교한 각 GPU의 실제 최솟값 |
| `exploratory_single_geometry_energy_optima`, `exploratory_single_geometry_overall_energy_optima` | 검증된 geometry 한 종류뿐인 fixed/overall 탐색 결과; 승인된 효율 최적점과 분리 |
| `own_clock_verified_distinct_geometry_count`, `own_clock_observed_distinct_geometry_count` | 해당 clock pair의 검증된/전체 유효 resource geometry 비교 수. `own_clock_distinct_geometry_count`는 검증 수 alias |
| `geometry_evidence_status`, `saturation_proven` | 승인 또는 단일 geometry 진단의 근거. 최소 2 비교도 실제 hardware saturation 증명이 아니므로 `saturation_proven: false` |
| `near_optimum_support_points`, `uncertainty_overlap_support_points` | 기본 최소 단가 5% 이내 / 95% 구간이 겹치는 실측 support points; 미측정 gap이나 연속 최적 구간을 보장하지 않음 |
| `observed_frequency_pairs_without_eligible_candidate` | 측정은 했으나 검증·활용·baseline 요건으로 최적점 후보를 만들지 못한 frequency pair |
| `verified_target_*` | NCU 통과·정확한 시간 정렬과 workload별 처리량 기준을 만족하는 최저 단가. HBM은 이론 BW 80%이며 관측 peak 95%를 추가로 요구하지 않음; 없으면 winner 없음 |
| `verified_target_coverage` | 전체/검증 peak·비율과 workload별 기준 통과 조건 수·미검증 또는 실패한 peak group을 보고 |
| `hbm_bandwidth` | HBM의 actual memory MHz·bus width·이론 byte/s·sustained logical byte/s·비율·80% 판정. 미지수는 미확정; 실제 DRAM bus 사용률 측정과 구분 |

`valid=true`는 기록의 품질 기준을 통과했다는 뜻이며, `target_verified=true`나 물리 블록 isolation의 증명이 아니다. worker는 약 1초 간격의 완료 batch 수와 실제 SM admission 수를 기록하고, 분석은 양 끝을 제외한 완료 구간에서 work count와 에너지를 함께 계산한다. 이 기록이 없는 과거 결과는 지속 처리량이 일정하다는 가정의 추정치로 남기고 검증된 최적값에는 사용하지 않는다. idle와 active의 실제 클럭이나 온도가 다르면 증가분에는 activation·주파수 상태·누설 변화가 섞일 수 있으며 경고가 남는다. 3–4회처럼 적은 반복의 bootstrap 범위는 거칠다. 수치 차이가 작으면 반복과 최소점 주변 지원 주파수 측정을 늘리고 온도·센서·counter evidence를 확인한다. V100·A100·H100의 최적 pJ/bit·pJ/FLOP 주파수는 각 UUID의 결과에서 독립적으로 선택하며 1110 MHz를 최적점으로 미리 지정하지 않는다. physical pJ/bit는 동일 energy-window의 계층 traffic provenance가 없어 현재 withheld이고 NCU replay bytes만으로 단가를 계산하지 않는다.

### H100 HBM 메모리 전력·에너지

H100에서 제공하는 NVIDIA **GPU Memory Power Readings**를 HBM 실험의 별도 결과로 보고한다. `nvmlDeviceGetFieldValues`에서 `NVML_POWER_SCOPE_MEMORY`의 순간·평균 전력을 조회하며, 사용 가능한 순간값을 우선하고 평균값으로 대체하면 그 출처를 기록한다. 평균값은 최근 1초의 전력이다. GPU 호스트에서 `nvidia-smi -q -d POWER`로 메모리 전력 항목을 확인할 수 있다. [NVIDIA 출처 S2·S10](docs/sources.md)

`hbm` workload의 `run`과 `analyze`에 별도 센서 옵션을 추가할 필요는 없다. 실제 응답으로 지원 여부를 판단하며 지원하는 다른 GPU의 메모리 센서도 같은 방식으로 기록한다.

| 결과 | 확인 경로 |
|---|---|
| 각 측정과 반복 집계의 센서 값·상태 | `summary.json`의 trial/group `hbm_memory_power`, `trials.csv` |
| 조건별 메모리 W·J·pJ/logical bit·idle 증가분·유효 센서 반복 수 | `hbm-memory-power.csv` |
| 전체 GPU 단가 옆에 표시한 메모리 센서 표 | `evaluation.html`의 **HBM memory sensor** 및 `evaluation.json` |
| 메모리 센서와 idle 증가분의 별도 그래프 | `analyze --plots`의 `hbm*-memory-sensor.png`·`.svg` |

`energy_j = ∫ memory_power_w dt`로 완료된 측정 구간을 적분하고, 같은 구간에서 센 benchmark 요청량으로 `pj_per_logical_bit = energy_j / (logical_bytes × 8) × 10¹²`를 계산한다. 전후 idle의 메모리 센서 값과 클럭·온도 등 상태가 맞는 경우에만 idle 대비 증가분을 별도 제공한다. 메모리 센서는 refresh와 주변 회로를 포함할 수 있으므로 이 증가분을 순수 HBM cell switching 에너지라고 단정하지 않는다.

센서 미지원은 `unavailable`과 null로, 일부 반복에서만 유효하면 `partial`과 실제 유효 반복 수로 표시한다. 누락 구간·음수 전력·오래된 센서 timestamp·순간/평균 출처 혼합 등으로 무효인 수치를 0으로 채우지 않는다. 센서 timestamp와 조회 시각이 없는 과거 기록은 `freshness_status=unverified`로 표시한다. 전후 메모리 idle 변화가 `max_idle_drift_fraction`을 넘으면 증가분을 보류한다. 정확한 동일 구간의 work count가 없으면 센서 W·J와 단가의 적합성을 구분하며, NCU replay의 bytes를 단가 분모로 대체하지 않는다. 메모리 센서 에너지는 전체 GPU 에너지에 더하거나 빼지 않으며, 기존 전체 GPU 에너지 최적점 선택과 독립적으로 보고한다.

## 단순 메모리 read 커널

L1·L2·HBM의 `access=read`는 **단일 stream의 32-bit load + uint32 덧셈 누산**을 사용한다. 현재 V3는 region word 수와 iterations가 UINT32_MAX 이하일 때 주소 index·loop counter를 32-bit로 처리하고, 큰 범위는 64-bit로 처리한다. 실제 pointer는 64-bit이며 wrap 주소 순서와 count는 같다. 이전 네 stream의 XOR 누산을 제거했고 L1 `.ca`, L2/HBM `.cg`를 기본으로 유지한다. 읽은 값을 전혀 사용하지 않으면 컴파일러가 중간 load를 제거하므로 덧셈 하나는 남긴다. 기본 iterations는 4096으로, 이전 read의 1024 × 4 loads와 같은 요청량이다. 직접 지정한 iterations는 변환하지 않는다. HBM의 `.ca`/`.cs` 비교는 별도 진단이다. [HBM 80%·오버헤드·cache 정책 설계](docs/hbm-bandwidth-design.ko.md)

[변경 내용·count·재실험 절차](docs/memory-read-v2.ko.md)를 참고한다. [메모리 read 전용 점검 설정](configs/memory-read-smoke.json)은 L1/L2/HBM 합계 12 trials·최소 9분이며, `--stage hbm_read`로 HBM만 실행하면 4 trials·최소 3분이다. NCU·준비 시간은 별도다. 새 binary로 plan과 profile을 다시 만들고, 이전 결과와는 implementation version·binary hash로 구분한다. 실제 에너지 개선은 GPU 재측정으로 확인해야 한다.

## 기존 실험보다 에너지 단가가 높을 때

먼저 같은 컴포넌트·단위·에너지 기준·실제 클럭을 비교한다. 전체 GPU 단가와 idle 증가분은 서로 다르고, 메모리의 legacy `pj_per_logical_byte`는 pJ/bit의 8배다. 같은 기준이라면 높은 단가는 전력 증가 또는 낮은 지속 처리량에서 생길 수 있다. 경로 검증 `pass`만으로 bandwidth 포화를 확인하지 않는다.

기존 raw를 새 출력 폴더로 `analyze`하면 `measurement_diagnostics`가 같은 구간의 J/count 재구성, 전체/idle/reference 비율, 누적 에너지·전력 적분 crosscheck, timing과 occupancy 근거를 제공한다. `evaluation.html`의 **Energy accounting and comparison checks**에서 각 기준을 나란히 확인할 수 있다. NCU evidence에는 sector·DRAM traffic amplification을 추가했다. 에너지 숫자에 임의의 보정 배율을 적용하지 않는다.

[원인 점검·재분석·추가 진단 지침](docs/high-energy-investigation.ko.md)에 비교 항목과 `component-diagnostics.json`의 stage별 실행 방법을 설명한다. 전체 진단은 clock 조건 하나에서 최소 약 2.29시간이며, 필요한 stage만 실행할 수 있다. 기존 raw는 원래 binary의 evidence를 유지하고 새 binary로 수행한 진단은 별도 결과로 저장한다.

[L1/L2 sector·정렬 검토](docs/cache-sector-review.ko.md)는 128-byte cache line과 32-byte 최소 접근 단위를 구분하고, offset 0/4/32/128 B와 stride 1/2/4/8/32의 비교 방법을 설명한다. `configs/cache-sector-diagnostics.json`으로 L1/L2만 실행할 수 있으며, clock 조건 하나에서 alignment 최소 24분, stride 최소 30분이다. NCU replay 시간은 별도다.

## NCU를 통한 적절성 판단

권장 순서는 **전력 sweep → 전체 조건의 NCU 검증 → 분석**이다. `validate-run`은 같은 condition의 반복 중 하나를 별도로 profile하고 모든 반복에 판정을 연결한다. 처리한 조건과 남은 조건을 manifest에 남기며 전체 energy trial을 보존한다. 고정 클럭 조건에는 전력 실행과 같은 클럭 적용 옵션을 사용한다.

```bash
source .tools/env.sh
python -m powermodeling validate-run --input results/saturation --plan saturation-plan.json --output results/validated --profiles-dir profiles --bench "$POWERBENCH" --ncu "$NCU" --apply-clocks --clock-method applications
python -m powermodeling analyze --input results/validated --output results/validated-report
```

`--limit-conditions N`은 일부 조건의 실행 점검용이다. 검증 coverage가 제한된 결과에서 전체 sweep의 verified 최적값을 확정하지 않는다. 개별 조건을 확인하거나 기존 evidence를 다시 평가하려면 `plan.json`의 `trial_id`를 사용한다.

```bash
python -m powermodeling profile --plan saturation-plan.json --trial-id TRIAL_ID --device 0 --bench "$POWERBENCH" --ncu "$NCU" --output profiles --apply-clocks --clock-method applications
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

실패·미확정 조건의 raw 에너지 측정은 보존하고 `verified_target_*` 최적값 선정에서 제외한다. HBM은 **해당 actual memory clock 이론 BW의 80% 이상**을 요구한다. 다른 workload의 상대 95% 기준은 미검증·target 실패 후보를 포함한 **전체 유효 고정 클럭 sweep의 최고 처리량**을 분모로 쓰며 검증된 후보들만으로 최고값을 낮추지 않는다. 해당 기준에 도달하는 검증 후보가 없으면 winner를 비워 두고 `verified_target_coverage`에 이유를 남긴다. 연결 대상이 아닌 trial도 새 결과 디렉터리에 그대로 보존한다. `pass`는 목표 경로의 적절성 판단이며 대역폭 포화·높은 처리량의 최소 에너지·순수 물리 회로 에너지 분리는 각각 별도 판단이다.

profiling은 전력 측정이 아니다. CUDA profiler start/stop 구간에 실제 대상 workload만 넣고 setup·initialization·warmup은 제외한다. cuBLAS의 내부 kernel 이름을 추측하지 않는다. 생성 명령은 `--profile-from-start off --cache-control none --clock-control none --replay-mode application --print-units base`를 사용하며, `--log-file`로 NCU CSV를 worker JSON과 분리한다. cache flushing을 껐다고 residency가 보장되지는 않으므로 실제 hit와 DRAM bytes를 확인한다. [공식 자료](docs/sources.md)

L2 Fabric counter는 `--extra-metrics`로 추가할 수 있다. `local-heavy`/`remote-heavy`/`mixed` 분류는 독립적으로 검증한 SM·주소·fabric 지도를 요구하며 기본값은 `unclassified`다. counter 통과만으로 해당 지도를 자동 생성하지 않는다.

현재 locality 라벨과 mapping 식별자는 분석의 repeat/group key에 포함되지 않는다. **같은 설정에서 라벨만 다른 기록은 합쳐질 수 있으므로**, 검증된 near/far 에너지 비교가 구현되었다고 해석하지 않는다. offset·`sm_ids`가 다른 설정은 기존 config key로 분리되지만 mapping별 분리를 대신하지 못한다. 아래 정합성 검토에 필요한 보완 범위를 기록했다.

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

입력·식별 요건을 충족하지 못한 fitting은 `status: "rejected"`와 이유를 JSON에 보존하고 exit code 2를 반환한다. OLS calibration이 식별되면 `status`는 `"fitted"`로 남고 계수와 실패 진단도 보존한다. `holdout_validation_status`는 별도다. validation 행이 없으면 `not_provided`, 허용 오차를 모두 넘지 않으면 `pass`, 하나라도 넘으면 `fail`이다. `fail`이면 단일 활동 예측과 mixed 예측을 모두 거부하고 `fit` CLI도 exit code 2를 반환한다. mixed 실패는 추가로 `additive_validated: false`이며, 통과한 mixed holdout의 convex hull 밖 예측도 거부한다.

`temperature_stratum`은 `status`(`qualified`/`unverified`/`rejected`), `minimum_c`, `maximum_c`, `median_c`, `maximum_span_c`(5)를 저장한다. 분석 행은 measure 구간의 최소·최대 온도를 폭에 포함한다. calibration·holdout의 알려진 온도 전체 폭이 5°C를 넘거나 일부 행만 온도가 없으면 rejected다. 온도가 모두 없으면 warning과 `unverified`만 남기고, 그 모델의 예측에는 현재 온도를 요구하지 않는다. 온도가 알려진 모델의 예측은 현재 온도를 요구하고, fitting 입력과 그 온도를 합친 폭이 5°C를 넘으면 거부한다. 더 넓은 온도 sweep를 한 모델로 맞추는 경로는 없다.

`execution_scope`는 `gpu_uuid`, `requested_clock_pairs`, `cross_clock_model`, `benchmark_sha256`, `measurement_stratum`, `treatment_design_stratum`을 저장한다. 여기의 clock은 요청값이다. `predict_power(model, features, *, context=None, allow_unbound_context=False, allow_extrapolation=False)`는 기본적으로 GPU UUID와 요청 clock context를 요구한다. `features` 안에 같은 context가 있으면 그것도 읽는다. 고정 클럭 모델은 저장된 요청 pair와 맞아야 한다. cross-clock 모델은 fit 때 각 행의 clock feature를 그 행의 config 요청 clock과 대조하고, 예측 때는 그 feature를 clock 입력으로 쓴다. `allow_unbound_context=True`는 context를 생략한 과거 진단 계산만 허용한다. 명시적 불일치, holdout 실패, mixed hull 밖 예측은 이 옵션으로 통과하지 않는다. `execution_scope`가 없는 legacy JSON도 기본 예측에서는 거부하며, 다시 fit하거나 같은 opt-in을 명시해야 한다.

현재 harness는 Tensor와 memory 활동률을 함께 측정하는 mixed-workload calibration을 자동 생성하지 않는다. 해당 workload를 별도로 실행하고 실제 활동률 counters로 feature rows를 준비해야 한다. 예측 가능한 모델의 완성 여부는 이 데이터와 독립적인 holdout 검증에 달려 있다.

실제 예측 대상의 context를 다음처럼 별도로 제공한다. clock은 모델과 같은 **요청 clock**, 온도는 현재 측정값이다. `benchmark_sha256`, `measurement_stratum`, `treatment_design_stratum`은 context에 제공된 경우 저장한 값과 비교한다. 생략된 환경 정보까지 검증하는 기능은 아니므로 적용 환경의 동질성을 확인해야 한다.

```python
from powermodeling.model import predict_power

watts = predict_power(model, measured_features, context={
    "gpu_uuid": "GPU-실제-대상-UUID",
    "graphics_clock_mhz": 1200,
    "memory_clock_mhz": 1593,
    "temperature_c": 50.0,
})
```

위 숫자는 호출 형식의 예시이며 GPU 측정값이 아니다. 반환 전력의 의미는 `model["target"]`에 따른다. `incremental_power_w` 모델은 idle 기준 증가분을 반환하므로 전체 소비전력과 구분한다. cross-clock feature의 단위는 `MHz`다. 변경 논의와 검증 결과는 [교차 검토 기록](docs/self-audit.ko.md)에 정리했다.

두 prediction 옵션은 Python boolean만 받으며 문자열 `"false"`나 숫자 `0`/`1`을 boolean으로 해석하지 않는다. clock feature와 config 요청값이 어긋난 fitting 행은 다른 provenance 오류와 같이 `skipped_rows`에 사유를 남기고 제외한다. `holdout_validation_status=pass`는 오차 통과만 뜻하며, mixed 독립성·범위·hull 검사를 대신하지 않는다.

Cross-clock 모델의 단일 활동 예측은 주파수 feature의 calibration 최소·최대 안에서 보간할 수 있다. 저장된 요청 pair 목록에 없는 조합도 그 범위 안에서는 허용되며 실제 지원 clock이나 물리적 DVFS 모델의 검증을 뜻하지 않는다. Mixed 활동은 통과한 holdout의 convex hull 제한도 충족해야 한다.

Tensor feature의 단위는 TFLOP/s, byte feature의 단위는 GB/s이다. fitting 계수의 단위는 각각 W/(TFLOP/s), W/(GB/s)이다. 같은 숫자는 전자의 경우 pJ/FLOP, 후자의 경우 nJ/byte로 변환된다. 모델은 intercept, residual, rank와 condition, 검증 오차를 보고한다. mixed 예측은 holdout이 `pass`인 독립 holdout feature들의 convex hull, 즉 실제 검증한 혼합 조건을 가중 평균해서 만들 수 있는 범위로 제한한다. holdout 하나만 통과하면 그 혼합 vector만 검증된 것이며 임의의 다른 혼합으로 확대하지 않는다.

Tensor의 클럭별 ceiling은 `SM 개수 × 실제 SM MHz × FLOP/SM/cycle × 10^-6` TFLOP/s로 계산한다. 공통 dense FP16·FP32 누산의 FLOP/SM/cycle은 V100 1,024, A100 2,048, H100 4,096을 사용한다. clock-specific issue ceiling이며, 공통 WMMA `m16n16k16`이 이 수치를 모두 달성한다는 보장은 없다. 이 ceiling은 raw PTX `mma` shape나 SASS lowering과 같은 값이 아니다.

400 W/312 TFLOPS만으로 idle 몫을 구할 수는 없다. TDP는 상한 사양이며 실제 부하 전력과 다르고, 312 TFLOPS는 특정 A100 SKU의 dense FP16 Tensor peak다. 측정한 idle, 실제 power, 실제 clock의 peak ceiling을 사용하여 idle 비율과 throughput utilization을 평가한다.

## 설계 원칙과 구현 정합성

검토 기준: **2026-10-09 UTC의 현재 작업 트리**. 아래는 이 프로그램을 설계할 때 요구한 원칙, 실험 전에 정해야 할 범위, 현재 코드가 실제로 보장하는 범위를 함께 정리한 것이다. **구현됨**은 코드·설정·합성 검증의 일치를 뜻하며 GPU 실측 완료를 뜻하지 않는다. **부분 구현 / 미구현** 항목은 실험 결과 해석과 후속 구현의 제약으로 남긴다.

### 목표: 높은 처리량에서의 에너지 단가

메모리의 같은 측정 구간에서 전력 `P`가 W, 유효 요청 대역폭 `B`가 decimal GB/s라면 `e_total = 125 × P / B` pJ/logical bit다. 대역폭이 낮으면 idle·기본 동작 전력의 몫이 커지기 쉽다. 전력 증가율보다 처리량 증가율이 클 때 단가가 개선되므로 **최대 bandwidth와 최소 pJ/bit가 항상 같은 geometry라는 가정도 하지 않는다**. 먼저 충분한 공급 병렬도와 계층별 bandwidth 기준을 확보하고, 처리량을 유지하는 후보에서 에너지를 비교한다.

HBM은 `B_theory = 2 × achieved_memory_MHz × 10^6 × bus_bits / 8` byte/s의 **80% 이상**에서 최소 pJ/logical bit를 선택한다. 버스 폭은 CUDA device metadata, memory MHz는 같은 에너지 구간의 실제 NVML 관측값이다. Peak clock 속성이나 요청 clock으로 누락값을 대체하지 않는다. **관측 peak의 95%는 HBM의 추가 탈락 조건이 아니다.** `--hbm-bandwidth-fraction 0.80`으로 기준을 명시할 수 있다. 버스 폭·클럭이 미지수인 기존 raw는 보존하되 승인 후보로 쓰지 않는다.

분자는 에너지 구간의 logical payload rate이며 물리적 HBM bus 사용률의 동시 측정은 아니다. NCU replay의 DRAM bytes를 에너지 분모로 섞지 않고, HBM 경로·coalescing을 별도 검증한다. 그 replay의 DRAM/logical 75%와 logical 80%를 곱한 60%는 같은 구간의 DRAM 전송률이 아니고, DRAM 사용률의 하한도 아니다. 80%는 사용자 선택 정책으로 물리적 포화를 뜻하지 않는다. 반복 품질·서로 다른 resource geometry·완료 count·clock·CI 조건도 유지한다.

다른 계층·연산에는 주파수별/전체 고정 클럭 sweep의 관측 peak 95% 정책이 남는다. **모든 측정이 느리면 상대 95%와 plateau도 통과할 수 있는 한계는 해당 계층에 남아 있다.** `qualified_observed_candidate`는 관측 격자 안의 승인 후보이며 `hardware_saturation_proven=false`다. [HBM 기준과 구현](docs/hbm-bandwidth-design.ko.md), [평가 구현](powermodeling/evaluation.py), [관측 최소점 분석](powermodeling/analysis.py)

### Static·dynamic과 세 가지 에너지 결과

| 구분 | 계산과 반영 | 해석의 한계 |
|---|---|---|
| 물리적 static | 주어진 전압·온도에서의 leakage를 구분해 생각한다 | 보드 센서와 idle 하나로 순수 leakage를 식별할 수 없음. Refresh는 주기적 동작이며 leakage와 구분 |
| 운영상 idle 기준 | 같은 process/context를 유지한 전후 idle 평균을 treatment 시점으로 선형 보간 | clock gating·refresh·background·context 전력이 포함되므로 `P_static`으로 명명하지 않음 |
| 전체 에너지 | 유효 counter 차분 또는 `∫P_device dt`; 완료 work로 나눔 | 기준 차감 없이 실제 부하 전체 비용을 보여 주는 기본 결과 |
| Idle 증가분 | `E_total − ∫P_idle_reference dt` | actual clock·온도·cap·상태·간섭·drift가 맞아야 최적점에 사용; 순수 physical dynamic이 아님 |
| Paired active-reference | 같은 process·할당·정책의 AB/BA arm에서 `(P_treatment−P_reference)/R_treatment` | control도 전력을 소비하고 명령·register·occupancy가 다를 수 있음; arm 길이가 다를 수 있으므로 raw energy 두 개를 단순 차감하지 않음 |
| H100 메모리 scope | 지원되는 메모리 센서 W·적분 J·동일 work 기준 단가와 별도의 idle 증가분 | 전체 GPU 값과 중복 합산하지 않음; HBM cell만의 switching 전력이라는 증명은 아님 |

음수 차분을 0으로 바꾸지 않고 진단으로 보존한다. 기준이 부적합하면 해당 차분의 사용을 제한하면서 유효한 treatment 전체 결과는 유지한다. `baseline_valid`, `operational_idle_increment_eligible`, `paired_active_reference_eligible`를 구분하며 `dynamic_attribution_eligible`도 물리적 분리를 입증하는 이름으로 읽지 않는다. SFU의 주 목적은 register control 대비 signed pJ/scalar instruction이고 전체 GPU·idle 목적은 진단용이다. [분석](powermodeling/analysis.py), [HBM 센서 검증](powermodeling/memory_power.py)

Pstate·enforced power cap은 조회되는 경우 비교하며, 모두 미지원인 것이 검증된 상태 일치를 뜻하지 않는다. 일반 차분의 group median에는 진단용 미승인 값이 포함될 수 있으므로 숫자와 함께 eligibility·반복 수·이유를 읽는다.

`idle_power / measured_load_power`는 관측 부하에서 기준 idle이 차지하는 비율이다. `idle_power / power_limit`는 전력 상한 대비 기준의 비율이다. TDP나 표기 TFLOPS를 실측 전력·처리량으로 대신하여 static/dynamic 비중을 역산하지 않는다.

### GPU 실행 구조: SM·block·warp·thread·GPC

| 대상 | 설계에서 구분할 것 | 현재 제어·관측 범위 |
|---|---|---|
| Thread / warp | warp는 32 threads. block당 thread 수와 SM당 동시 상주 thread 수는 다른 상한 | `threads`는 32의 배수인 32–1024; 실제 장치·커널의 한계를 추가 확인 |
| Block / CTA | CTA는 같은 block의 다른 이름. launch한 block 수와 동시에 상주한 block 수는 다름 | `blocks`는 전체 grid 수; `sm_count*4`는 각 SM에 정확히 4개를 배치하라는 명령이 아님 |
| SM | V100·A100·H100의 2048 threads/SM, 64 warps/SM은 상주 상한 | 실제 `max_threads_per_sm`, block 한계, SM 수와 compiled kernel 자원을 조회; 2048 threads/block을 요청하지 않음 |
| Occupancy | registers/thread, 할당 단위, shared memory, threads/block과 CTA 상한이 함께 제한 | CUDA occupancy API의 `max_active_blocks_per_sm`은 이론 상한; 실제 residency·issue 효율·bandwidth는 별도 측정 |
| SM filter | dispatch와 실행 admission을 구분 | `sm_ids`는 dispatch 이후 best-effort 허용 검사. SM ID는 sparse할 수 있고 누적 admission 수는 동시 상주 수가 아님 |
| GPC / L2 경로 | 같은 SM 번호 구간이 같은 GPC·partition이라는 가정 금지 | GPC 강제 배치나 사용하지 않는 SM의 전원 차단 기능은 없음. 실제 SM 분포와 주소 mapping 검증 필요 |

공개 설정에 `blocks_per_sm` 매개변수는 없다. 전체 launch 평균은 `blocks/SM_count`이고, 코드 내부의 같은 이름 카운터는 실행 중 누적 admitted CTA를 센다. `grid_average_blocks_per_sm`, occupancy 상한, achieved occupancy를 서로 대체하지 않는다. SM filter는 custom Tensor/memory 진단에 사용하고 cuBLAS GEMM·streaming nonlinear·register SFU에는 적용하지 않는다. [CUDA worker](cuda/gpu_bench.cu), [SFU worker](cuda/sfu.cuh), [planner 제한](powermodeling/planner.py)

`S=실제 SM 수`, `T=threads/block`, `R(T)=해당 compiled kernel의 최대 상주 blocks/SM`라 두면 thread 상한은 `T×R(T)≤2048`이다. 다른 제약이 없을 때 64/128/256/512/1024 threads는 각각 32/16/8/4/2 blocks/SM으로 2048을 채운다. 이것은 **실험 범위를 정하는 계산**이며 100% occupancy가 에너지 최적이라는 뜻은 아니다. 충분한 outstanding requests를 만들려면 대략 `in-flight bytes ≈ 목표 bytes/s × latency(s)`가 필요하므로 thread 수 외에도 load 독립성·eligible warp·issue 병목을 확인한다.

현재 `memory-read.json`은 T=128/256, 전체 grid=2S/4S/8S를 사용한다. T=128의 상단은 launch 평균 1024 threads/SM이며 T=256의 상단만 2048에 해당한다. `saturation.json`의 1S/2S/4S 범위는 T=256에서도 평균 1024이다. **따라서 현 preset이 모든 GPU의 충분한 병렬도 범위를 이미 포함한다고 보장하지 않는다.**

### 메모리 계층·sector·stride·실제 방문 범위

기본 energy read는 `stride_words=1`, `offset_bytes=0`, full warp의 scalar 4 B load다. L1/L2의 128 B line은 32 B sector 4개로 구성된다. 정렬된 warp에서 stride 1/2/4/8은 각각 4/8/16/32 sectors와 이상적인 payload 효율 100/50/25/12.5%를 만든다. 이 조건의 최소 정렬은 32 B이며 128 B 정렬은 같은 line에 들어가는 추가 조건이다. Sector 효율·cache hit·DRAM bandwidth는 각각 검사하고, stride 실험을 coalesced 에너지 최적점과 섞지 않는다. [sector 검토](docs/cache-sector-review.ko.md), [geometry 검사](powermodeling/memory.py)

| 계층 | Footprint와 경로 설계 | 실험 전 확인 |
|---|---|---|
| L1 | `.ca`, CTA별 작은 slice. V100/A100/H100의 128/192/256 KiB는 shared와 합친 최대 용량 | 대략 `동시 상주 CTA×slice`가 사용 가능한 L1에 맞는지 확인. `memory-read.json`은 CTA당 16 KiB이며 작은 4 KiB 시작점은 별도 제안 |
| L2 | `.cg`로 L1 우회, L2-resident working set | L2 hit·DRAM 유입·fabric 경로 확인. `.cg`는 L2 우회 명령이 아님 |
| HBM | `memory-read.json`은 `max(8×L2,256 MiB)` 할당, coalesced read | 유한 iterations·stride·SM filter 아래에서 실제 방문하는 범위와 DRAM traffic 확인. 큰 할당만으로 HBM 실험이 되지 않음 |

현재 read는 thread/iteration당 u32 load 하나와 합계 누산 하나를 사용한다. 주소·loop·issue 비용과 부족한 load 병렬도가 bandwidth를 제한할 수 있으므로 iterations/batching 진단과 compiled SASS·counter를 확인한다. 매 launch에서 주소 순회가 다시 시작하므로 전체 stride cycle의 footprint와 한 launch의 실제 footprint도 다르다. Write/copy의 count 규약은 별도이며 copy는 read와 write payload를 모두 센다. 데이터 seed·압축 가능성·working set·cache warm 상태도 비교 조건으로 보존한다.

V3는 일반적인 footprint에서 32-bit index 경로로 주소·제어 명령을 줄인다. CUDA 12.9의 SM70/80/90 SASS에서 L2/HBM `.cg` 16-load loop는 V2의 189개 명령에서 91–92개로 줄었다. 이 값은 **정적 명령 수**이며 실행 시간이나 에너지 감소율이 아니다. Sum32는 load 생존성을 위해 남기며 CPU oracle·입력 생성은 측정 밖이다. 같은 memory clock에서 SM clock을 조절해 80% HBM 요청률을 유지하는 조건을 찾고, 전체 전력과 H100 memory-scope 전력을 별도로 본다. 범용 XOR/산술 control을 빼서 순수 HBM 에너지로 해석하지 않는다.

### `.ca`·`.cg`·`.cs`와 짧은 HBM 진단

| 정책 | 설계 판단 |
|---|---|
| `.ca` | L1/L2 캐싱. L1 측정 기본이며 HBM에서는 L1 재사용이 결과를 바꿀 수 있음 |
| `.cg` | L1 우회·L2 캐싱. L2/HBM 기본; HBM 도달 여부는 footprint와 DRAM counter로 검증 |
| `.cs` | streaming/evict-first 힌트. L2 bypass를 보장하지 않으며 실측 전 효율 우위를 가정하지 않음 |

`--read-cache-policy auto|ca|cg|cs`는 memory read 전용이며 auto=L1 ca/L2·HBM cg다. HBM의 ca/cs는 진단으로만 실행하고 최적값 후보와 섞지 않는다. [HBM cache 진단 preset](configs/hbm-cache-policy-diagnostics.json)은 blocks=8S·T=256·stride=1·iterations=4096의 한 geometry에서 cg/ca/cs를 비교한다. 지원된 exact 1110·최대 SM clock·advertised default 고정 pair를 사용하고, 같은 memory/SM pair 안에서 비교한다. 3회 반복·paired control 없이 nominal trial 27초로, 보통 2–3 pair이면 초기화·settle·NCU 제외 약 8–12분이다. 이후 cg의 iterations=4096/16384와 필요한 geometry만 추가 진단한다. [실행 명령·공식 근거·해석 한계](docs/hbm-bandwidth-design.ko.md)

Jia 등의 Volta, Abdelkhalik 등의 Ampere, Luo 등의 Hopper microbenchmark 연구는 warmed cache, latency와 throughput 커널의 분리, 독립 load 공급과 경로 검증의 근거로 사용한다. 논문의 thread/block 수를 현재 scalar-read 커널의 에너지 최적으로 복사하지 않는다. 원문·버전·해당 절은 [연구 사례 검토](docs/cache-sector-review.ko.md)에 기록되어 있다.

### A100 20+20 MiB·H100 25+25 MiB와 near/far L2

설계에서는 전체 L2 `C`와 한 partition `P`를 구분하고 **A100 P=20 MiB, H100 P=25 MiB**를 기준으로 footprint를 검토한다. 실제 SKU와 CUDA 조회 바이트 수도 함께 기록한다. MB와 MiB를 섞지 않으며 이 두-partition 용량 가정을 V100에 자동 적용하지 않는다. 용량 구분은 연속 가상주소를 반으로 나누면 각 partition에 놓인다는 뜻이 아니다.

| Partition 기준 footprint | A100 | H100 | 용도 |
|---|---:|---:|---|
| 0.25P | 5 MiB | 6.25 MiB | 작은 cache-resident 점 |
| **0.5P** | **10 MiB** | **12.5 MiB** | 권장 첫 locality 에너지 비교점 |
| 0.75P | 15 MiB | 18.75 MiB | 용량 내 추가점 |
| 1.25P | 25 MiB | 31.25 MiB | 단일 partition 용량 초과 진단; remote 증거는 아님 |

현 `memory-read.json`의 `0.5C`는 한 partition 전체 용량 P에 해당하며, 위 첫 비교점 `0.5P`와 다르다. `locality.json`의 SM/latency 진단은 `0.125C`를 사용한다. 위 표의 범위로 preset이 자동 변경되는 것은 아니다.

Near/far 비교에는 **SM 집합↔partition↔실제 주소 집합의 검증된 지도**가 필요하다. Pointer chase로 지연시간 후보를 찾고 L2 hit·낮은 DRAM leakage·지원되는 fabric counter를 확인한 뒤, 같은 주소 집합을 두 SM 집합에서 읽는 양방향 2×2 대조를 별도의 고처리량 커널로 실행한다. 동일 clock pair·활성 SM 수·geometry·방문 bytes·반복 기준에서 `e_remote−e_local`과 비율·CI를 보고해야 한다. 일부 SM의 처리량은 같은 SM 범위의 기준과 비교한다. 다른 allocation에서 얻은 mapping을 energy run이나 NCU replay에 자동 복사하지 않는다.

**현재는 latency/offset/SM 진단까지 구현되어 있다.** Mapping별 에너지 그룹화, `local-heavy/remote-heavy/mixed/unclassified`의 독립 최적점, remote−local 단가 표는 미구현이다. `attach_verification`이 라벨을 보존해도 같은 config의 라벨만 다른 기록은 현재 합쳐질 수 있다. Near/far별 기록은 해당 보완 전까지 별도 입력·출력으로 유지하고 합친 결과를 locality 차이로 보고하지 않는다. [locality 설정](configs/locality.json), [검증 메타데이터](powermodeling/validation.py), [분석 그룹 key](powermodeling/analysis.py)

### 클럭 고정과 공정한 비교

최소 대조는 **advertised default 고정 pair**와 **동일 memory MHz의 exact 1110 MHz pair**다. Default는 NVML default applications pair를 조회한 값이며 순간 boost·최대 clock·무설정 DVFS와 다르다. Default graphics가 이미 1110이면 한 조건에 두 라벨이 붙고 독립 비교가 아니다. Exact 1110이 해당 memory domain에서 미지원이면 `not_applicable`과 이유를 기록한다. 다른 공통 memory domain을 추가한다면 native default pair를 보존하고 추가 비교임을 명시한다.

정식 sweep은 900 MHz 이상·기본 90 MHz 간격의 지원 grid, 끝점, default pair, 지원되는 1110 pair와 incoming-policy reference를 포함한다. 요청 clock과 실제 SM/memory clock·온도·throttling을 함께 본다. DVFS에서 같은 MHz가 같은 전압이라는 가정은 하지 않는다. 적용 실패를 무설정 실행으로 바꾸지 않고 이전 정책 복원을 시도한다. Application clock 복원은 policy readback을 검사하지만 locked clock의 `restored=True`는 복원 API 성공이며 이전 순간 주파수를 실측해 복원했다는 뜻은 아니다. [클럭 정책](powermodeling/clocks.py), [계획 검사](powermodeling/runner.py)

**비교 표의 현재 한계:** exact-1110 비교는 동일 memory MHz를 요구한다. 반면 factory-default 개선율은 default **pair 전체**에 대한 비교라 candidate의 memory MHz가 다를 수 있다. 이를 SM clock만 바꾼 효과로 해석하지 않는다. 동일 memory clock에서 default graphics와 1110을 직접 비교하는 요건은 pair 존재·geometry·실측 클럭을 별도로 확인해야 한다. [anchor 비교 구현](powermodeling/evaluation.py)

GPU 간에는 UUID·SXM SKU·실제 SM 수·HBM 용량/세대·ECC·MIG·power cap·온도·driver·CUDA/cuBLAS·binary·datatype·count 규약을 맞추거나 별도 층으로 보고한다. 물리 GPU와 CUDA ordinal/UUID 대응을 확인하고 MIG·다른 프로세스 간섭을 검사한다. V100/A100/H100 공통 비교 도구는 CUDA 12.9·NCU 2025.2.1이며 수동 CUDA 13 결과는 버전 층을 분리한다. 공통 WMMA와 Hopper WGMMA/TMA 또는 cuBLAS 내부 geometry도 같은 구현으로 간주하지 않는다.

### NVML API·측정 시간·NCU의 차이

| API / 값 | 단위·범위·시간 의미 | 적용 |
|---|---|---|
| `nvmlDeviceGetPowerUsage` | mW; GPU와 관련 회로. V100·A100 **GA100**은 현재 전력, H100은 약 1초 평균 | `power_w`로 환산; 폴링을 빠르게 해도 센서 갱신 주기가 빨라지지 않음 |
| `nvmlDeviceGetTotalEnergyConsumption` | driver reload 이후 누적 mJ; 지원 여부 확인 | 동일 구간 counter 차분을 우선. 감소/reset을 가로지르는 차분은 버리고 가능한 경우 power 적분으로 대체 |
| `NVML_FI_DEV_POWER_INSTANT` / `AVERAGE`, GPU scope | 요청한 field/scope의 마지막 측정값 / 최근 1초 평균 | 위 legacy API 및 module scope와 구별해 기록; 고립된 core rail로 해석하지 않음 |
| 같은 field의 MEMORY scope | 별도 메모리 subsystem 전력; H100에서도 SKU/driver 응답 확인 | 순간값 우선·평균값 fallback, mW→W. 센서 source·상태·timestamp를 유지하여 HBM 별도 결과로 제공 |
| Host `t_s` / field timestamp | 순차 NVML 호출의 monotonic midpoint / CPU epoch microseconds | 서로 다른 timebase. Host 조회 시각을 물리 센서의 정확한 획득 시각으로 부르지 않음 |
| `latencyUsec` | NVML field 갱신 latency; 여러 field에 공유될 수 있음 | 물리 센서 지연이나 1초 averaging window 길이와 동일한 값이 아님 |

기본 50 ms polling, warmup 3초, arm 12초, 전후 idle 각 6초를 사용하고 active 양 끝 2초·idle 양 끝 1초를 제외한다. 완료 batch epoch만 선택하여 work와 에너지를 같은 경계로 맞춘다. 평균 센서의 필터링이 사라지는 것은 아니며 host launch·sync·readback의 공백도 sustained 에너지/처리량에 포함된다. CUDA-event kernel rate는 별도 진단값이다. 짧은 실행·큰 sample gap·clock/온도 drift·counter 불일치는 정책에 따라 판정한다. [텔레메트리](powermodeling/telemetry.py), [구간 정렬](powermodeling/analysis.py), [공식 API 출처](docs/sources.md)

H100 HBM 센서는 W·J 측정, 동일 count의 pJ/logical bit, matched idle 증가분을 각각 승인한다. 전후 memory idle drift와 오래된 field timestamp를 별도로 검사하며 기본 freshness 허용 상한 2.5초는 프로젝트 정책이다. Timestamp 비교 근거가 없는 과거 기록은 `freshness_status=unverified`로 남는다. 실패·미지원은 null과 이유로 표시하며 instant/average 반복을 섞어 median을 만들지 않는다. 전체 GPU 채널로부터 HBM 값을 더하거나 빼서 core 전력을 생성하지 않는다. 출력 경로는 위 [H100 센서 절](#h100-hbm-메모리-전력에너지)에 정리했다.

NCU는 별도 application replay에서 setup/warmup/reference를 제외한 target을 검증한다. Binary·UUID·workload·geometry·clock·할당 준비 상태의 binding을 확인하고 profiler clock/cache control을 끈다. Replay의 전력·busy-time throughput·DRAM bytes를 원 energy run의 분자/분모로 대체하지 않는다. **현재 physical pJ/bit는 withheld**이며 logical-bit 단가와 cache/traffic 근거를 나란히 보고한다. Counter 통과는 높은 bandwidth나 순수 회로 attribution의 증명이 아니다.

### 탐색 시간을 줄이기 위한 범위와 단계

넓은 Cartesian product를 처음부터 실행하지 않는 것이 설계 목표다. 아래는 **권장 수동 절차이며 자동 adaptive sweep 기능은 아직 없다**. GPU별 occupancy와 bandwidth를 본 뒤 후보를 좁혀야 하므로 고정 thread/block 값을 사전 최적으로 선언하지 않는다.

| 단계 | 먼저 비교할 범위 | 확대 조건·주의점 |
|---|---|---|
| 준비 | 장치·clock pair·커널 자원·센서·smoke 확인 | smoke/current-policy 결과는 에너지 최적점 증거가 아님 |
| Geometry pilot | 두 고정 clock에서 T=128/256/512, grid=`S×ceil(R(T)/2)`와 `S×R(T)` | 같은 T라도 registers/shared/커널별 R이 다름. 현재 planner는 R 표현식을 지원하지 않아 수동 계산 필요 |
| 공급량 추가 | 양쪽 clock에서 유망한 T의 합집합 1–2종에 `2S×R(T)` | 계속 빨라지면 T=64/1024나 추가 wave 검토. 다른 geometry만 측정한 두 clock을 직접 비교하지 않음 |
| Footprint / ILP 진단 | stride 1·offset 0 유지, L1 작은 slice·L2 0.5P·HBM 큰 footprint 한 점부터 | residency/경로가 맞지 않을 때 footprint·iterations·batching·독립 load 공급을 한 축씩 점검 |
| 정식 에너지 | 선택한 geometry를 동일하게 유지해 충분한 arm 길이·짝수 반복·필수 clock coverage 검사 | 최소 2 geometry는 비교의 하한일 뿐; 관측 plateau에는 최소 3 resource 수준과 counter 근거 필요 |
| Locality | 검증된 mapping이 있으면 1 geometry×1 footprint×양방향 4셀×두 clock부터 | mapping 구축과 near/far별 에너지 분석은 추가 구현·측정 필요 |

Geometry pilot은 보통 **계층당 두 clock 합계 14–16조건**, 모든 T에 추가 wave를 넣으면 18조건이다. 1초 warmup+2초 timing×3회의 짧은 pilot은 worker 수준의 처리량 진단으로만 생각하며 정식 pJ 측정으로 사용하지 않는다. 현 planner는 측정 ≥10초·warmup ≥1초·idle ≥6초·반복 ≥3을 요구한다. 두 clock으로 줄인 진단 config도 full `energy_sweep`의 전체 memory domain·grid coverage를 대신하지 못한다. `--stage`는 지정한 실험을 골라 실행하는 기능이며 이전 결과를 보고 후보를 자동 제거하지 않는다.

기본 paired trial의 계획상 하한은 `3×warmup + 2×measure + 2×idle = 45초`이고 4회 반복이면 geometry×clock 한 조건당 3분이다. 현재 `memory-read.json`은 계층 3개×geometry 6개이므로 **clock 조건당 최소 54분**, 서로 다른 고정 pair 두 개만으로도 최소 108분이다. 초기화·clock/온도 안정화·NCU replay·batch 초과 실행·mapping 비용은 별도다. 실제 plan의 trial 수·`estimated_minimum_seconds`를 확인하고 stage별 중간 결과와 `--resume`을 활용한다. [계획 확장과 예산](powermodeling/planner.py)

### 모델링·반복·판정의 경계

기본 증가분 모델의 설계 형태는 `P_work = P_idle_reference + intercept + Σ(e_i×activity_i) + residual`이다. `fit`의 기본 target은 명시적으로 준비한 **기준 대비 증가분 전력**(`incremental_power_w`)이며, 전체 또는 paired 전력 target을 선택하면 해당 범위와 적합성 기준으로 회귀한다. Intercept는 남은 baseline/activation 효과일 수 있어 pure static으로 이름 붙이지 않는다. Tensor·L1·L2·HBM microbenchmark는 공유 경로를 사용하므로 단가 네 개를 더한다고 혼합 workload 전력이 복원되지 않는다.

같은 GPU·요청 clock·측정 정의에서 활동률을 독립적으로 바꾼 calibration과 rank/condition 검사가 필요하다. 온도가 알려진 행은 measure 최소·최대를 포함한 폭 5°C 안에서만 한 모델에 들어간다. 미측정 feature를 0으로 채우지 않는다. TFLOP/s 계수는 pJ/FLOP, GB/s 계수는 nJ/byte이며 후자를 pJ/bit로 바꿀 때는 125를 곱한다. 독립 mixed holdout을 통과한 범위의 convex hull 안에서만 혼합 예측을 허용한다. **Mixed workload의 자동 동시 수집은 미구현**이고 수동 feature row의 source 문자열 존재 검사는 실제 센서 scope·동일 시간창·locality의 일치를 입증하지 못한다. 입력 작성자가 그 근거와 동질성을 확인해야 한다. [모델 구현](powermodeling/model.py)

반복 중앙값과 1,000회 bootstrap 95% 구간을 보고하며 관측 수 3개 미만에는 CI를 내지 않는다. 3–4회의 구간은 거칠고 센서 calibration bias·scope 불확실성·control model 오차를 포함하지 않는다. Clock·seed·binary·실험 protocol·working set·SM 범위가 다른 자료를 반복으로 섞지 않는다. 미측정 주파수나 geometry는 보간으로 최적 구간을 만들지 않고 실제 근접 측정점과 CI 겹침을 별도로 제공한다.

### 정합성 재검토 결과와 남은 작업

| 설계 요구 | 현재 판정 | 코드 대조·남은 검증 |
|---|---|---|
| 전체/idle 증가분/paired 대비와 static·dynamic 구분 | 구현됨 | `analysis.py`, `test_analysis.py`; 순수 leakage/switching 분리는 주장하지 않음 |
| Thread/block 한계와 compiled occupancy 기록 | 구현됨 | `planner.py`, CUDA worker, `test_planner.py`; 실제 residency는 GPU에서 확인 |
| 충분히 높은 bandwidth에서 효율 비교 | **부분 구현** | HBM은 actual memory clock·bus 폭 이론 BW의 80% gate 구현. 다른 계층의 관측 95% 한계와 GPU 실측 검증은 남음 |
| Read의 연산 오버헤드와 cache 정책 | 구현됨 | V3의 32-bit index 경로·sum32·SASS 검사; HBM cg 기본, ca/cs 별도 진단. 실측 BW·pJ 개선은 미검증 |
| Occupancy에 맞춘 좁은 pilot→후보→정식 sweep | **미구현** | preset은 고정 Cartesian grid. 수동 진단은 가능하나 자동 후보 선택·R 기반 범위 생성 없음 |
| Default 고정 pair·exact 1110 포함 | 구현됨 | `planner.py`, `runner.py`, clock tests; device/driver 지원·actual MHz는 실측 필요 |
| 동일 memory clock에서 clock 효과 비교 | **부분 구현** | 1110 비교는 일치 검사. Factory-default 개선율은 memory 변경 효과를 포함할 수 있음 |
| Sector/stride와 cache·DRAM 경로 검증 | 구현됨 | `memory.py`, `validation.py`, memory tests; 높은 bandwidth나 순수 component energy는 별도 요건 |
| 20+20 / 25+25 MiB에 따른 near/far 단가 분리 | **부분 구현** | latency/SM/offset 진단만 있음. 검증된 mapping key·별도 그룹·remote−local 결과 필요 |
| NVML API 의미와 counter/시간창 구분 | 구현됨 | `telemetry.py`, `analysis.py`, telemetry/analysis tests; 평균 센서 필터와 bias는 잔존 |
| H100 HBM 메모리 센서의 별도 W·J·pJ 보고 | 구현됨 | `memory_power.py`, sensor/report tests; 센서 지원·HBM 경로와 수치 정확도는 장비에서 확인 |
| 실험 정의·단위·rank·holdout에 근거한 모델 | **부분 구현** | `model.py`의 입력/식별/예측 gate는 있음. 동시 mixed 수집과 수동 provenance 검증은 별도 필요 |
| 재현 가능한 도구·결과·오류 보존 | 구현됨 | 설치 스크립트, raw JSON, plan, binary/toolkit 정보, `--resume`; 실제 GPU 측정 검증을 대체하지 않음 |

기존 HBM 상대 기준은 peak 100 bytes/s인 합성 sweep도 승인할 수 있었으나, 이제 유효한 장치 이론 BW의 80%를 충족해야 한다. 같은 L2 설정의 repeat에 서로 다른 검증 locality 라벨만 붙였을 때 한 group으로 합쳐지는 한계는 남아 있다. 이는 **정책/그룹화의 합성 사례**이며 GPU 실측 숫자가 아니다. 후속 보완은 다른 계층의 bandwidth 기준과 pilot 자동화, mapping별 locality 그룹/비교, 동일 memory clock의 default 대조, mixed calibration 자동화다.

## 검증 및 상세 설계

설치와 같은 `.venv` interpreter로 CPU 테스트를 실행한다. 아래 명령은 가상환경 활성화 여부와 관계없이 해당 환경을 사용한다.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

`--build`까지 완료했으면 GPU 없이 컴파일된 CLI의 입력 검증도 실행할 수 있다.

```bash
source .tools/env.sh
POWERBENCH_CLI_TESTS=1 python -m unittest discover -s tests -v
```

CPU 테스트는 데이터 분석·모델 식별·plan·NVML mock·클럭 복원을 검증한다. CUDA 컴파일은 GPU 없는 호스트에서도 가능하지만, actual GPU 실행·NCU profiling·cache attribution과 측정 정확도는 V100/A100/H100 장비에서 확인해야 한다.

- [실험 설계: static/dynamic 기준, DVFS, hierarchy, near/far, fairness](docs/experiment-design.ko.md)
- [비선형 SFU 마이크로벤치: register loop·차분 pJ/instruction·control·SASS 검증](docs/nonlinear-experiments.ko.md)
- [이전 streaming 전체 함수 실험: EXP·TANH·RMSNorm·Softmax·SiLU](docs/legacy/nonlinear-streaming.ko.md)
- [컴포넌트별 평가·시각화: coverage·반복 정밀도·plateau·에너지 후보](docs/evaluation-design.ko.md)
- [높은 Tensor·L1·L2·HBM 단가: 기준·계산·처리량 점검과 추가 진단](docs/high-energy-investigation.ko.md)
- [전체 구현 자가점검: 발견 사항·수정·검증·남은 실측](docs/self-audit.ko.md)
- [NVIDIA 공식 출처 및 검증이 필요한 주장](docs/sources.md)
