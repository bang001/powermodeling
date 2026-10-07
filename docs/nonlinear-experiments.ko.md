# 비선형 함수의 에너지 실험

이 문서는 global 입출력을 포함한 **전체 함수 비용**을 다룬다. SFU 기본 명령의
register-resident 반복과 대응 control의 차분은 별도
[SFU register 실험](sfu-register-experiments.ko.md)을 사용한다.

EXP, TANH, SiLU, RMSNorm, Softmax를 `powerbench`의 독립 workload로 측정한다.
현재 구현은 **FP32 입력·출력, CUDA 표준 math, 실제 global load/store를 포함한
완전한 함수 적용**이다. `--use_fast_math`는 사용하지 않는다. FP16/BF16,
근사 math, register-only SFU 반복, framework/vendor fused kernel의 비용은
다른 실험이며 이 결과로 대신하지 않는다.

## 함수와 분모

| workload | 정의 | 주 분모 | 추가 단위 | 원소당 논리적 payload |
|---|---|---|---|---|
| `exp` | `expf(x)` | 출력 원소 1개 | 함수 적용 1회와 같음 | 입력 4 B + 출력 4 B |
| `tanh` | `tanhf(x)` | 출력 원소 1개 | 함수 적용 1회와 같음 | 입력 4 B + 출력 4 B |
| `silu` | `x / (1 + expf(-x))` | 출력 원소 1개 | 완전한 SiLU 적용 1회 | 입력 4 B + 출력 4 B |
| `rmsnorm` | `x[i] * gamma[i] / sqrt(mean(x²) + 1e-5)` | 완전한 RMSNorm의 출력 원소 1개 | 행 전체 적용 1회 | x 두 번 + gamma 한 번 + 출력 = 16 B |
| `softmax` | `exp(x[i]-max(x)) / sum(exp(x-max(x)))` | 완전한 Softmax의 출력 원소 1개 | 행 전체 적용 1회 | x 세 번 + 출력 = 16 B |

RMSNorm은 평균을 빼지 않고, 열별 gamma를 포함한다. Softmax는 행별 max와
합계를 reduction한 뒤 출력을 쓴다. exp 값을 저장하는 임시 버퍼를 쓰지 않아
원소당 exp를 두 번 평가하는 **3-pass 구현**이다. 따라서 그 pJ/element에는
두 exp 평가, reduction, 정규화, 메모리 접근이 모두 포함된다.

`operations`는 이 workload에서 **출력 원소 수**를 뜻한다. 이를 FP FLOP 수나
MUFU/SFU instruction 수로 변환하지 않는다. `row_evaluations`는 RMSNorm과
Softmax의 완전한 행 적용 횟수다. 행 너비 N에서 `pJ/row = N * pJ/element`이며
같은 함수라도 N이 다른 결과는 서로 다른 실험 층으로 비교한다.
[컴포넌트 평가·시각화 설계](evaluation-design.ko.md)의 coverage·정밀도·plateau·
default/1110 개선 판정을 적용하며 `evaluation.html`에서 함수/footprint/행 너비를
선택한다. `--plots`는 CI와 anchor·품질을 표시한 PNG/SVG를 생성한다.

## 커널·정확성·카운트

`Q = working_set_bytes / 4`를 입력 원소 수로 정의한다. 기본 `grid_mode: "auto"`는
pointwise EXP·TANH·SiLU에 `ceil(Q/threads)`개 CTA를 실행한다. 각 thread가
전역 인덱스의 원소를 처리하고 마지막 CTA의 Q 밖 thread는 쓰지 않는다.
RMSNorm·Softmax는 `Q/row_width`개 CTA를 실행하며 CTA 하나가 행 하나의
reduction과 출력을 담당한다. Q는 행 너비의 배수여야 한다. 행 너비는 1..65536이고 thread 수보다
크거나 warp 크기의 배수가 아니어도 된다. reduction은 96 같은 non-power-of-two
thread 수를 지원한다. 전체 Q의 처리와 출력 표본 검증을 위해 nonlinear workload에는
SM admission filter, stride 변경, 주소 offset을 허용하지 않는다.

Q를 생략하면 GPU SM 수·threads와 무관하게 1,048,576 원소(입력 4 MiB)다.
자동 grid와 다른 `blocks`를 함께 지정하면 오류다. 계산된 grid가 1,000,000개
CTA 또는 장치 grid 한도를 넘으면 오류를 내며 몰래 잘라내지 않는다.
`grid_mode: "fixed"`는 명시적 양의 `blocks`를 요구하고 pointwise grid-stride,
rowwise row-stride로 Q 전체를 처리한다. Q가 blocks의 배수일 필요는 없다.

이전 v1의 2/4×SM grid도 Q 전체를 처리했지만, Q가 커질수록 CTA 내부의 순차
반복만 길어졌다. 제한된 grid로는 큰 Q에서의 병렬성 확대를 평가할 수 없었다.
다만 4 blocks/SM은 실제 residency 지정이 아니며 일부 SM만 사용했다는 뜻도
아니다. Q 기반 grid 역시 자동 포화를 보장하지 않아 별도 성능 검증이 필요하다.

예를 들어 132 SM 장치의 `4×SM`은 528 CTA다. Q=2²⁴, threads=256인
pointwise auto grid는 65,536 CTA를 준비한다. 실제 동시 residency는 registers,
threads, 장치 한도에 따라 결정되지만, scheduler가 처리할 CTA 수가 Q와 함께
늘어난다는 차이가 있다. 작은 고정 grid가 낮은 처리량과 높은 pJ/element의 원인이
될 수는 있어도, 실측 전력·완료 처리량 없이 에너지 차이의 크기나 원인을 확정할 수 없다.

`working_set_bytes`는 **입력 전체 크기**이고 출력은 같은 크기의 별도 버퍼다.
RMSNorm은 너비 N의 FP32 gamma 벡터도 할당한다. 입력은 seed로 정해진 FP32
[-4,4) 값, gamma는 [0.75,1.25) 값이다. Softmax 입력은 [96,104)로 이동시켜
max 차감 없이 exp를 적용하면 overflow하는 경우도 검증한다. 안정화 후 exp의
입력 범위는 [-8,0]이다. 실제 분포와 math 구현은 result에 기록한다.

`iterations`마다 같은 입력에 **함수 전체를 다시 적용**한다. inline PTX global
load/store와 출력의 데이터 의존성을 사용하며, 컴파일된 코드에서도 반복과
load/store가 유지되는지 확인해야 한다. 원소를 반복 처리한 횟수도 분모에 포함되며, 고유 원소 수와는
다르다. 작은 footprint는 cache에 머물 수 있다. payload는 실제 DRAM traffic과
다르므로 메모리 계층이나 함수만의 물리 에너지로 단정하지 않는다.

V2는 **동기화로 완료를 확인한 kernel launches × Q × iterations**로 각 work
epoch의 `elements`와 `operations`를 계산한다. 마지막 CTA의 padding을
원소 수에 더하지 않는다. 완료 CTA 수는 launches × blocks로 계산하고
`block_completion_count_source: "synchronized_completed_launches"`로 기록한다.
Nonlinear treatment와 paired control의 CTA별 SM admission atomic을 제거하여
대규모 grid에서 계측용 atomic 경합이 새 병목이 되는 것을 피한다. 실제 SM별
분포는 측정하지 않았으므로 unknown이며, 균등 분포로 만들어 기록하지 않는다.
Power 적분과 **같은 완전한 epoch**의 원소 수만
에너지 분모로 사용한다. 잘못된 element/row/byte count, 부분 행, 수치 검증
실패, 부정확한 epoch는 분석에서 invalid 처리하고 pJ/element를 내지 않는다.

실행 전과 측정 후, 입력/출력을 CPU double 기준값과
비교한다. Pointwise는 Q의 처음·중간·끝과 경계 표본, rowwise는 분산한 행 전체를
검사한다. 허용 오차는 `2e-6 + 2e-4 * abs(reference)`이고 Softmax의 행 합계도
1과 비교한다. 검사 횟수·오차·pass 여부를 기록하며 실패한 worker는 nonzero로
종료한다. 이는 대표 표본 검증이며 전체 출력이나 모든 입력 범위의 증명이 아니다.

V2는 `math_implementation: "cuda_fp32_q_grid_v2"`,
`kernel_implementation_version: "fp32_complete_nonlinear_q_grid_v2"`다.
기존 v1 기록은 원래 admission × slice 계약으로 읽고 V2와 합치지 않는다.
`fixed`는 새 커널의 제한 grid 진단이며 v1의 배치·계측을 그대로 재현하지 않는다.

## 에너지와 비교 설계

다음 세 값을 독립적으로 유지한다.

- `total_pj_per_element`: 해당 GPU의 측정 scope 전체 에너지 / 처리 원소 수.
- `operational_idle_increment_pj_per_element`: 같은 clock·온도·power cap 조건으로
  승인된 전후 powered idle 대비. 기준 불일치 시 eligibility는 false다.
- `paired_active_reference_pj_per_element`: 같은 process·할당·clock·geometry에서
  AB/BA로 실행한 integer issue-loop reference 대비. memory/reduction과 실제
  instruction mix는 일치하지 않으므로 함수만의 순수 동적 에너지가 아니다.

행 함수에는 각 `*_pj_per_row`도 제공한다. 기존 `*_pj_per_op` 별칭에서 op는
이 workload의 출력 원소이며, 행 전체 연산의 단가는 `*_pj_per_row`로 읽는다.
Signed contrast는 보존하지만 음수 대비를 효율 최적점으로 선택하지 않는다.

`configs/nonlinear-smoke.json`은 5개 함수 × 4 repeats = 20 trials다. 각 조건은
AB/BA를 2회씩 실행한다. 최소 시간은 15분이며 준비·overrun·프로파일링은 제외한다.
무설정 clock smoke는 기능 점검이고 고정 clock 최적점을 확정하지 않는다.

`configs/nonlinear.json`은 pointwise와 rowwise stage, auto grid, threads
128/256/512, Q=2²⁴/2²⁵/2²⁶(입력 64/128/256 MiB), 행 너비
128/1024/4096을 제공한다. 해당 장치의
모든 지원 memory domain의 900 MHz 이상 graphics grid(기본 90 MHz, 60/120 등 선택), advertised default와 지원되는
exact 1110 MHz 및 incoming-policy reference를 교차한다. 입력 크기의 의미는
실제 L2 크기·traffic을 확인해 판단한다. 기본 값은 출발점이다.
큰 Q에서 batch overrun이 생길 수 있으므로 batch duration,
epoch 길이와 power sample alignment를 확인한다.

Clock 조건 하나당 81개 조건 × 4 repeats × 45초 = 최소 **4.05시간**이다.
Pointwise만 1.35시간, rowwise만 2.70시간이다. GPU별 실제 지원 clock 조건 수와
준비·overrun·별도 NCU 시간을 반영해야 하므로 plan의 예상 시간을 확인한다.

비교 층은 GPU UUID·binary·clock·정밀도·math 구현·입력 분포·footprint·행 너비·
RMSNorm 정의·grid mode·구현 버전을 분리한다. 반복 median과 bootstrap CI, 고정 clock별 실제 최고
처리량과 비교한 조건, 주변 측정점은 기존 summary 형식을 따른다. 검증된 두 개
이상의 실행 geometry 없이 포화/최적점을 확정하지 않는다. seed/크기만 바꾼
조건은 독립적인 실행 resource geometry로 세지 않는다.
주 에너지 그래프와 후보는 정의·행 너비·Q별로 나누고 pJ/element와 Gelement/s를
사용한다. 별도 Q-scaling 진단에서는 같은 threads·clock·구현·함수·행 너비의
Q만 바꿔 비교한다. 가장 큰 유효 Q 3개 모두 경로·count·반복·CI·objective 요건을
통과하고, 전체 유효 Q 곡선의 관측 peak 95% 이상이며 처리량 폭이 5% 이내일 때
`observed_input_size_plateau`로 표시한다. 선택 후보의 Q도 이 구간에 있어야 한다.
미검증인 빠른 조건을 peak 분모에서 빼거나 큰 Q의 실패를 숨겨 plateau를 만들지 않는다.
이 근거는 같은 Q 안의 최소 2개 geometry·클럭별/전체 95% 기준을 대체하지 않는다.

Auto pointwise에서는 blocks×threads가 거의 Q이므로 threads를 바꿔도 자원 축이
증가하지 않는다. 따라서 예전 resource plateau를 그대로 적용하지 않고 후보별
Q-scaling 근거를 사용한다. 큰 Q는 cache 사용과 launch 비용 비중도 바꾸므로
처리량 안정이 곧 SFU 하드웨어 포화라는 뜻은 아니다. NCU의 active warps,
DRAM throughput과 지원되는 SFU activity counter를 함께 확인한다. Counter가
없으면 unknown으로 남긴다. `fixed`와 v1은 기존 resource plateau 판정을 유지한다.

`evaluation.html`의 Q scaling 그림·표와 `evaluation.json`의 `input_scaling`에서
Q, blocks, threads, 완료 count 출처, 처리량/에너지 CI와 제외 이유를 확인한다.
`--plots`는 같은 곡선을 중복 없이
`<workload>-q-curve-<curve_id>-<objective>-input-scaling.png/.svg`로 저장한다.

## 실행

Linux와 전용 NVIDIA GPU, driver/NVML, CMake가 필요하다.
V100/A100/H100 공통 빌드는 CUDA 12.x, 공통 NCU는 GV100을 지원하는 2025.2.x
등 호환 버전을 사용한다. **A100은 CUDA 13.0 + `sm_80`을 지원**하며 이때
CUDA 13.0을 지원하는 Linux R580 이상 driver와 NCU 2025.3 이상을 사용한다.
다음 두 빌드 중 장치·Toolkit에 맞는 하나를 실행한다.

```bash
cd /workspace/powermodeling
source .venv/bin/activate
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES='70;80;90'
cmake --build build -j2
export POWERBENCH=build/powerbench
```

```bash
# A100 전용 CUDA 13.0 빌드. 실제 Toolkit 경로에 맞춰 지정한다.
cd /workspace/powermodeling
source .venv/bin/activate
cmake -S . -B build-a100-cuda13 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-a100-cuda13 -j2
export POWERBENCH=build-a100-cuda13/powerbench
```

두 경로 모두 아래 명령으로 EXP·TANH·SiLU·RMSNorm·Softmax를 실행한다.
`POWERBENCH`는 선택한 빌드와 GPU 테스트·runner가 같은 바이너리를 사용하게 한다.

```bash
# 짧은 실제 CUDA 수치·카운트 검증; 에너지를 측정하는 테스트는 아님
POWERBENCH_GPU_TESTS=1 python -m unittest discover -s tests -p test_nonlinear.py -v
POWERBENCH_GPU_TESTS=1 python -m unittest discover -s tests -p test_nonlinear_q_grid.py -v

# 기존 runner의 NVML 센서·독점 사용·clock 복원 검사를 그대로 사용
python -m powermodeling discover --device 0 --bench "$POWERBENCH" --output results/nonlinear-discovery.json
python -m powermodeling plan --config configs/nonlinear-smoke.json --device 0 --bench "$POWERBENCH" --output results/nonlinear-smoke-plan.json
python -m powermodeling run --plan results/nonlinear-smoke-plan.json --device 0 --bench "$POWERBENCH" --output results/nonlinear-smoke
python -m powermodeling analyze --input results/nonlinear-smoke --output results/nonlinear-smoke-report --plots

# 실측과 분리한 NCU replay를 condition별로 실행하고 근거를 붙임
python -m powermodeling validate-run --plan results/nonlinear-smoke-plan.json --input results/nonlinear-smoke --output results/nonlinear-validated --profiles-dir results/nonlinear-profiles --device 0 --bench "$POWERBENCH"
python -m powermodeling analyze --input results/nonlinear-validated --plan results/nonlinear-smoke-plan.json --output results/nonlinear-validated-report --plots

# 전체 sweep: 먼저 trial 수와 예상 시간을 검토. --stage pointwise/rowwise로 분할 가능
python -m powermodeling plan --config configs/nonlinear.json --device 0 --bench "$POWERBENCH" --output results/nonlinear-plan.json
python -m powermodeling run --plan results/nonlinear-plan.json --device 0 --bench "$POWERBENCH" --output results/nonlinear --apply-clocks --clock-method applications
```

고정 clock은 전용 GPU의 기존 정책을 읽고 복원할 수 있을 때 적용한다. NCU도
같은 요청 clock으로 검증할 때 `validate-run --apply-clocks`를 사용한다. 상세한
locked-clock 복원 옵션은 README를 따른다. 전체 sweep를 소수 trial로 잘라 얻은
결과는 불완전한 coverage를 갖는다. 결과 폴더는 원본/검증 후/분석 출력을 분리한다.
CUDA 12와 13은 compiler/runtime/cuBLAS 버전이 다르므로 동일한 반복 표본으로
합치지 않는다. 버전과 binary가 분석 층에 포함되어 결과가 분리된다.

NCU는 해당 nonlinear kernel 이름, 양의 SFU instruction activity, spill·duration,
count·수치 검증·profile/energy parameter binding을 확인한다. 필요한 counter가
지원되지 않으면 inconclusive로 남긴다. SFU activity 통과는 전체 구현 경로의
근거이며 SFU의 물리 에너지나 처리량 포화를 증명하지 않는다.

## 현재 검증 범위

CPU 테스트는 pJ/element·pJ/row 단위, epoch/count 오류 차단, 계획/CLI/그래프,
NCU 근거의 수치 판정과 행 너비 binding을 검증한다. GPU opt-in 테스트는 실제
worker에서 모든 함수와 너비 1·129·1024, 96 threads, 반복 2회를 검사한다.
Q-grid 테스트는 Q=1, threads−1, threads, threads+1 등의 tail, fixed grid의
남는 행, launch/epoch count, padding 분모 거부도 검사한다.
GPU가 없는 개발 환경에서는 실제 GPU 테스트를 명시적으로 skip한다.

현재 Cloud에서 CMake 3.31.10으로 다음 전체 컴파일·링크를 통과했다.

- CUDA 12.9.86: V100/A100/H100 `sm_70;sm_80;sm_90`.
- CUDA 13.0.48: A100 전용 `sm_80`.

모든 빌드에서 5개 비선형 커널의 ptxas 보고서에 spill load/store가 없었다.
SASS에서도 입력 load·함수 계산·출력 store가 traversal/repeat loop에 남고,
nonlinear treatment와 전용 control에 admission atomic·SMID 명령이 없는 것을
확인했다. Tensor·메모리·latency·기존 control의 SASS는 변경 전과 같았다.
CUDA 13 실행 파일에 실제 `sm_80` cubin/PTX가 포함되고 CUDA runtime/cuBLAS 13
라이브러리로 연결되는 것도 확인했다. 기본 아키텍처·명시적 override·환경변수
우선순위·CUDA 13의 Volta 거부를 실제 CMake configure로 확인했다.
이 결과는 컴파일·CPU 검증이며 runtime NCU 검증을 대신하지 않는다.
각 실행 파일의 `--help`도 동작한다. `--describe`는 NVIDIA driver가 없는
머신에서 CUDA driver/runtime 오류로 실패했다.

이 Cloud의 컴파일 도구와 호환 glibc sysroot는 checkout 밖에 설치되어 있다.
이미 준비된 도구로 다시 빌드할 때는 다음 명령을 사용할 수 있다. 일반 GPU
서버에서는 앞의 표준 CMake 명령을 사용한다.

```bash
.venv/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/workspace/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/workspace/powermodeling-onboarding/cuda-host-g++ \
  -DCUDAToolkit_ROOT=/workspace/cuda-12.9 \
  -DCMAKE_BUILD_RPATH=/workspace/cuda-12.9/targets/x86_64-linux/lib \
  -DCMAKE_CUDA_ARCHITECTURES='70;80;90'
.venv/bin/cmake --build build -j2
```

현재 Cloud의 A100/CUDA 13.0 컴파일 경로는 다음과 같다.

```bash
.venv/bin/cmake -S . -B build-a100-cuda13 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/workspace/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/workspace/powermodeling-onboarding/cuda-host-g++ \
  -DCUDAToolkit_ROOT=/workspace/cuda-13.0 \
  -DCMAKE_BUILD_RPATH=/workspace/cuda-13.0/targets/x86_64-linux/lib \
  -DCMAKE_CUDA_ARCHITECTURES=80
.venv/bin/cmake --build build-a100-cuda13 -j2
build-a100-cuda13/powerbench --help
```

이 저장소에는 이 기능의 실측 pJ 숫자가 아직 없다. 실제 센서 정확도, cache/DRAM
traffic, 컴파일 후 instruction mix, 처리량 plateau와 GPU별 에너지 단가는
전용 GPU에서 위 실험을 실행한 후 판단해야 한다.
