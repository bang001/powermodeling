# Register-resident SFU 기본 명령 실험

이 실험의 주 결과는 **SFU 명령이 있는 register loop와, 같은 반복 구조에서
그 명령을 뺀 control 사이의 차분 pJ/scalar instruction**이다. 기존
`configs/nonlinear*.json`은 global 입출력을 포함한 전체 함수의 pJ/element이며
이 목적에 사용하지 않는다. 두 결과는 workload·분모·구현·reference 정의로 분리한다.

## 직접 측정하는 연산

| workload | 의미 | PTX 명령 | V100 | A100/H100 |
|---|---|---|---|---|
| `sfu_ex2` | 2ˣ | `ex2.approx.ftz.f32` | 지원 | 지원 |
| `sfu_lg2` | log₂x | `lg2.approx.ftz.f32` | 지원 | 지원 |
| `sfu_rcp` | 1/x | `rcp.approx.ftz.f32` | 지원 | 지원 |
| `sfu_rsqrt` | 1/√x | `rsqrt.approx.ftz.f32` | 지원 | 지원 |
| `sfu_sqrt` | √x | `sqrt.approx.ftz.f32` | 지원 | 지원 |
| `sfu_tanh` | tanh(x) | `tanh.approx.f32` | native 미지원 | 지원 |

FP32 **근사** 명령을 측정한다. EX2는 자연 지수 EXP/표준 `expf`가 아니며,
LG2도 자연 로그가 아니다. `.ftz`와 입력 분포를 raw 결과에 기록한다.
사용하는 입력은 양의 normal FP32이므로 subnormal/NaN/Inf 비용은 이 실험의 범위 밖이다.
TANH는 `sm_75` 이상을 요구한다. V100에서는 소프트웨어 식으로 대체하지 않고
`plan.unsupported_experiments`에 미지원 이유를 남긴다. 공통 설정은
`skip_if_unsupported: true`를 TANH에만 지정한다. 직접 요구하면 오류다.

CUDA 12.9의 sm70/80/90과 CUDA 13.0의 sm80에서 각 지원 PTX 명령이 대응하는
MUFU 한 개로 컴파일되는 것을 확인한다. SIN/COS의 PTX 근사 명령에는 입력 변환
FMUL이 추가되어 이번 단일 native 명령 실험에서 제외한다. RMSNorm·Softmax·SiLU는
산술·reduction이 필요한 복합 함수이며 이 표의 명령 하나로 완성되지 않는다.
여러 primitive의 단가를 더한 값을 전체 함수의 실측 에너지로 표시하지 않는다.

## Register loop와 완료 수

`sfu_lanes`는 논리적 작업 lane 수 Q다. global 입력 버퍼는 없고 lane/seed로
register를 초기화한다. 기본 Q=262144, threads=256, `sfu_chains=4`,
`iterations=16384`, `batch_launches=1`이다. `grid_mode=auto`는 `ceil(Q/threads)`
블록을 실행하며 마지막 블록의 Q 밖 thread는 제외한다. `fixed`는 명시적
blocks를 요구하고 grid-stride로 Q 전체를 처리한다. Q는 메모리 footprint가 아니다.

Thread마다 1/4/8개의 독립 register chain을 유지한다. 매 iteration/chain에서
native SFU 명령을 한 번 수행하고, 결과의 비트에 iteration nonce를 XOR한 뒤
mantissa/exponent를 조합해 다음 입력을 **[0.5,1)**로 제한한다. EX2의 반복 overflow,
TANH·RSQ의 단순 고정점 수렴, RCP의 역수 쌍 단순화를 피하고 모든 결과를 다음
iteration에 연결한다. 이는 일반 입력 분포 전체를 대표하지 않는 명시적 실험 분포다.

Control은 같은 초기화·chain·XOR/remap·loop·최종 hash를 사용하고 SFU 명령을
PTX MOV로 대체한다. MOV는 compiler에서 없어질 수 있으므로 실행된 MOV의 비용을
뺀다고 주장하지 않는다. 실제 SASS에서 **SFU가 없는 공통 register loop**인지
검사한다. SFU 결과가 다르므로 두 arm의 operand trajectory는 동일하지 않다.

반복 내부에는 global/shared/local load/store, admission atomic, SMID 계측이 없다.
모든 chain의 최종 값을 hash해 **loop 후 lane당 uint32 store 한 번**을 수행한다.
완전한 memory-free kernel이라고 부르지는 않는다. 초기화·loop·hash·최종 store·
launch 비용도 측정 시간에 포함되며, store의 발행량은 명령당
`4 / (iterations × chains)` byte다. `sfu-register-amortization.json`은 iteration을
늘려 이 고정 비용의 영향이 작아지는지 확인한다.

분모는 아래와 같으며, CUDA 동기화로 완료된 launch/epoch에만 적용한다.

```
scalar SFU instructions = completed launches × Q × iterations × chains
```

`operations`와 `sfu_instructions`가 이 수를 나타내고 `elements`, `logical_bytes`는
0이다. padding을 세지 않는다. Control의 `reference_loop_slots`는 대응하는
반복 slot 수이고 실제 `sfu_instructions`는 0이다. NCU의 SFU instruction counter는
warp 단위로 검증하므로 `launches × ceil(Q/32) × iterations × chains`와 비교한다.
Warp count를 scalar 에너지 분모에 그대로 넣지 않는다.

실행 전후 one-step native 출력 표본을 CPU double 기준으로 검사한다. 근사 연산
결과의 비트가 다음 입력으로 쓰이는 전체 recurrence는 CPU double과 동일한
경로라고 가정하지 않는다. 최종 hash 표본과 one-step 수치 검사의 범위를 구분한다.

## 주 결과와 해석 범위

동일 process·allocation·Q·chain·iterations·grid·clock 정책의 treatment/control을
AB/BA로 균형 있게 실행한다. 양쪽 모두 warmup 후 기본 12초씩 측정한다.
각 arm의 완료 epoch와 전력을 따로 정렬한다. 주 결과는 다음과 같다.

```
R_T = treatment의 완료 scalar SFU instruction 수 / treatment 시간
sfu_reference_delta_pj_per_instruction = (P_T − P_C) / R_T × 10^12
```

**주 결과는 idle을 그냥 뺀 값이 아니다.** `P_C`는 SFU를 제거한 register-loop
control의 실측 전력이다. 전체 GPU 단가와 idle 증가분은 별도 진단값으로만 보존한다.
에너지 추천은 register-control 차분에 대해서만 제공한다. 음수와 0을 포함하는
신뢰구간을 숨기거나 0으로 잘라내지 않으며, 양의 명령 비용을 확인한 최적점으로
선정하지 않는다. 반복별 차분의 중앙값과 bootstrap CI를 사용한다.

이 식은 **같은 시간 기준의 평균 전력 차분을 treatment 처리율로 나눈 operational
estimate**다. `(E_T−E_C)/동일 작업 수`가 아니다. SFU가 없는 control은 더 빠를 수
있고 초당 remap/loop/launch 횟수도 다르다. Register 수·theoretical residency도
서로 다를 수 있어 두 arm의 자원·지속 시간·완료 수·rate를 함께 남긴다.
Register file, instruction issue, scheduling, clock/thermal state의 영향을 완전히
제거한 **물리 SFU rail의 절대 에너지**라고 주장하지 않는다.

Q·threads·chains를 바꿔 처리량을 비교한다. Q별 에너지 통계는 분리하며, 같은
clock·threads·chains·iterations에서 큰 Q 구간의 처리량 안정 여부를 별도로
평가한다. 경로 증거·완료 수·정밀도·관측 peak 95%와 Q plateau를 통과해도
관측 범위의 후보이지 SFU 하드웨어 포화나 물리 에너지 분리의 증명은 아니다.

## 실행

[README의 CUDA 빌드](../README.md)를 먼저 수행한다. V100은 CUDA 12,
A100은 CUDA 13.0/sm80 빌드도 지원한다. 아래 예시는 전용 GPU에서 실행한다.

```bash
export POWERBENCH=build/powerbench
# A100/CUDA13: export POWERBENCH=build-a100-cuda13/powerbench

# 실제 GPU에서 tail·chain·native primitive·paired count 검사; 전력 측정과 별도
POWERBENCH_GPU_TESTS=1 python -m unittest discover -s tests -p test_sfu.py -v

# 실제 사용할 binary의 SASS 증거. cuobjdump/nvdisasm은 해당 아키텍처 호환 버전 사용.
python tools/check_sfu_sass.py --binary "$POWERBENCH" \
  --output results/sfu-sass.json

python -m powermodeling plan --config configs/sfu-register-smoke.json \
  --bench "$POWERBENCH" --device 0 --sfu-sass-evidence results/sfu-sass.json \
  --output results/sfu-smoke-plan.json
python -m powermodeling run --plan results/sfu-smoke-plan.json \
  --bench "$POWERBENCH" --device 0 --output results/sfu-smoke
python -m powermodeling analyze --input results/sfu-smoke \
  --output results/sfu-smoke-report --plots

# 전체 fixed-clock sweep
python -m powermodeling plan --config configs/sfu-register.json \
  --bench "$POWERBENCH" --device 0 --sfu-sass-evidence results/sfu-sass.json \
  --output results/sfu-plan.json
python -m powermodeling run --plan results/sfu-plan.json --bench "$POWERBENCH" \
  --device 0 --output results/sfu --apply-clocks --clock-method applications
python -m powermodeling validate-run --plan results/sfu-plan.json \
  --input results/sfu --output results/sfu-validated --profiles-dir results/sfu-profiles \
  --bench "$POWERBENCH" --device 0 --apply-clocks --clock-method applications
python -m powermodeling analyze --input results/sfu-validated --plan results/sfu-plan.json \
  --output results/sfu-report --plots
```

NCU 호환 버전을 `--ncu`로 지정할 수 있다. V100에는 Volta를 지원하는 2025.2.x,
A100/CUDA 13에는 CUDA 13을 지원하는 2025.3 이상이 필요하다. Clock 적용·복원
정책은 기존 runner와 같다. SASS certificate는 binary SHA256·architecture·primitive·
chain에 연결되며, 기록된 pass flag를 신뢰하지 않고 raw SASS/resource를 재검사한다.
증거가 없거나 바뀐 binary이면 SFU 경로 검증을 승인하지 않는다.

| 설정 | 목적 | 최소 시간 |
|---|---|---|
| `sfu-register-smoke.json` | 5/6 primitive × 4 repeats, 현재 clock 정책 기능 점검 | V100 15분, A100/H100 18분 |
| `sfu-register.json` | Q=131072/262144/524288 × threads=128/256 × chains=1/4/8, clock sweep | clock 조건 하나당 V100 4.5시간, A100/H100 5.4시간 |
| `sfu-register-amortization.json` | Q·threads·chains 고정, iterations=4096/8192/16384/32768 | V100 60분, A100/H100 72분 |

시간은 준비·overrun·NCU를 제외한 계획 하한이다. Amortization 설정의 null clock은
진단 기본값이므로, 에너지 차이를 해석할 때는 장치가 지원하는 동일한 고정
graphics/memory pair를 지정한다. 일반 SFU sweep은 900MHz 이상/기본90MHz 간격,
advertised factory default와 지원되는 exact1110MHz를 포함한다.

`evaluation.html`의 기본 SFU objective는 register-control 차분이다. Q별 단가·
처리량·CI·차분 해석과 raw evidence를 확인하고 PNG/SVG로 내보낼 수 있다.
이 개발 환경에는 GPU가 없으므로 실제 pJ 값과 센서 분해능, 포화 여부는 위 실험으로
확인해야 한다. Compiler/SASS·CPU 테스트는 실제 전력 측정을 대신하지 않는다.
