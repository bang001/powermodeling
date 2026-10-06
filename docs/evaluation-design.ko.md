# 컴포넌트 실험의 평가와 시각화

이 문서는 V100·A100·H100의 Tensor, GEMM, L1, L2, HBM, locality,
EXP·TANH·SiLU·RMSNorm·Softmax 실험에서 **무엇을 확인한 뒤 어떤 결론을
허용하는지** 정의한다. 구현은 `powermodeling/evaluation.py`,
`dashboard.py`, `reporting.py`이며 `analyze`가 평가를 함께 수행한다.
평가 대상은 측정한 GPU 전체 scope의 해당 workload 에너지다.

## 1. 클럭 범위와 필수 비교 조건

기본 graphics/SM sweep은 **900 MHz 이상**이다. 200–300 MHz의 일반 grid와
native 최저 클럭 끝점은 실행하지 않는다. `graphics_min_mhz`로 하한,
`graphics_step_mhz`로 간격을 지정한다. 기본 간격은 90 MHz이며 60·120 MHz
등 양의 정수 간격도 사용할 수 있다.

```json
{
  "study_design": "energy_sweep",
  "clock_sweep": {
    "graphics_min_mhz": 900,
    "graphics_step_mhz": 90,
    "all_memory_clocks": true,
    "required_graphics_mhz": [1110],
    "include_advertised_default": true,
    "include_default_policy": true
  }
}
```

각 memory domain에서 하한 이상의 native 지원값만 grid에 매핑하고 그 평가
범위의 양 끝을 포함한다. 예를 들어 지원값이 885/915 MHz이면 900 목표는
915로 매핑한다. 정확한 목표가 지원되지 않아 생기는 매핑 오차·간격은 plan에
기록한다. domain 전체가 하한보다 낮으면 grid를 만들지 않고 범위를
`not_applicable`로 남긴다. memory MHz는 SM 간격으로 생성하지 않는다.

다음 조건은 모든 해당 실행 geometry에서 유지한다.

- **Factory default 기준:** NVML의 advertised default applications
  graphics/memory **고정 pair**. 필수 anchor이므로 이 pair가 하한보다 낮아도
  포함한다. 기본 pair를 조회·지원 확인하지 못하면 energy sweep 실행을 막는다.
- **Exact 1110 MHz:** 각 memory domain에서 정확히 지원될 때 포함한다.
  미지원은 적용 불가 사유를 기록하며 가까운 주파수로 대신하지 않는다.
- **Incoming-policy reference:** `null/null`로 들어온 driver 정책을 유지하는
  실행. 이 정책은 이전 사용자 설정·lock을 포함할 수 있다. factory-default
  DVFS가 확인된 실행이라고 자동으로 이름 붙이지 않는다.

Advertised default 고정 pair 비교와 factory-default DVFS/boost 정책의 비교는
다른 질문이다. 후자는 이전 정책이 실제 기본 상태인지 별도 장치 근거가
필요하다. 현재 구현은 전자를 필수 anchor로, 들어온 정책을 별도 reference로
평가한다. default가 1110/grid와 겹치면 한 조건의 tag로 모두 기록한다.

Runner는 선언한 하한·간격과 coverage를 비교하고 native 목록에서 grid·anchor를
재계산한다. 각 geometry에 조건이 빠지면 실행 전에 거부한다. 예전 최저 클럭
sweep plan은 새 설정으로 다시 생성한다. 이미 측정한 raw 파일은 분석 가능하다.

## 2. 평가 단위와 비교 층

GPU UUID, benchmark binary, CUDA/runtime/cuBLAS, power scope/cap,
ECC/MIG, treatment protocol, 함수 정의를 분리한다. GPU 간 결과를 같은 반복
표본으로 합치지 않는다. A100/CUDA 13과 CUDA 12 결과도 별도 층이다.

메모리는 access·stride·주소 offset·SM filter·유효 footprint를 분리한다.
L1은 전체 할당 대신 **CTA당 slice 크기**를 사용해 blocks를 늘리는 geometry
실험을 같은 조건으로 비교할 수 있게 한다. nonlinear은 FP32/math 구현,
입력 분포, footprint(Q), 행 너비, epsilon/gamma, grid mode와 구현 버전을 분리한다. locality는
blocks/threads/iterations와 SM filter 크기도 분리하고 SM ID·offset을 지도 축으로
남긴다. 같은 pJ 단위여도 이 정의가 다르면 평균하거나 같은 곡선으로 연결하지 않는다.

| 컴포넌트 | 주 결과 | 경로·정확성 평가 | 성능과 추가 진단 | 시각화 |
|---|---|---|---|---|
| Tensor WMMA | pJ/FLOP, TFLOP/s, FMA=2 | FP16 입력·FP32 누산, Tensor activity, count/epoch, spill; 현재 output 검사는 finite sample이며 완전한 수학 reference는 아님 | blocks/threads/accumulators별 plateau, actual SM MHz 기반 dense ceiling 비율 | clock–성능·에너지, geometry, energy/throughput, dense peak 비율 표 |
| cuBLAS GEMM | 전체 dense GEMM pJ/FLOP, TFLOP/s | input/output·지원 연산 정의, 실제 Tensor kernel과 보조 kernel/spill, count; finite sample 한계 표시 | m×n×k 증가의 problem-size plateau; vendor 알고리즘 변경 가능, issue-resource plateau와 구분 | shape 정보·clock 곡선·energy/throughput·problem-size 축 |
| L1 | pJ/logical bit, logical GB/s | global L1 hit·하위 L2/DRAM 이동·carveout·slice·spill | 같은 CTA slice에서 geometry별 성능, actual clock, requested/physical traffic 차이 | memory domain별 clock·energy·resource·NCU admission |
| L2 | pJ/logical bit, logical GB/s | L2 hit·DRAM 유입·L1 bypass·finite 주소 footprint·spill | footprint/stride/access별 성능; write/copy의 현재 residency 판정 한계 | footprint/stride/access 분리 그림; 누락/미확정 cell 표시 |
| HBM | pJ/logical bit, logical GB/s | read/write 방향의 DRAM/logical ratio·L2 hit·sector inflation·finite footprint | memory별 SM clock 공급능력 plateau, replay DRAM rate는 별도 진단 | memory×SM energy matrix, bandwidth/energy, 조건·품질 표 |
| L2 locality | dependent cycles/access | admitted SM·offset·loads/cycles, L2/fabric counter는 별도 근거 필요 | concurrent blocks·loop overhead·온도/clock; energy optimum으로 선택하지 않음 | clock/geometry별 SM×offset 지도; near/far label 없음 |
| EXP/TANH/SiLU | pJ/element, Gelement/s | 전후 분산 CPU double sample, complete output element/epoch, 표준 math·SFU activity·spill | auto grid의 Q scaling과 Q별 geometry/clock 비교, 메모리 비용 포함 | 함수·Q별 독립 그림, Q-scaling 그림·표, 수치 검증 표 |
| RMSNorm/Softmax | pJ/element와 pJ/row | 위 조건 + row count/width, RMS epsilon/gamma, stable max/sum 및 Softmax 행 합 | 행 너비별 성능·에너지; `pJ/row = width × pJ/element` | 행 너비별 독립 그림과 row 단가 표 |
| Control / paired arm | active reference 전력과 signed contrast | 같은 process/context·geometry·clock·온도·cap, AB/BA 균형·완료 epoch | matching 실패 시 contrast 선택 제외 | 전력/온도 trace와 order/quality; component 단가로 선택하지 않음 |

기존 NCU threshold는 `validation.py`의 명시적 정책을 그대로 재평가한다.
양의 Tensor/SFU activity는 경로 근거이며 성능 포화 증명이 아니다. 자동 L2
write/copy residency 판정이 미확정이면 해당 에너지는 보존하지만 검증 후보에서
제외한다. nonlinear CPU 비교도 대표 표본이며 전체 입력 범위의 증명이 아니다.

## 3. 다섯 단계의 판단

1. **수집 coverage:** 원래 plan의 trial ID, repeat index, UUID, workload,
   요청 clock, parameters, binary(계획에 기록된 경우), AB/BA order와 결과를
   연결한다. 누락·중복·불일치·실패 trial과 아예 데이터가 없는 workload를 낸다.
   plan 없이 데이터만 분석하면 coverage는 `unknown`이다.
2. **측정 품질:** runner 오류·간섭·clock/temperature/throttle·sensor gap와
   완료 epoch의 work/power 정렬을 검사한다. 최소 유효 반복 3회, 기본 계획은
   4회다. 제외된 반복과 이유도 남긴다.
3. **컴포넌트 근거:** NCU identity/parameter/clock binding과 수치 counter를
   재검사한다. `pass`/`fail`/`inconclusive`/`unprofiled`를 구분한다.
   미검증·실패한 빠른 조건도 유효 측정이면 성능 peak 분모에서 빼지 않는다.
4. **Plateau:** 같은 clock pair·입력 정의에서 검증된 증가 자원 level의 상위
   3개가 모두 전체 유효 관측 peak의 95% 이상이고, 그 처리량 폭이 5% 이내인지
   확인한다. 반복 CI도 정밀해야 한다. 자원은 blocks×threads, Tensor는
   accumulator까지 포함한다. GEMM은 m×n×k의 problem-size 근거로 별도 표시한다.
   seed/offset만 다른 조건은 새 자원 level이 아니다. 다른 clock의 geometry를
   합쳐 plateau를 만들지 않는다. 결과는 관측 근거이며 hardware saturation
   증명은 계속 false다.
   **Nonlinear V2 auto grid 예외:** blocks×threads가 Q에 묶이므로 기존 자원
   plateau를 적용하지 않는다. 함수·행 너비·threads·iterations·grid mode·binary·
   환경·고정 clock을 맞춘 별도 Q 곡선에서 가장 큰 유효 Q 3개를 선택한다.
   각 Q에 경로·정확한 count·반복·CI·해당 에너지 objective 요건을 통과한 근거가
   있고, 모두 전체 유효 Q 곡선 peak의 95% 이상이며 처리량 폭이 5% 이내여야 한다.
   추천 후보의 Q도 이 구간에 있어야 `observed_input_size_plateau`가 된다.
   미검증인 큰 Q를 건너뛰어 낮은 구간을 확정하지 않는다. Q별 에너지 통계·후보는
   분리하며 이 진단 곡선에서 합산하지 않는다. Q 변화는 cache와 launch 비용도
   바꾸므로 SFU 포화 증명이 아니다. fixed/v1 nonlinear은 기존 판정을 유지한다.
5. **에너지 후보:** 같은 clock에서 최소 2개의 검증 geometry와 exact count,
   반복/CI 요건을 통과한 조건 중 처리량 기준을 만족하는 에너지 최소를 고른다.
   own-clock 최소와 **전체 관측 peak의 95% 성능을 유지하는 전체 clock 후보**를
   구분한다. 후자만 component recommendation으로 낸다.

에너지와 처리량의 bootstrap 95% CI 폭/중앙값이 기본 10%를 넘으면 정밀도가
부족한 후보로 제외한다. plateau 5%, 3 levels, CI 10%는 프로젝트 평가 정책이다.
`--throughput-fraction`과 `--evaluation-policy policy.json`으로 변경하고 사용값을
보고서에 보존한다. policy JSON 필드는 `plateau_tolerance_fraction`,
`minimum_resource_levels`, `maximum_relative_ci_width`다.

판정은 `no_qualified_candidate`, `provisional_candidate`,
`qualified_observed_candidate`다. 원 계획이 없거나 불완전하고, full energy sweep가
아니거나 plateau가 미확정이면 관측 후보는 **잠정**이다. 완료된 계획·경로·정렬·
정밀도·plateau를 통과해도 가장 좋은 **관측** 조건이며 미측정 주파수의 최적점은
아니다. `summary.json`의 이전 empirical optimum 항목은 자체 관측 선택 범위를
유지하고, 위의 더 강한 종합 판단은 `evaluation.recommendations`에서 확인한다.

## 4. 기준 대비 개선과 불확실성

전체 에너지, 승인된 idle operational increment, paired active-reference
contrast를 별도로 평가한다. 차감 결과로 전체 에너지를 대체하지 않는다.
음의 contrast는 그래프/표에 남기지만 에너지 최소 후보에서는 제외한다.

Factory default와 exact 1110은 같은 input·seed·geometry·환경의 reference와
비교한다. 1110은 후보와 같은 memory domain을 사용한다. reference가 유일하지
않거나 검증·반복·CI가 부족하면 개선을 계산하지 않고 이유를 남긴다.

`energy reduction = 1 − candidate energy / reference energy`,
`throughput ratio = candidate throughput / reference throughput`를 함께 보고한다.
각 에너지 CI 끝점으로 만든 개선 envelope도 내며, 두 구간이 분리되지 않으면
`difference_unresolved`로 남긴다. 이 envelope는 보정된 ratio 95% CI나 유의성
검정이 아니다. bootstrap는 반복 변동을 나타내며 센서의 calibration 오차와
winner 선택 편향을 포함하지 않는다.

미세한 개선을 확정하려면 선정한 후보·default·1110을 **새 반복과 새 출력 폴더**로
독립 재측정한다. 입력·clock·프로파일 조건을 유지하고 순서를 무작위화한다.
확인 실험과 원 sweep의 결과를 같은 반복으로 합치지 않는다. 곡선 보간만으로
지원하지 않는 MHz의 에너지 숫자나 연속 최적 구간을 만들지 않는다.

## 5. 보고서와 실행

```bash
python -m powermodeling analyze --input results/study \
  --plan study-plan.json --output results/study-report --plots

# NCU 검증 결과가 별도 디렉터리에 있으면 원래 energy plan을 명시한다.
python -m powermodeling analyze --input results/validated \
  --plan study-plan.json --output results/validated-report --plots
```

원본 run 디렉터리의 `plan.json`은 자동 사용한다. 검증 후 디렉터리에 원 계획이
없으면 `--plan`을 지정한다. 원 plan과 무관한 추가 trial도 coverage 불일치로 낸다.

- `evaluation.html`: 외부 서버/라이브러리 없이 동작하는 독립 HTML.
  GPU·함수·입력 층·objective·memory·clock 필터, hover/CI/anchor, SVG 다운로드,
  품질·coverage·개선·NCU·수치 검증·GPU별 비교 표를 제공한다.
- `evaluation.json`, `evaluation.csv`: 판정, 조건별 단위/반복/CI/eligibility와 근거.
- `summary.json`, `trials.csv`: 기존 raw 분석 및 반복 요약/탈락 이유.
- `--plots`: 입력·GPU·Toolkit별로 분리한 PNG와 SVG. 에너지 그림은 clock–처리량,
  clock–에너지, energy/throughput, 자원–처리량, SM×memory 최소 에너지 cell,
  NCU별 valid/rejected 반복을 보여준다. 여러 memory domain은 그림도 분리한다.
  locality는 주파수별 SM×offset 지도, control은 실제 전력/온도 trace를 낸다.

Factory default는 보라 ring, exact 1110은 검은 ring, 후보는 큰 marker로 표시한다.
counter 상태는 색·marker로 구분한다. 없는 값/미승인 에너지 cell은 공백이고,
미측정 영역을 보간하지 않는다. power/temperature trace는 대표 valid repeat의
phase당 최대 96개 원본 표본을 보여주며, **적분은 원본 전체 표본과 exact epoch**로
수행한다. trace 표시 표본 수를 에너지 계산이나 sensor 갱신률로 해석하지 않는다.

GPU 비교 표는 각 SKU/UUID, 정의·footprint·row width, Toolkit, 후보의 판정 범위를
유지한다. 같은 raw memory MHz를 같은 HBM bandwidth로 간주하거나 서로 다른
정의의 pJ를 단순 순위로 만들지 않는다. 모델 계수 평가는 기존 `fit`의 독립 mixed
holdout·rank/condition·잔차/오차·convex-hull 조건을 따르며, 단일 component 단가를
더해 혼합 workload 전력을 맞춘다고 가정하지 않는다.

## 6. 구현 검증과 남은 실측

회귀 검증은 900 MHz 하한, 60/90/120 간격, 하한 밖 default 예외, exact anchor,
누락·ID만 일치한 parameter 위조·repeat/버전 불일치, 미검증 빠른 peak,
변동이 큰 반복, 다른 clock의 geometry 혼합, 서로 다른 footprint/row width,
latency의 SM/offset 보존, HTML escaping과 실제 CLI export를 포함한다.
브라우저에서는 필터·그래프 view·SVG 다운로드·반응형 화면을 검증한다.
합성 fixture는 기능 검증이며 V100/A100/H100의 실제 pJ 결과가 아니다.

현재 Cloud에는 GPU/driver가 없다. 실제 장치에서 수치·clock·sensor·NCU·plateau,
후보의 독립 확인 실험을 실행한 뒤 실측 결론을 낼 수 있다.
