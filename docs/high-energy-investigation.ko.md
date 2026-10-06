# Tensor·L1·L2·HBM 에너지 차이의 원인 점검

검토 기준: 2026-10-06 (Asia/Seoul). 대상은 이미 실행한 Tensor·L1·L2·HBM 실험이다.
비선형 함수/SFU 실험의 결과를 가정하지 않는다. H100에서 기존 15 pJ와 새 23 pJ라는
차이를 전달받았지만, 컴포넌트·분모·에너지 기준·raw 자료는 아직 확정되지 않았다.
따라서 아래 코드 점검과 합성 반례는 이 실측 차이의 원인을 확정한 결과가 아니다.

## 먼저 기존 raw를 재분석한다

```bash
python -m powermodeling analyze --input results/validated \
  --plan original-plan.json --output results/energy-audit --plots
```

NCU evidence가 이미 붙은 결과는 재측정 없이 분석할 수 있다. NCU가 없는 원본도
전력·작업량·기준 차이를 진단할 수 있지만 경로 검증을 통과했다고 처리하지 않는다.
출력은 원본과 다른 디렉터리를 사용한다. `--plots`에는 plots 의존성이 필요하다.

`evaluation.html`의 **Energy accounting and comparison checks**에서 같은 조건의
전체·idle 증가분·paired 대비 단가, 전력, 처리량, 기준 승인 여부를 나란히 확인한다.
`summary.json`의 trial/group 및 `evaluation.json`·`evaluation.csv`에는
`measurement_diagnostics`가 저장된다.

| 진단 항목 | 확인할 것 | 해석 |
|---|---|---|
| `denominator` | FLOP/logical bit, 같은 적분 구간의 count·rate·시간, exact 여부 | 과거 whole-run rate 추정은 정확한 count로 승격하지 않음 |
| `objectives` | 각 기준의 J/count 또는 W/rate 재구성과 보고 값의 차이 | 수치 계산 오류와 에너지 범위 차이를 분리 |
| `unit_conventions` | byte당 8 bit, dense FMA당 2 FLOP, legacy alias | pJ/byte·pJ/bit, pJ/FMA·pJ/FLOP를 같은 숫자로 비교하지 않음 |
| `power_contributions`, `scope_factors` | 전체 W, idle/reference W와 분율, total/contrast 비율 | 기준 차이가 어느 정도의 단가 차이를 만들 수 있는지 확인; 보정 계수가 아님 |
| `telemetry_crosscheck` | 누적 에너지와 전력 적분의 일치, 사용한 에너지 source | NVML mW→W, mJ→J 변환과 센서 차이를 확인 |
| `timing`, `throughput_crosscheck` | 완료 epoch·readback·batch 시간, 별도 NCU replay의 처리량 | CUDA event span도 host gap을 포함; replay busy rate로 에너지 분모를 대체하지 않음 |
| `execution.kernel_resources` | registers·shared/local memory·occupancy 상한 | 자원 제약의 근거이며 실제 utilization/포화의 증명이 아님; 기존 raw에서는 없을 수 있음 |
| `energy_selection_diagnostics` (evaluation component) | 같은 clock 최고 성능 기준 후보와 전체 sweep 최고 성능 기준 후보의 단가·처리량 비율 | 더 빠른 처리량 조건 때문에 에너지가 더 높은 후보를 선택했는지 구분 |

Group 값은 유효 반복별 진단값의 median이다. 에너지/count의 median과 별도로
집계한 energy median/count median은 일반적으로 다르므로 재구성은 반복별로 한다.
Baseline 미승인 차감값은 진단으로 표시하고 승인된 효율 후보와 분리한다.

## 15→23 pJ 차이를 분해하는 방법

같은 분모·에너지 기준일 때 `새 단가 / 기존 단가 = (새 전력 / 기존 전력) ×
(기존 처리량 / 새 처리량)`이다. 23/15는 약 1.533이다.

- 전력이 같다면 새 처리량이 기존의 약 65.2%일 때 이 차이가 생긴다.
- 처리량이 같다면 새 에너지 범위의 전력이 기존보다 약 53.3% 높을 때 생긴다.
- 기존이 idle 차감이고 새 값이 전체 GPU 단가라면, 전체 전력의 약 34.8%가
  기준 idle인 경우에도 같은 차이가 생긴다. 이는 가능한 회계 반례이며 실제 H100
  idle 전력을 추정한 것이 아니다. 기준의 actual clock·온도·cap·pstate 일치를 확인해야 한다.

이 비교에는 기존/새 workload와 단위, 전체/idle/paired 범위, read/copy 방향,
graphics·memory 실제 클럭, ECC/MIG/power cap, 입력 크기·stride, 처리량, binary와
CUDA/NCU 버전이 필요하다. Literature의 isolated component pJ와 전체 GPU J/logical work는
분자부터 다르므로 일치하도록 숫자를 낮추지 않는다.

## 코드 점검에서 확인한 것

아래 커널 점검은 `8892f8c`까지의 four-stream read 구현을 대상으로 했다. 현재는
[단일 stream read v2](memory-read-v2.ko.md)로 변경했으며, read는 iteration당
한 접근과 단순 uint32 합계를 사용한다. 기본 iterations를 4배로 늘려 기본 launch
요청량을 유지했고, 명시적인 iteration 값은 변환하지 않는다. 이전 raw의 계산은 보존한다.

당시 Tensor 분모는 WMMA당 `2 × 16 × 16 × 16 = 8192 FLOP`, memory는
thread당 iteration마다 4개 32-bit 접근이며 copy는 read+write를 함께 센다.
NVML의 전력 mW→W, 누적 에너지 mJ→J, logical byte→bit 계산에서도 공통 배율 오류는
발견하지 못했다. 이는 실장비의 계측 정확도까지 확인했다는 뜻은 아니다.

대신 다음 설계 요인은 실제 처리량을 낮추거나 비교 범위를 다르게 만들 수 있다.

1. Memory kernel은 scalar load·주소 wrap·checksum을 함께 수행한다. PTX에서
   16 B/thread/iteration당 4개 주소 wrap 경로를 확인했다. 특히 L1에서는 메모리
   bandwidth보다 instruction issue/주소 계산이 제한할 수 있다.
2. Tensor accumulator 1개는 의존성을 숨기기 어렵고, accumulator가 많으면 register
   사용량이 늘어난다. 어느 geometry가 높은 처리량인지 실제 sweep와 counter로 판단한다.
3. L1 입력 크기는 CTA당 크기다. 여러 CTA가 같은 SM에서 실행되면 active working set이
   L1 용량을 넘길 수 있다. grid의 blocks/SM을 resident CTA 수로 단정하지 않는다.
4. 지속 에너지 구간에는 launch·host synchronization·counter readback 비용이 포함된다.
   에너지와 work가 같은 시간 범위를 쓰면 계산 오류는 아니지만 busy-kernel 단가와 다르다.
5. 경로 검증 `pass`는 포화가 아니다. 기존 sector inflation 허용 범위는 stride 실험을
   위해 최대 8.25배이며, 실제 8배 traffic에서도 `pass`가 가능한 합성 반례를 확인했다.
6. 전체 sweep 최고 처리량의 95%를 유지하는 추천은 느린 clock의 더 낮은 pJ 후보를
   제외할 수 있다. `energy_selection_diagnostics`에서 같은 clock의 성능 기준을 통과한
   최소 에너지와 전체 최고 성능 기준을 적용한 후보를 나란히 확인한다. 기존 실험이
   성능 제약 없이 최소 단가를 찾았다면 동일한 선택 규칙으로 비교해야 한다.

NCU 진단에 L1/L2 sector bytes와 DRAM read/write bytes의 logical payload 대비 비율을
추가했다. Replay의 physical traffic은 해당 replay의 logical payload와만 비교하며
별도 energy run의 physical pJ 분모로 쓰지 않는다. Counter 단위나 launch binding이
불명확하면 비율을 미확정으로 남긴다.

추가로 확인한 검증 결함을 수정했다. custom memory profile의 보고 launch count와
실제 NCU launch ID 수가 다르거나 같은 ID가 다른 kernel 이름에 묶이면 실패한다.
필수 ID/count가 없으면 미확정이다. 이 결함이 전달받은 H100 차이의 원인인지는 raw를
확인해야 한다. 기존 raw 에너지 숫자를 바꾸는 보정은 추가하지 않았다.

## 재측정과 커널 변경의 판단 기준

먼저 에너지 기준·단위·원본 count 재구성 문제를 정리하고, 같은 GPU·클럭·입력 조건에서
아래 짧은 진단을 수행한다. Throughput·target traffic·반복 정밀도가 통과한 경우에만
기존 단가와 비교한다. 필요하면 그 결과를 근거로 별도 kernel 구현을 A/B 비교한다.
검증 없이 현재 커널을 다른 구현으로 교체하거나 기존 raw에 배율을 적용하지 않는다.

### Tensor·L1·L2·HBM 에너지 값이 높을 때의 추가 진단

`configs/component-diagnostics.json`은 에너지 값을 임의 보정하지 않고, 반복 길이·host batching·Tensor dependency·L1 footprint가 원인인지 비교하는 독립 진단 설정입니다. 비선형 함수는 포함하지 않습니다. 기존 측정 디렉터리를 덮어쓰지 않고 새 디렉터리에 저장합니다.

기본 설정은 incoming clock policy이므로 실행 중 DVFS가 변할 수 있습니다. 원래 실험과 비교할 때는 설정 사본의 `clock_pairs`를 **원래 실험과 동일한, GPU가 지원하는 graphics·memory 고정 pair**로 바꾸고 `--apply-clocks --clock-method applications`를 사용합니다. 원래 실험이 incoming policy였다면 그 policy와 측정된 실제 클럭 분포를 함께 비교해야 합니다. 서로 다른 실제 클럭·power cap·CUDA binary의 결과를 반복 길이 효과로 해석하면 안 됩니다.

```bash
python -m powermodeling plan --config configs/component-diagnostics.json \
  --stage launch_overhead --device 0 --bench "$POWERBENCH" \
  --output results/component-launch-plan.json

python -m powermodeling run --plan results/component-launch-plan.json \
  --device 0 --bench "$POWERBENCH" --output results/component-launch
```

고정 pair를 지정한 사본을 사용했다면 위 `run`에 `--apply-clocks --clock-method applications`를 추가합니다. 이후 NCU profile과 analyze 절차는 기존 run과 동일합니다. 원래 binary의 raw 결과는 원래 binary로 profile하고, 새 binary 진단 결과는 새 binary로 profile해야 합니다. 새 실행 파일 SHA에 기존 profile을 억지로 연결하지 않습니다.

| Stage | 비교 목적 | Trial 수 | 프로토콜 시간 하한 |
|---|---|---:|---:|
| `launch_overhead` | Tensor iterations 1024/8192/65536, memory iterations 256/2048/16384 × batch launches 1/16. Blocks=4×SM, threads=256 고정 | 96 | 72분 |
| `tensor_dependency` | accumulators 1/4/8 × blocks 1/2/4/8×SM. Threads=256, iterations=8192, batching=16 | 48 | 36분 |
| `l1_residency` | CTA당 2 KiB/16 KiB × blocks 1/2/4/8×SM. Threads=256, iterations=2048, batching=16 | 32 | 24분 |
| `tensor_reference` | FP16 input/FP32 output cuBLAS GEMM, M=2048/4096/8192, N=K=4096 | 12 | 5.4분 |
| 전체 | 위 네 stage | 188 | 약 2.29시간 |

표는 clock 조건 하나, 4 repeats에 대한 값입니다. V100/A100/H100 모두 고정된 시간 프로토콜을 사용하므로 같은 하한이지만 allocation·클럭 안정화·측정 구간 overrun·NCU replay 시간은 다릅니다. 실제 GPU 성능에 따른 추가 시간은 실측해야 합니다. 이 반복 길이·batching 값은 진단 시작점이며 peak 달성을 보장하지 않습니다.

HBM의 짧은 iterations 조건은 한 launch에서 전체 요청 footprint를 방문하지 못할 수 있습니다. `finite_launch_reachable_bytes_*`와 NCU의 실제 DRAM traffic을 확인하고, L2에 머문 조건은 HBM 에너지 비교에서 제외합니다. 반복 길이를 늘릴 때 주소 순환 범위도 달라질 수 있으므로 bandwidth 변화 전체를 launch overhead로 단정하지 않습니다.

해석 순서는 다음과 같습니다.

1. 같은 input·clock·geometry에서 iterations/batching을 늘렸을 때 wall throughput이 높아지고 pJ/work가 낮아지면 launch/synchronization overhead가 후보 원인입니다. NCU kernel busy rate와 energy 측정의 wall rate 차이를 함께 확인합니다.
2. Tensor accumulators를 늘렸을 때 Tensor activity/throughput이 높아지면 dependency hiding이 원인이었을 수 있습니다. Accumulator 증가는 register 사용량도 늘리므로 occupancy 상한과 local-memory spill을 같이 봅니다. cuBLAS GEMM은 operand memory 및 auxiliary work를 포함하므로 순수 Tensor transistor energy 비교 기준이 아닙니다.
3. L1 CTA footprint를 줄였을 때 L1 hit 비율이 개선되고 downstream traffic이 감소하면 cache-capacity contention이 후보입니다. 전체 grid blocks/SM이나 CUDA occupancy 상한은 실제 동시 resident CTA 수를 증명하지 않습니다.
4. 값의 개선은 같은 board-energy scope·logical-work unit·baseline·precision에서만 비교합니다. 문헌의 isolated physical component energy와 board energy/logical FLOP 또는 bit를 직접 같다고 취급하지 않습니다.

새 CUDA binary의 raw `benchmark.kernel_resources`는 registers/thread, local bytes/thread, shared bytes/CTA, occupancy 상한, grid 평균 blocks/SM, L1 slice 합의 이론적 상한을 기록합니다. 이 query는 측정 phase 밖에서 수행하며 실제 kernel 동작과 count 식은 변경하지 않습니다. cuBLAS는 internal kernel별 자원이 달라 이 object가 `null`이고, NCU의 kernel별 정보를 확인해야 합니다. 실제 occupancy·Tensor utilization·cache hit rate는 NCU 등 별도 관측으로 확인합니다.

## 128-byte line과 32-byte sector의 확인

[L1/L2 sector·정렬 검토](cache-sector-review.ko.md)에 공식 Nsight Compute 원문 근거, 현재 scalar load의 단위, 정렬·stride별 예상 요청 sector 수와 추가 진단 설정을 정리했다. 기존 locality offset들은 모두 128 B 정렬이므로 misalignment 검증을 대신하지 못한다. Counter는 sector당 32 B로 올바르게 변환하고 있어, line 크기를 이유로 기존 에너지 값을 4로 나누는 보정은 하지 않는다.
