# HBM bandwidth·연산 오버헤드·cache policy 설계

2026-10-10 사용자 요구를 반영한 기준이다. 목표는 HBM에 충분한 요청을 공급하면서
메모리 이외의 연산·launch·clock 변화가 에너지 단가를 지배하지 않는 조건을 찾는 것이다.
아래 80%는 사용자 선택 정책이며, 논문에서 보장한 달성률이나 물리적 포화의 증명이 아니다.

## HBM 후보는 해당 memory clock의 이론 bandwidth 80% 이상

```text
B_theoretical [byte/s] = 2 × achieved_memory_clock_MHz × 10^6 × memory_bus_width_bits / 8
logical_fraction = sustained_logical_bytes_per_second / B_theoretical
HBM candidate: logical_fraction >= 0.80
```

대상 V100/A100/H100 HBM의 DDR clock convention을 사용한다. CUDA에서 조회한
버스 폭과 **에너지 측정 구간에서 NVML로 관측한 memory clock**을 기록한다.
CUDA의 `cudaDevAttrMemoryClockRate`는 peak 장치 속성이므로 현재 DVFS clock의
대체값으로 사용하지 않는다. 요청 clock만 있거나 버스 폭·실제 memory clock이
없으면 비율을 추정하지 않고 미확정으로 남긴다. MIG·간섭·clock drift 등 기존
실험 품질 조건도 유지한다. 모델명만 보고 버스 폭이나 peak GB/s를 채우지 않는다.

이 분모는 memory clock을 낮추면 함께 낮아지고 SM clock만 바꾸면 유지된다.
예를 들어 1,000 MHz·4,096 bit의 가상 조건은 이론 1,024 GB/s,
기준선 819.2 GB/s다. 실제 제품의 측정 결과를 뜻하지 않는다.

**HBM은 관측 최고값 대비 95% 조건을 추가로 요구하지 않는다.**
80% 이상 조건에서 전체 GPU pJ/logical bit 최소를 선택하고, idle 증가분과
paired contrast는 각각 적격 조건에서 별도 선정한다. 이로써 관측 최고값에
도달하지 않았더라도 80% 이상을 공급하면서 전력이 낮은 SM clock을 후보로 남긴다.
기존 상대 95%·plateau는 HBM에서 진단 정보다. 다른 계층·연산의 기존 상대
95% 정책, HBM의 최소 반복·여러 geometry·NCU·정확한 시간 정렬·CI 조건은 유지한다.
그룹에서는 유효 반복별 비율의 중앙값에 80%를 적용한다. 모든 유효 반복에 bus/clock
근거가 있어야 하며, 누락된 반복을 제외해 중앙값을 만들지 않는다. 아래 80% 기준 미달인 default·
1110 anchor도 나머지 측정 검증을 통과하면 대조값으로 남긴다.

```bash
python -m powermodeling analyze --input results/validated \
  --plan memory-read-plan.json --output results/report --hbm-bandwidth-fraction 0.80
```

`--throughput-fraction`은 다른 컴포넌트의 관측 peak 대비 정책이며 HBM의 이론
기준을 바꾸지 않는다. 기존 HBM raw 결과도 다시 분석할 수 있지만 필요한 장치
metadata가 없으면 새 80% 기준의 승인 후보가 되지 않는다. 원래 측정값은 보존한다.

분자는 같은 에너지 구간의 **완료된 logical payload/host 시간**이다. NCU replay의
DRAM bytes를 이 분자로 대체하지 않는다. 따라서 80%는 유효 요청률의 기준이며,
실제 HBM bus의 사용률을 동시 측정한 값은 아니다. L2 hit·DRAM/logical traffic·
coalescing의 별도 검증이 필요하다. 특히 허용된 cache hit가 있으면 logical rate와
physical DRAM rate가 달라진다. 이론치를 넘는 logical 비율은 cache 재사용·정의·
count 점검을 위한 진단으로 남기며 물리적 모순이라고 자동 탈락시키지 않는다.
예를 들어 logical rate 101%에서 20%가 L2에 hit하면 단순 모델의 DRAM rate는
약 80.8%일 수 있다. 이 예시는 측정이 아니며 실제 DRAM 기여는 counter로
확인한다. NCU의 `% of peak sustained`도 별도 기준이다.

logical 80%와 다른 replay의 DRAM/logical 75%를 곱한 60%는 같은 에너지 구간의
DRAM 전송률이 아니고, DRAM bus 사용률의 하한도 아니다. 두 숫자는 서로 다른
실행의 서로 다른 비율이다. 이번 범위에는 에너지 구간 DRAM byte/s로 80%를
판정하는 gate를 추가하지 않는다.

## Read 커널과 DVFS의 연산 오버헤드

기존 V2 read는 이미 XOR 대신 scalar u32 load당 uint32 덧셈 하나를 수행한다.
다만 주소 순환과 loop counter가 64-bit라 add·compare·wrap의 명령 수가 남았다.
V3 `scalar_single_stream_read_v3`는 region word 수와 iterations가 UINT32_MAX
이하일 때 32-bit index와 loop counter를 사용한다. Device pointer는 64-bit다.
범위를 넘으면 64-bit 경로를 사용하며 `memory_read_index_math`에 기록한다.

주소는 `advance = lanes × stride mod region_words`, `boundary = n − advance`로
정규화한다. `position >= boundary ? position − boundary : position + advance`는
덧셈 overflow 없이 기존 순서를 유지한다. Load된 데이터로 다음 주소를 정하지 않는다.
Read 수는 계속 `admitted_blocks × threads × iterations`, logical bytes는 그 4배다.

각 load를 최종 sum32에 반영해 컴파일러가 중간 load를 제거하지 못하게 한다.
이 덧셈·주소 생성·제어·최종 sink 비용은 남는다. `asm volatile`만으로는 ptxas의
중간 load 제거를 막지 못한 전례가 있어 checksum을 없애지 않는다. 입력 생성과
CPU sum oracle 검증·host checksum은 측정 구간 밖에서 수행한다. Write/copy와
active control의 XOR/산술은 별도 구현이며 read의 무연산 메모리 경로로 해석하지 않는다.

SASS에서 load 수와 loop 증가량, XOR·local spill 부재, register/occupancy와
주소 연산 수를 확인한다. NCU의 사용 가능한 instruction/issue 지표는 병목 진단에
쓰며, 그 수치에 임의의 pJ를 곱해 전체 전력에서 빼지 않는다. V2/V3와 cache/index
변형은 구현·binary·metadata를 구분해 분석한다. 새 커널의 bandwidth나 에너지
개선 여부는 GPU에서 다시 측정해야 한다.

SM clock은 요청 발행·주소 연산·L2/fabric 동작에도 영향을 준다. 같은 memory
clock에서 advertised default 고정 pair와 exact 1110 MHz를 먼저 비교하고,
80% 기준이 유지되는 영역에서 SM clock을 조절한다. Default pair의 memory MHz가
다르면 이를 core-clock만의 효과로 비교하지 않는다. power cap·actual clock·throttle·
온도를 확인하며, clock lock은 부하에 따른 실제 throttle을 없애는 보장이 아니다.

H100의 별도 memory-scope W·J·pJ/logical bit는 전체 GPU 결과와 함께 보고한다.
전체 GPU 에너지는 SM·L2·fabric·기타 비용을 포함한다. Memory scope도 HBM 셀만의
dynamic 에너지로 단정하지 않으며 전체 전력에 더하지 않는다. 범용 산술 control을
빼서 주소 계산 비용이나 순수 HBM 에너지가 정확히 분리됐다고 주장하지 않는다.

## `.ca`·`.cg`·`.cs` 선택

| Load 정책 | 목적과 cache 동작 | 이번 설계 |
|---|---|---|
| `.ca` | 모든 cache level에서 캐싱 | L1 기본. HBM에서 사용하면 L1 재사용이 결과를 바꿀 수 있어 별도 진단 |
| `.cg` | L1 우회, L2 및 하위 메모리 경로 | L2/HBM 기본. 큰 footprint와 실제 DRAM counter가 함께 필요 |
| `.cs` | 한 번 쓰는 streaming 데이터의 cache 정책, L1/L2 evict-first 힌트 | HBM 비교 진단. L2 bypass나 항상 DRAM 도달을 보장하지 않음 |

PTX cache operator는 성능 힌트이며 메모리 일관성 의미를 바꾸지 않는다. `.cs`가
더 빠르거나 낮은 에너지라고 사전에 확정하지 않는다. 동일 allocation 크기·seed·
stride·geometry·iterations·clock에서 세 정책을 비교하고 L1/L2 hit와 DRAM traffic을
함께 확인한다. 기본 에너지 실험은 `.cg`를 유지하며 `.ca`/`.cs` 결과는 진단으로만
보고한다. 새로운 기본 정책의 승격은 그 결과를 근거로 별도 결정한다.

Worker의 `--read-cache-policy auto|ca|cg|cs`에서 auto는 L1=ca, L2/HBM=cg로
해석한다. L1은 ca, L2는 cg만 허용하며 HBM read에 세 정책 비교를 제공한다.
`read_cache_policy`의 실제 값과 `memory_read_index_math`는 profiler binding에서도
대조하므로 다른 variant의 counter를 재사용하지 않는다.

[HBM cache 진단 preset](../configs/hbm-cache-policy-diagnostics.json)은 T=256,
blocks=8×SM, stride=1, iterations=4096, batch=16의 한 geometry에서 세 정책만
비교한다. 최대 memory domain의 exact 1110·최대 SM clock과 advertised default
고정 pair를 포함한다. 중복은 합쳐지고 미지원 anchor는 coverage에 남는다.
미지원인 1110을 근사값으로 바꾸거나 incoming policy를 default 고정 pair로
대신하지 않는다. 기본 pair와 memory domain이 다르면 같은 pair 안에서 cache만 비교한다.

```bash
python -m powermodeling plan --config configs/hbm-cache-policy-diagnostics.json \
  --device 0 --bench "$POWERBENCH" --output hbm-cache-plan.json
python -m powermodeling run --plan hbm-cache-plan.json --device 0 \
  --bench "$POWERBENCH" --output results/hbm-cache --apply-clocks --clock-method applications
python -m powermodeling validate-run --input results/hbm-cache --plan hbm-cache-plan.json \
  --profiles-dir profiles/hbm-cache --bench "$POWERBENCH" --ncu "$NCU" \
  --output results/hbm-cache-validated --apply-clocks --clock-method applications
python -m powermodeling analyze --input results/hbm-cache-validated --plan hbm-cache-plan.json \
  --output results/hbm-cache-report --plots
```

Preset은 policy 3개 × 반복 3회 × 지원된 clock pair 수다. 각 trial의 nominal
시간은 27초(12초 측정 + 3초 warmup + 전후 idle 각 6초)이므로 2 pair는 약 8분,
3 pair는 약 12분에 초기화·clock settle·NCU replay 시간이 더해진다. 이 진단은
paired 산술 control을 실행하지 않는다. 실제 예상 시간은 생성된 plan을 따른다.
그 뒤 `.cg`에서 iterations=4096/16384를 한 축씩 비교해 host gap·loop overhead를
점검하고, 필요한 경우만 geometry 범위를 늘린다. 이 preset의 한 geometry만으로
에너지 최적점을 승인하지 않는다. 정식 실험은 `memory-read.json`을 사용한다.

## 근거의 범위

[NVIDIA CUB v2.8.2 thread_load.cuh](https://github.com/NVIDIA/cccl/blob/v2.8.2/cub/cub/thread/thread_load.cuh)는
`LOAD_CA`를 "Cache at all levels", `LOAD_CG`를 "Cache at global level",
`LOAD_CS`를 "Cache streaming (likely to be accessed once)"로 정의하고 각각의
PTX opcode에 연결한다. [PTX cache operators](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)를
설계 참조로 사용한다. 이번 환경에서는 PTX 문서 직접 접근이 제한되어 새 원문
인용은 하지 않았으며, CUB 공식 소스와 설치된 CUDA 헤더를 확인했다.

Luo 등의 Hopper microbenchmark는 연산부가 memory test의 병목이 될 수 있음을
보고한다. 그 논문의 float4 5-read+1-write benchmark의 달성률을 이 read-only
커널의 하한으로 복사하지 않는다. [연구 사례와 측정 정의](cache-sector-review.ko.md),
[출처 기록](sources.md)을 참조한다.
