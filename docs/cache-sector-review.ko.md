# L1/L2의 128-byte cache line과 32-byte sector 검토

검토일: 2026-10-06 UTC. 대상은 V100/A100/H100에서 현재 worker가 실행하는
scalar global read이다. 아래 주소 예시는 실측 에너지 결과가 아니다.

후속 변경으로 현재 read는 [single-stream v2](memory-read-v2.ko.md)를 사용한다.
한 thread/iteration당 4 B 한 번을 읽고 uint32 합계에 더하며, 기본 iterations는
4096이다. Sector 표는 scalar load 하나 기준이므로 그대로 유효하다. 아래의
four-stream 관련 내용은 `8892f8c`까지의 이전 구현을 검토한 기록이다.

## 128 B를 모든 연산의 고정 전송량으로 해석하지 않는다

NVIDIA Nsight Compute 2025.2.1의 Metrics Reference는 **L1과 L2 모두
128 B cache line, line당 32 B sector 4개**라고 명시한다. Memory Tables는
두 계층의 최소 접근 크기를 sector 하나로 정의하고, L2 request는 한 line의
1–4 sectors에 접근한다고 설명한다. 따라서 line, warp의 load 요청, sector,
하위 계층의 전송량은 구분해야 한다. 이 프로젝트의 NCU byte 변환 단위도
**32-byte sector**이다. 이를 cache array 전체의 동작 횟수나 에너지의
최소 단위와 동일하다고 가정하지 않는다.

현재 커널에서 full warp의 32 threads가 각각 4 B를 연속해서 읽으면
한 load 명령의 logical payload는 128 B이다. 시작 주소가 **32 B 정렬**이면
4 sectors를 덮는다. 128 B 정렬은 그 요청이 하나의 128 B 주소 구간 안에
들어가는 조건이며, 4 sectors를 요구하는 데 반드시 필요한 조건은 아니다.
cache hit라면 그 요청량 전체가 L2나 HBM으로 내려가지 않는다.

| 한 warp의 scalar load 조건 | Logical bytes | 서로 다른 32 B sectors | Sector bytes / logical bytes |
|---|---:|---:|---:|
| stride 1, offset 0 B | 128 | 4 | 1.00 |
| stride 1, offset 4 B | 128 | 5 | 1.25 |
| stride 1, offset 32 B | 128 | 4 | 1.00 |
| stride 1, offset 128 B | 128 | 4 | 1.00 |
| stride 2, offset 0 B | 128 | 8 | 2.00 |
| stride 4, offset 0 B | 128 | 16 | 4.00 |
| stride 8 또는 32, offset 0 B | 128 | 32 | 8.00 |

표는 128 B 정렬된 기준 주소, full warp, thread당 4 B, 서로 겹치지 않는
thread 주소, 충분한 footprint를 가정한다. stride는 4 B 원소 단위다.
각 주소 `base + offset + lane × stride × 4`의 `floor(address / 32)`를 세면
sector 수를 구할 수 있다. Offset 32 B는 두 128 B 주소 구간에 걸쳐도 여전히
4 sectors다. Sector 수가 같아도 line/tag 조회나 하위 request 수는 달라질 수 있다.
유한 buffer의 wrap이나 broadcast로 주소가 겹치면 표를 그대로
적용하지 않는다. 이 수는 요청 주소가 덮는 sectors이며, 실제 L2/DRAM
traffic이나 에너지가 같은 배율로 증가한다는 뜻이 아니다.

## 이전 four-stream 구현에서 확인한 것과 부족했던 점

- 이전 [CUDA 커널](https://github.com/bang001/powermodeling/blob/8892f8cf38b394af5ec894df78f6839e01ff9f88/cuda/gpu_bench.cu)은 thread당 iteration마다 서로 분리된
  `ld.global.ca.u32` 4개(L1) 또는 `ld.global.cg.u32` 4개(L2/HBM)를 실행한다.
  따라서 full warp의 iteration당 logical payload는 **512 B**다.
  이상적인 stride 1 조건에서는 네 명령에 걸쳐 16 sector 요청을 기대한다.
  한 번의 vector load나 한 번의 512 B 물리 전송을 뜻하지 않는다.
- [NCU 검증](../powermodeling/validation.py)은 L1/L2 sector 수에 32 B를 곱한다.
  sector 하나에 128 B를 곱하거나 request마다 무조건 128 B를 배정하지 않는다.
  DRAM은 byte counter를 그대로 사용한다. 단위의 4배 오류는 발견하지 못했다.
- `l1_sectors_per_global_read_request`와 계층별 sector/logical-byte 비율을
  진단한다. `.cg`의 L1 bypass에서 lookup sectors가 0으로 보고되면 L1 비율을
  coalescing의 증거로 쓰지 않고 L2 read sectors와 해당 replay의 logical count를 본다.
- 기존 locality 설정의 offset 0/4096/2097152 B는 모두 128 B 정렬을 유지한다.
  이 설정만으로 **4 B misalignment와 32 B sector 경계를 검증하지 못한다**.
  기존의 sector inflation 최대 8.25 정책도 stride 실험을 허용하기 위한 경로
  판정이며, coalesced 접근이나 충분한 bandwidth를 보장하지 않는다.

## 추가 비교 설계와 실행

[cache-sector-diagnostics.json](../configs/cache-sector-diagnostics.json)은
L1/L2 read만 비교한다. Clock 조건 하나에서 `sector_alignment`는
offset 0/4/32/128 B, `sector_stride`는 stride 1/2/4/8/32를 비교한다.
Threads 256, blocks=4×SM, iterations 2048, batch launches 16, repeats 4를
고정한다. L1은 CTA당 8 KiB, L2는 조회 용량의 절반으로 시작한다.
작은 L1 slice에서 stride 32가 warp 안의 중복 주소를 만들지 않도록 했다.
stride에 따라 실제 방문 footprint는 달라지므로 hit와 finite footprint도 확인한다.
이 8 KiB slice·256 threads 설정의 이전 구현은 stride 8/32에서 네 stream이
같은 주소를 읽었다. 새 single-stream 구현도 stride 8/32이면 주소 advance가 0이므로
각 thread가 같은 원소를 반복 읽는다. Warp 내부 주소는 서로 달라도 재사용은 달라진다.
이 설정은 커널의 sector 요청과 wrap 효과를 진단하며,
관측된 pJ 변화 전부를 sector overfetch에 귀속하는 통제 실험은 아니다.

```bash
python -m powermodeling plan --config configs/cache-sector-diagnostics.json \
  --stage sector_alignment --device 0 --bench "$POWERBENCH" \
  --output results/cache-alignment-plan.json
python -m powermodeling run --plan results/cache-alignment-plan.json \
  --device 0 --bench "$POWERBENCH" --output results/cache-alignment
```

`sector_stride`는 stage와 출력 경로를 바꿔 별도로 실행한다. 기본 설정은 incoming
clock policy다. 원본과의 통제 비교에는 설정 사본의 `clock_pairs`를 원래 사용한
지원 graphics/memory pair로 바꾸고 `run`에 `--apply-clocks --clock-method applications`를
추가한다. 이후 [README의 validate-run/analyze 절차](../README.md#ncu를-통한-적절성-판단)를
같은 binary로 수행한다. 시간 하한은 alignment 32 trials/24분,
stride 40 trials/30분, 전체 72 trials/54분이다. 준비·NCU replay·overrun은 별도다.
Incoming policy 실행은 counter 관찰에 사용할 수 있지만 controlled-clock 검증은
미확정으로 남을 수 있다. 이 진단만으로 효율 최적점을 승인하지 않는다.

판정은 다음 순서로 한다.

1. 생성된 plan과 raw에서 offset, stride, effective/finite footprint, 실제 클럭을 확인한다.
2. NCU에서 L1의 sectors/request와 L1 read-sector bytes/logical-read bytes,
   L2의 TEX-origin read-sector bytes/logical-read bytes를 조건별로 비교한다.
   L2 requests를 SM의 warp load 수와 같다고 가정하지 않는다.
3. L1/L2 hit 및 하위 traffic으로 목표 계층의 공급 여부를 확인한다. Offset 4 B에서
   sector 요청이 늘어도 인접 warp의 재사용/병합 때문에 DRAM bytes는 같은 비율로
   증가하지 않을 수 있다. Offset 32 B가 느려져도 sector 수 증가로 단정하지 않는다.
4. 같은 에너지 기준의 pJ/logical bit, 전력, logical bandwidth를 나란히 비교한다.
   Counter 변화와 throughput 저하가 함께 관측되는지 본다. Paired control은 메모리
   요청이 달라서 sector overhead를 자동 제거하지 못한다.
5. 별도 NCU replay의 sector count로 energy run의 분모를 교체하지 않는다.
   논문의 pJ/access와 비교할 때는 access가 thread load, warp request, sector,
   cache-line lookup 중 무엇인지와 측정 전력의 범위를 먼저 맞춘다.

## 근거와 문헌 검토 범위

NVIDIA 웹 문서의 직접 조회는 HTTP 403으로 차단되었지만, NVIDIA 공식 배포
[패키지 `nsight-compute-2025.2.1.3-0.conda`](https://conda.anaconda.org/nvidia/linux-64/nsight-compute-2025.2.1.3-0.conda)에 포함된 Profiling Guide 원문을
확보해 Metrics Reference와 L1/TEX·L2 Memory Tables를 확인했다.
패키지 SHA256은
`acdabd89efb74923c164c0771f9627e18c42955908276ed511cd6309895edec3`이다.
`"The L1 and L2 both have 128 byte cache lines"`와
`"the minimum access size in L2 is one sector"`를 함께 읽어야 한다.
L1 표 역시 최소 접근 크기를 one sector로 명시한다. V100/A100/H100에
사용하는 counter 해석 근거이며, 이것만으로 각 장치의 pJ/access를 알 수는 없다.

연구논문은 공개 미러의 **영문 원본 PDF**에서 제목·arXiv ID·버전을 확인하고
아래 절을 읽었다. arXiv 사이트 자체에서 내려받은 것은 아니다.

| 자료 | 대조할 내용 | 이번 확인 상태 |
|---|---|---|
| [Nsight Compute Profiling Guide, Memory Tables](https://docs.nvidia.com/nsight-compute/2025.2.1/ProfilingGuide/index.html#memory-tables) | Request/sector/line의 정의와 계층별 counter의 범위 | 공식 2025.2.1 패키지 원문 확인 |
| [CUDA Best Practices Guide 12.9.1, Coalesced Access](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) | CC 6.0 이상에서 32 B 단위 coalescing, sequential/misaligned/strided 접근 사례 | 원문 추가 확인 차단 |
| [Jia et al., Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826) | v1 Table 3.1, §3.2, Listing 3.2; 인쇄 p.19/23 | [원본 PDF 미러](https://raw.githubusercontent.com/AbsurdMirror/translated-papers/main/papers/0x01-2/source.pdf) 확인 |
| [Abdelkhalik et al., Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis](https://arxiv.org/abs/2208.11174) | v1 §III.B; PDF p.4의 dependent load/latency 방법 | [원본 PDF 미러](https://raw.githubusercontent.com/AbsurdMirror/translated-papers/main/papers/0x01-4/source.pdf) 확인 |
| [Luo et al., Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499) | v1 §III.A, §IV.A; latency/throughput 커널과 실제 GPU 모델 | [원본 PDF 미러](https://raw.githubusercontent.com/AbsurdMirror/translated-papers/main/papers/0x01-6/source.pdf) 확인 |

검토 결과와 설계에 적용하는 범위는 다음과 같다.

1. **Volta 연구의 용어와 공식 counter 정의를 섞지 않는다.** Table 3.1은 V100
   L1 line/load granularity를 32 B, update granularity를 128 B로 보고하고,
   L2 line을 64 B로 보고한다. §3.2도 64 B라고 명시한다. 이 보고는 Nsight의
   L1/L2 128 B line 설명과 다르다. 두 자료의 line이 동일한 관측 대상을 뜻하는지
   이 검토만으로 해소하지 못했으며, 단순히 64 B를 128 B와 같은 의미라고
   재해석하지 않는다. NCU sector는 그 문서가 정의하는 32 B 단위로 처리한다.
   논문의 pJ/access를 가져올 때도 어떤 접근 단위인지 독립적으로 확인해야 한다.
2. **Latency와 bandwidth 실험을 구분한다.** Ampere 연구는 dependent pointer
   chasing으로 load를 직렬화하고, `.cg`와 L2보다 작은 배열로 L2 latency를
   측정한다. 이는 load를 병렬 공급하는 bandwidth/energy 커널의 대체 기준이
   아니다. 문헌의 cache operator 설명을 그대로 residency 보증으로 쓰지도 않는다.
3. **Throughput 커널의 작업량과 명령 폭을 맞춘다.** Hopper 연구는 L1에 1024-thread
   block 하나를 쓰고 L2에는 많은 blocks를 공급한다. Global-memory latency는
   4 threads × 8 B로 32 B 접근을 구성하고, throughput은 thread당 `float4`로
   5회 read와 1회 write를 수행한다. 이전 네 stream 및 현재 단일 stream의 scalar read와 명령 수·read/write
   구성·공급 병렬도가 다르다. Vectorized throughput reference는 추가 비교 후보이며,
   현재 커널을 곧바로 peak 구현이라고 판단하지 않는다.
4. **Hopper 연구를 H100 SXM 실측으로 부르지 않는다.** 논문의 실제 장비는
   A100 PCIe, RTX4090, H800 PCIe다. 방법론은 참고하지만 수치를 H100 SXM의
   bandwidth나 component pJ 기준으로 옮기지 않는다.

V100 연구의 line 크기·latency·최대 bandwidth를 A100/H100에 그대로 적용하거나,
문헌의 수치에 맞추어 23 pJ를 15 pJ로 보정하지 않는다. 실제 H100 차이는
기존/신규 raw와 동일 조건의 counter를 확보해야 판단할 수 있다.
