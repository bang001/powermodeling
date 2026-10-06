# 단순 메모리 read 커널

L1·L2·HBM의 `access=read`는 `scalar_single_stream_read_v2`를 사용한다.
기존 네 stream의 load·XOR 누산을 단순한 grid-stride read로 변경했다.
각 thread는 iteration마다 32-bit 원소 하나를 읽고 uint32 합계에 더한 뒤 다음 주소로 이동한다.
L1은 `.ca`, L2/HBM은 `.cg`를 유지한다. Stride·offset·L1 CTA별 slice·SM filter도
그대로 적용하며, 주소는 읽은 데이터에 의존하지 않는다.

## 무엇이 달라지는가

| 항목 | 이전 read | 새 read |
|---|---|---|
| Thread당 iteration의 요청 | scalar 4 B × 4 streams | scalar 4 B × 1 |
| 반복 내부 결과 사용 | 네 XOR 누산 체인 | 단일 uint32 덧셈 누산 |
| 반복 내부 주소 갱신 | 여러 stream 주소 및 base wrap | 단일 위치 갱신 |
| 관측용 출력 | XOR 누산 결과 | thread의 전체 read 값의 합 (modulo 2³²) |
| 기본 iterations | 1024 | 4096 |
| 기본 logical read 양/thread/launch | 16 KiB | 16 KiB |

초안의 `asm volatile`만으로는 load가 보존되지 않았다. 마지막 값만 sink에 기록하면
PTX assembler가 중간 load들을 제거하는 것을 실제 SASS에서 확인했다. PTX의
`ld.volatile`은 기존 `.ca`/`.cg`와 결합할 수 없고 cache 동작도 달라질 수 있어
사용하지 않는다. 각 읽은 값을 단순 uint32 합계에 더해 최종 sink에 기록하고,
실제 GPU 명령에 모든 load가 남는지 검사한다. 이 **덧셈 하나의 비용은 남는다**.
출력 표본은 CPU의 독립 주소·입력 계산과 합계를 대조하지만 전체 메모리의 정확성이나
cache residency의 증명은 아니다.
반복 길이·launch batching·동시 warp 공급과 주소 계산 비용은 여전히 존재한다.
명령을 줄인 변경이며 실제 bandwidth나 pJ 개선은 GPU에서 확인해야 한다.

Write/copy는 기존 구현과 iteration당 네 접근을 유지한다. `l2_latency`의
dependent pointer chase와 Tensor·비선형 함수 커널도 이 read 변경의 대상이 아니다.

## Count와 기존 결과의 비교

새 read의 완료 count는 다음과 같다. `admitted_blocks`는 해당 측정 구간에
완료된 모든 launch에서 실제로 admission된 CTA의 합이다.

```text
operations = admitted_blocks × threads × iterations
logical_bytes = operations × 4
```

전체 결과와 각 energy epoch가 같은 규칙을 사용한다. 기존 read의 `×4 accesses`
규칙을 새 결과에 적용하지 않는다. 분석은 보고된 완료 byte/count를 사용하며
기존 pJ 값에 4배 또는 1/4배 보정을 적용하지 않는다.

기본 iterations는 같은 launch 요청량을 유지하도록 4배 늘렸다. **명시한
iterations는 자동 변환하지 않는다.** 이전 read에서 `iterations=N`을 썼다면
같은 launch 요청량·주소 방문 범위를 비교하는 새 계획은 `iterations=4N`으로
작성한다. 기존 진단 config의 명시적인 iteration grid도 그대로다. 동일한 총
요청량이 동일한 실행 시간·메모리 병렬성·전력을 보장하지는 않는다.

Paired control은 treatment의 iterations를 공유한다. 따라서 기본 반복 수를
변경하면 control의 launch당 작업량도 달라진다. 새 paired 차이를 이전과 동일한
baseline으로 간주하지 말고 total·idle 증가분·control power를 각각 비교한다.

Raw에는 `kernel_implementation_version`과
`memory_accesses_per_thread_iteration=1`을 기록한다. 이전 binary/hash·version과
새 결과는 별도 그룹으로 분석하고 NCU evidence도 해당 binary로 수집한다.
기존 raw는 재분석할 수 있지만 새 binary로 이전 profile을 대체하지 않는다.
같은 클럭·geometry·working set·stride·offset·전력 범위에서 **W와 logical GB/s를
각각 비교**해야 pJ 변화가 전력 변화인지 처리량 변화인지 구분할 수 있다.

## L1/L2/HBM만 짧게 확인하기

새 binary로 빌드하고 plan을 다시 생성한다. 아래 예시는 세 계층 각각 한 geometry,
4 repeats이며, 총 12 trials·프로토콜 시간 하한 9분이다. 하나의 stage만 실행하면
4 trials·3분이다. NCU replay·준비·측정 overrun은 별도이며 이 진단은 최적점 탐색이 아니다.

```bash
cmake --build build -j
export POWERBENCH=build/powerbench
python -m powermodeling plan --config configs/memory-read-smoke.json \
  --device 0 --bench "$POWERBENCH" --output results/memory-read-v2-plan.json
python -m powermodeling run --plan results/memory-read-v2-plan.json \
  --device 0 --bench "$POWERBENCH" --output results/memory-read-v2
python -m powermodeling validate-run --plan results/memory-read-v2-plan.json \
  --input results/memory-read-v2 --output results/memory-read-v2-validated \
  --device 0 --bench "$POWERBENCH"
python -m powermodeling analyze --input results/memory-read-v2-validated \
  --plan results/memory-read-v2-plan.json --output results/memory-read-v2-report --plots
```

A100/CUDA 13은 build 경로를 `build-a100-cuda13`으로 바꾸고 호환 NCU를 선택한다.
HBM만 실행하려면 `plan`에 `--stage hbm_read`를 추가한다. L1/L2는 `l1_read`/`l2_read`다.
기본 config는 incoming policy다. 통제 비교에는 config 사본의 `clock_pairs`를
기존 실험과 동일한 지원 pair로 바꾸고 run과 validate-run에
`--apply-clocks --clock-method applications`를 동일하게 추가한다.
Incoming policy에서는 controlled-clock 검증이 미확정으로 남을 수 있다.

HBM은 유한 launch의 실제 footprint와 DRAM read bytes를 확인하고, L1/L2는
hit와 하위 traffic을 확인한다. 작은 footprint가 cache에 남는 결과를 HBM 개선으로
해석하지 않는다. 여러 geometry와 clock의 본 실험은 기존 saturation/DVFS 설정으로
새 plan을 만들어 진행한다.

## Load 보존과 실제 GPU 검사

GPU 없이도 CUDA toolkit의 `cuobjdump`와 호환 `nvdisasm`을 PATH에 두고
다음 구조 검사를 실행할 수 있다.

```bash
python tools/check_memory_read_sass.py --binary "$POWERBENCH"
```

이 검사는 L1/L2 특수화의 반복문에서 global load가 사라지는 회귀를 탐지한다.
Unroll된 load 일부만 남는 오류까지 모두 증명하지는 않으므로, 커널 수정 시
반복 카운터 증가량과 load 수를 대조하고 실제 NCU 경로·traffic 검사를 유지한다.

실제 GPU에서는 작은 buffer·wrap·stride 경계의 결과 합계와 count를 검사할 수 있다.

```bash
POWERBENCH_GPU_TESTS=1 python -m unittest discover -s tests -p test_memory_read_contract.py -v
```

CUDA 빌드·명령어 검사·CPU 테스트만으로 GPU의 bandwidth나 에너지 개선을
주장하지 않는다. 실제 pJ와 cache residency는 재측정 대상이다.
