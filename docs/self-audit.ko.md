# 전체 구현 자가점검

검토 기준일: 2026-10-04 (UTC). 대상은 FP16 Tensor·L1·L2·HBM workload, 전력 수집, 클럭 제어, NCU 판정, 결과 분석, 모델 fitting과 CLI이다.

**NCU counter로 목표 경로의 적절성을 평가하고 최적값 선정에 반영하도록 수정했다.** 수동 verified 표시, 전체 실행의 처리량과 잘라낸 전력 구간의 혼합, 기본 DVFS 결과의 고정 클럭 비교 등 결과를 과신하게 만들 수 있는 경로를 함께 점검했다. 실제 V100/A100/H100 장비는 이 검토 환경에 없어 hardware 측정·NCU 실행의 적절성까지 통과했다고 주장하지 않는다.

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
| 높음 | process inventory 누락·sampler 실패·malformed worker event로 간섭 또는 raw 실패 근거가 감춰질 수 있었음 | compute/graphics/MPS inventory·active/pending MIG 확인; parse/phase/UUID/sampler/interrupt 실패의 stdout·events·samples를 raw에 보존 | telemetry 13개·clock 9개·runner audit 10개 테스트; blocked native NVML call는 강제 중단하지 않고 bounded stop/lock 실패로 표시 |
| 높음 | model 입력의 단위·실측 근거가 없고 좁은 holdout 하나를 넓은 mixed 예측으로 일반화할 수 있었음 | 단위·feature/power provenance·GPU/고정 클럭 검증과 독립 mixed holdout 범위 제한 | 누락 provenance·단위 혼합·중복 holdout·calibration 범위 밖 예측 테스트; 실제 mixed calibration은 별도 필요 |
| 중간 | seed·실제 DVFS 상태가 다른 결과를 repeat로 합치거나 기본 DVFS를 공정한 고정 클럭 최소값으로 해석할 수 있었음 | seed·binary·clock policy 보존, uncontrolled clock의 탐색용 결과 분리 | grouping·uncontrolled selection 테스트 |
| 중간 | NVML field 응답의 ID/scope와 sensor timestamp 시계를 충분히 구분하지 못함 | `fieldId`/`scopeId` 검증, `timestamp_clock=unix_epoch_microseconds`와 host monotonic/realtime query bracket 분리 | field identity 불일치·timestamp schema 테스트 |

## 전체 구현 범위별 판정

| 구성 | 코드에서 점검한 기준 | 현재 해석과 한계 |
|---|---|---|
| FP16 Tensor WMMA | FP16 입력·FP32 누산, 독립 accumulator, `FMA=2 FLOP`, output sanity | 공통 WMMA 경로의 에너지; H100의 모든 최신 최고 성능 경로를 대표하지 않음 |
| cuBLAS GEMM | datatype·누산, `2MNK`, library version, deterministic profile region | Tensor와 메모리 경로가 함께 동작; 순수 Tensor rail 에너지로 이름 붙이지 않음 |
| L1 | `.ca`, block별 working set, global-load hit·하위 traffic | shared/L1 carveout와 residency에 따라 실제 용량 변화; NCU 확인 필요 |
| L2 | `.cg`, L2 hit와 DRAM 이동, sector→byte 변환 | SM·interconnect·L2 비용 포함; hit만으로 near/far 판정하지 않음 |
| HBM | `.cg`, finite footprint, read/write/copy 실제 크기, DRAM byte | controller·L2·SM 공급 비용 포함; 압축과 coalescing은 counter로 확인 |
| GPC·SM·warp·block | warp=32, SM admission count, `%smid`, block/thread sweep | SM filter는 배치된 block의 admission 제어이며 물리 SM enable/disable·GPC 고정 API가 아님 |
| L2 locality | 의존 pointer chase의 SM/offset latency·독립 fabric 지도 | 후보 지도만 생성; bandwidth saturation과 near/far energy 검증은 별도 |
| 클럭·DVFS | 지원 pair·actual clock·throttling·변경 실패·복원·resume policy | MHz가 같아도 전압·공정·power cap·온도가 같다는 보장은 없음 |
| NVML 센서 | UUID 대응, capability probe, null/error, power/energy/scope timestamp | A100 GA100 현재 전력과 H100 평균 전력 차이 기록; rail 범위는 실측 지원 확인 |
| 전후 idle 기준 | context 유지, 요청/실제 클럭·온도 일치, 전후 drift | `incremental`은 기준 대비 증가분; 순수 static/dynamic 분리의 증명은 아님 |
| 시간·적분·처리량 | monotonic phase, 완료 epoch, exact count, matched energy window | CUDA event elapsed는 launch gap 포함; power sensor의 지연·평균창은 남음 |
| NCU 적절성 | 필수 metric·단위·valid range·target launch·clocks·identity·local-memory traffic | 판정 threshold는 프로젝트 정책; missing evidence, 짧은 active clock sample 누락과 L2 write/copy residency는 미확정 |
| 최적값 분석 | 반복 중앙값·전체 유효 고정 클럭 sweep 최고 처리량의 95%·전체/증가분 단가·Pareto | NCU·정확한 시간 정렬이 통과해도 전체 최고값의 95% 미달이면 verified winner 없음; coverage와 이유 표시 |
| 모델 | explicit measured feature·단위/provenance·rank/condition·독립 holdout | empirical workload 모델; 혼합 calibration 없는 계수 단순 합산 금지 |
| CLI·provenance | validate-run/profile/evaluate/attach/analyze, 원본 보존, UUID·binary·condition | batch validation checkpoint·coverage 기록; 실패 evidence도 남김; binary hash·toolchain·PID로 결과 추적 |
| 빌드·CI·문서 | Python 회귀 테스트와 CUDA 12의 `sm_70;sm_80;sm_90` 컴파일 | 컴파일/CPU 테스트가 실제 센서·cache 적절성 검증을 대신하지 않음 |

## 재현 가능한 확인

```bash
python -m unittest discover -s tests -v
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="70;80;90"
cmake --build build -j
python -m powermodeling --help
```

Python 회귀 테스트 **142개가 로컬에서 통과했다**. 테스트는 생성된 데이터와 NVML/subprocess mock으로 실패·반례를 검증한다. NCU timeout·중단 시 소유한 process group만 종료하고 부분 출력·실패 evidence·clock 복원 결과를 보존하는 경로도 포함한다. 독립 주소 열거에서 write/copy 소유권 9,543 조건과 finite footprint 30,720 조건을 확인했으며, 이는 주소 수학의 검증이다. CUDA 12 다중 아키텍처 빌드는 컴파일 가능성을 검증한다. 최종 실행 결과와 CI 상태는 [PR #1](https://github.com/bang001/powermodeling/pull/1)에 기록한다. 테스트 이름·조건은 `tests/`에서 확인할 수 있다.

## 실제 GPU에서 남은 확인

| 확인할 것 | 통과 판단의 근거 | 통과 전 내릴 수 없는 결론 |
|---|---|---|
| 세 GPU의 전력 API·rail | discovery의 실제 지원 scope·오류·sensor window, 독립 계측 교차 검증 | H100에서 항상 HBM rail을 읽을 수 있다는 결론 |
| profiler 버전 | V100은 GV100 지원 NCU(예: 2025.2.x), 실제 metric 목록·driver 요구사항 | 최신 NCU가 세 GPU를 모두 profile할 수 있다는 결론 |
| CUDA 수치·메모리 정확성 | Tensor/GEMM의 기준 결과 비교와 memory sanitizer; finite sample/checksum sanity와 구분 | 계산값이 올바르거나 data race가 없다는 실장비 검증 주장 |
| 각 목표 workload의 NCU | 동일 조건의 필수 counter와 실제 클럭·worker identity | L1/L2/HBM label에 해당하는 측정이라는 결론 |
| 고처리량 포화 | geometry·clock sweep의 반복 plateau와 physical traffic | utilization 100% 또는 이론 peak 도달 주장 |
| baseline 상태 | idle/active actual clock·온도·전후 drift | 증가분 전력을 순수 dynamic으로 분리했다는 결론 |
| near/far | 반복 SM/주소 latency와 fabric 지도, 같은 조건의 고처리량 재실행 | SM 번호나 주소 절반이 물리 partition이라는 결론 |
| concurrent mixed workload | 독립 활동 변화의 counter feature, held-out mixed 오차·지원 범위 | 네 microbenchmark 계수의 합으로 일반 앱 전력 예측 |
| idle 비율 | 실측 idle W·부하 W·actual clock와 SKU 사양 구분 | 400 W/312 TFLOPS 사양만으로 idle W 역산 |

NCU 적절성 통과 → 높은 지속 처리량 확인 → 최소 에너지 비교 → 모델 calibration/holdout 검증 순서로 결론을 좁힌다. 이 단계들은 서로 대체하지 않는다. 실행 명령은 [README](../README.md), 설계 가정은 [experiment-design.ko.md](experiment-design.ko.md), 공식 출처는 [sources.md](sources.md)에 있다.
