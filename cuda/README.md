# CUDA workload contract

Build with CUDA 12.x and CMake 3.23+:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build/powerbench --describe --device 0
```

The default fat binary targets Volta `sm_70`, Ampere `sm_80`, and Hopper
`sm_90`. CUDA 13 removed offline Volta compilation; use CUDA 12 for V100.
`ptxas` prints register and local/spill resource usage during compilation.

```sh
build/powerbench --device 0 --workload hbm --seconds 10 \
  --warmup-seconds 3 --idle-seconds 6 --blocks 216 --threads 256 \
  --working-set-bytes 536870912 --stride-elements 1 --access read
```

Use the Python runner for power measurements and clock policy. This executable
does not sample power or change clocks. It synchronizes GPU execution before
phase boundaries and emits flushed JSONL on stdout. Diagnostic errors go to
stderr; a failed coverage/sanity check exits nonzero even if a result record was
emitted, and such a trial must be rejected.

| Workload | Issued work | Interpretation |
| --- | --- | --- |
| `tensor` | FP16 WMMA 16×16×16, FP32 accumulators, register operands loaded once per launch | Tensor reuse microbenchmark; identical operand matrices across warps. It does not implement Hopper WGMMA. |
| `gemm` | cuBLAS dense FP16 × FP16 → FP32 GEMM | Sustained dense throughput reference; includes input/cache/memory and output work. Launch geometry and SM filters cannot control cuBLAS internals. |
| `l1` | Four independent coalesced scalar `.ca` loads per thread iteration | Total allocation is divided into disjoint per-block slices; maximum L1 carveout is requested. Cache residency requires counters. Read only. |
| `l2` | Four independent `.cg` loads/stores per thread iteration | Shared footprint across all blocks; L1 bypass. Cache residency requires counters. |
| `hbm` | Same `.cg` loop over a larger incompressible footprint | The name is a requested target, not proof that payload reached DRAM. Verify DRAM counters and effective address footprint. |
| `l2_latency` | One active lane per admitted block traverses a randomized full-cycle linked list through dependent `.cg` loads | Per-SM cycles/access diagnostic; clocks, loop overhead and concurrent blocks matter. Address/SM probes do not establish near/far partition labels. |
| `control` | Four integer issue operations per inner loop plus result/telemetry stores | Active issue-loop power reference, not transistor static power or a precisely matched memory/Tensor control. |

`--iterations` defaults to 1024 inner iterations per microkernel, and one cuBLAS
invocation for GEMM. `--batch-launches` defaults to 16 for microkernels and one
for GEMM: multiple launches are queued before host synchronization. Defaults are
starting points; use geometry/iteration sweeps and profiler counters to confirm
that launch gaps and address generation do not cap throughput.

`--working-set-bytes` is total input footprint, not per-SM footprint. L1 rounds
it down to an integer number of 32-bit words per block; the effective size is
reported. `--stride-elements` is in 32-bit words. Neighboring lanes are adjacent
at stride 1. Offset wrapping uses precomputed normalized offsets and subtraction
inside the loop, avoiding hot 64-bit division even for 20/25 MiB footprints.
`--offset-bytes` shifts the allocation base and must be 4-byte aligned.

For a strided region with `n` words, potential reachable words are
`n / gcd(n, stride)`. Allocation capacity alone therefore does not establish an
HBM workload. Result fields `logical_reachable_bytes` and
`potential_cache_sector_bytes` describe a complete address period; finite
iterations and SM admission can touch less. The sector value is a conservative
bound unless `potential_sector_footprint_exact` is true. Neither is observed
cache/DRAM traffic.

`--sm-ids 0,1,...` admits blocks only when `%smid` is in the requested set. This
is best effort dispatch admission: it does not disable SMs or map GPCs. Actual
block counts and SM IDs are recorded. Missing requested SMs reject the trial.
`allocated_sm_working_set` reports L1 slice bytes multiplied by the average
number of blocks dispatched to that SM per launch. It describes assignment, not
concurrent residency or cache occupancy.

Each run allocates and initializes data before any measurement phase:

1. `warmup` runs the selected workload.
2. `idle_pre` leaves the existing CUDA context and buffers allocated.
3. `measure` runs the workload for the requested window.
4. `idle_post` retains the same context and buffers.

Phase records carry `host_monotonic_ns` from Linux `CLOCK_MONOTONIC`, so a host
power sampler can clip readings to the boundaries. An idle interval can change
P-states and temperature; the runner must trim transitions and qualify measured
clocks. Dirty writeback/background effects may affect the post-idle baseline.

`duration_s` is CUDA-event elapsed **experiment window**, including launch gaps;
it is not the sum of profiler kernel busy times. `host_duration_s` is also
reported. FLOPs use dense FMA = 2. Memory bytes are requested logical payload;
copy includes one read and one write. Sector inflation, cache fills/writebacks,
telemetry, initialization and sink/output transfers are not included in logical
byte counts. No component rail attribution is performed by this executable.

For deterministic Nsight application replay, use fixed counts instead of a
wall-clock loop:

```sh
build/powerbench --workload l2 --warmup-seconds 0 --idle-seconds 0 \
  --warmup-batches 1 --fixed-batches 1 --batch-launches 1 --iterations 1024
```

Filter the intended workload kernel, skip its first (warmup) launch and profile
the next one. Profiling results are counter diagnostics; they must not be treated
as uninstrumented energy measurements. cuBLAS may launch several internal
kernels per invocation, so `kernel_launches` is null for `gemm` and
`gemm_invocations` is reported separately.
