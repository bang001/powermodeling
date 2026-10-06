# CUDA workload contract

Build with CUDA 12.x or 13.0 and CMake 3.22+:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build/powerbench --describe --device 0
```

With CUDA 12, the default fat binary targets Volta `sm_70`, Ampere `sm_80`, and
Hopper `sm_90`. With CUDA 13, the default targets `sm_80;sm_90`. CUDA 13 removed
offline Volta compilation; use CUDA 12 for V100. Explicit
`CMAKE_CUDA_ARCHITECTURES` and `CUDAARCHS` settings override these defaults.

For A100 with CUDA 13.0, use a separate build directory and a CUDA 13-capable
Linux driver (R580 or newer):

```sh
cmake -S . -B build-a100-cuda13 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-a100-cuda13 -j
build-a100-cuda13/powerbench --describe --device 0
```

Point `powermodeling --bench` and the GPU test's `POWERBENCH` environment variable
at that executable. CUDA 13 profiling on A100 needs a compatible Nsight Compute,
such as 2025.3 or newer. `ptxas` prints register and local/spill resource usage
during compilation. Device clock metadata uses `cudaDeviceGetAttribute`, since
CUDA 13 removed the clock fields from `cudaDeviceProp`.

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
| `exp`, `tanh`, `silu` | Complete FP32 elementwise standard CUDA math, input load and output store | Counted output elements; memory and supporting arithmetic included, not isolated SFU energy. |
| `rmsnorm` | FP32 row sum-of-squares, epsilon 1e-5, column gamma and normalized output | Two input passes, gamma and output; pJ/element and pJ/row; no mean subtraction. |
| `softmax` | Stable FP32 row max, exp-sum reduction and normalized output | Three input passes and two exp evaluations per element; complete function energy. |

Nonlinear workloads default to one complete input application per launch and
16 launches per batch. `--iterations` repeats the entire application with fresh
volatile loads/stores. Input/output slices are disjoint per block; the footprint
must divide into complete equal slices/rows. `--row-width` defaults to 1024 for
RMSNorm/Softmax; these workloads support full-grid execution only. Before and
after measurement, distributed outputs are checked against CPU double references.
See [the nonlinear experiment protocol](../docs/nonlinear-experiments.ko.md).

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

Write/copy rounds the effective number of words down to a whole multiple of
`blocks * threads * stride`. This address ownership tile ensures that separate
threads never store to the same word even after wrapping. A footprint smaller
than one tile is rejected. Both `requested_working_set_bytes` and the effective
`working_set_bytes` are reported. This can change the effective footprint between
geometry trials; compare the reported geometry and counters, rather than assuming
the requested allocation was used unchanged.

For a strided region with `n` words, potential reachable words are
`n / gcd(n, stride)`. Allocation capacity alone therefore does not establish an
HBM workload. Result fields `logical_reachable_bytes` and
`potential_cache_sector_bytes` describe a complete address period; finite
iterations and SM admission can touch less. The sector value is a conservative
bound unless `potential_sector_footprint_exact` is true. Neither is observed
cache/DRAM traffic. Every non-latency memory launch restarts the same address
sequence, so a large allocation can still touch only a small cache-resident
subset. `finite_launch_reachable_bytes_upper_bound` and
`finite_launch_sector_bytes_upper_bound` report finite-work bounds; their
respective `*_exact` fields identify exact address-footprint calculations.
For unfiltered read/write/copy, the unique per-region word count is
`min(n / gcd(n, stride), lanes * 4 * iterations)`; L1 uses per-block `n` and
`lanes`, then sums disjoint slices. Bounds for SM-filtered runs do not prove the
actual subset was visited. Dependent latency probes use a varying deterministic
start on a randomized full cycle and report a bound over measured probes.

`--sm-ids 0,1,...` admits blocks only when `%smid` is in the requested set. This
is best effort dispatch admission: it does not disable SMs or map GPCs. Actual
block counts and SM IDs are recorded. Missing requested SMs reject the trial.
`allocated_sm_working_set` reports L1 slice bytes multiplied by the average
number of blocks dispatched to that SM per launch. It describes assignment, not
concurrent residency or cache occupancy.

Each run allocates and initializes data before any measurement phase. New plans
use a paired active reference for custom Tensor/L1/L2/HBM kernels; standalone
`powerbench` requires `--paired-reference` to enable it. The initial target
`warmup` precedes `idle_pre`, which keeps the CUDA context and all buffers
allocated. Every arm then has its own warmup, outside its measurement window:

| Order | Sequence after `idle_pre` and before `idle_post` |
|---|---|
| `AB` | `warmup_reference` → `active_reference` → `warmup_treatment` → `measure` |
| `BA` | `warmup_treatment` → `measure` → `warmup_reference` → `active_reference` |

`A` is an integer issue-loop control and `B` is the requested treatment. Each arm
runs for the configured sustained window (at least 10 s); each arm's warmup is
at least 1 s, and each idle bracket is at least 6 s. A plan assigns AB/BA order by
repeat parity with a seeded random flip per condition. Four repeats give exact
order balance; three leave one extra order. Changing trial order and balanced
crossover order address different sources of drift. GEMM can opt in, but its
cuBLAS launch geometry is not matched and its active-reference contrast is
ineligible for attribution. Latency/control diagnostic workloads are unpaired by
default.

The reference shares the custom treatment's blocks, threads, iterations, SM
admission mask and batch-launch count. Dedicated reference sinks and admission
counters keep a BA reference from overwriting the treatment's output or counts.
The context, allocations and requested clock policy remain constant across both
arms. This is a practical **operational contrast**, not an exact instruction
counterfactual: register pressure, occupancy, cache/DRAM residency, instruction
mix and duration per launch can differ. Even a clock/temperature qualified
contrast does not isolate transistor switching or one physical component's
energy. Negative contrasts are retained as diagnostics, never clipped to zero
or promoted to an energy optimum. Retain treatment total energy, powered-idle
state energy and paired-reference energy separately.

`type=treatment_protocol` records the actual order, phase order and scope.
`type=active_reference_result` records reference timing, exact per-epoch complete
batch/admission counts and requested-SM coverage; `operations` and
`logical_bytes` are zero because the integer reference is neither target FLOPs
nor target memory payload. The Python runner stores these as top-level
`treatment_protocol` and `active_reference` and rejects missing/duplicate arms,
wrong order, changed geometry or overlapping warmup/measurement phases. The
usual `type=result` still contains treatment `measure_epochs`.

Without the paired flag (including legacy plans), the sequence is `warmup` →
`idle_pre` → `measure` → `idle_post`. Legacy measurements remain visible with
no paired-reference claim.

Phase records carry `host_monotonic_ns` from Linux `CLOCK_MONOTONIC`, so a host
power sampler can clip readings to the boundaries. Idle is a **powered state**
that includes clocks, leakage, HBM refresh and background activity. An idle
interval can change P-states and temperature; the analyzer trims transitions,
interpolates pre/post idle drift at each measured arm, and qualifies actual
clocks and temperatures. Dirty writeback/background effects may affect the
post-idle baseline. An active-minus-idle value is an operational state increment,
not automatically physical dynamic energy. Paired-reference telemetry is
qualified independently, with its complete epoch windows and its actual state.

`duration_s` is CUDA-event elapsed **experiment window**, including launch gaps;
it is not the sum of profiler kernel busy times. `host_duration_s` is also
reported. `measure_epochs` records approximately one second of complete batches
with exact admitted block counts, operations and logical bytes. Epochs carry
`host_monotonic_start_ns` / `host_monotonic_end_ns` plus `start_s` / `end_s` in the
same Linux monotonic timebase as power samples. They include counter readback
gaps; `counter_readback_ns` records that overhead. Microbenchmarks read back
32 KiB of admission counters at most once per roughly one second and once for
the final partial epoch. No per-epoch stdout occurs during measurement. The
analyzer can sum complete epochs wholly inside the trimmed steady-state window
and integrate power over those exact bounds. It must not interpolate work in an
unknown partial batch or silently substitute full-run throughput. FLOPs use dense
FMA = 2. Memory bytes are requested logical payload;
copy includes one read and one write. Sector inflation, cache fills/writebacks,
telemetry, initialization and sink/output transfers are not included in logical
byte counts. `auxiliary_work` quantifies issued result-output payload, Tensor
operand loads, telemetry atomic updates and host counter readbacks. These are
logical work counts, not physical traffic, and exclude possible compiler spills.
The GEMM output payload is an algorithmic minimum; internal kernels can add work.
No component rail attribution is performed by this executable.

For deterministic Nsight application replay, use fixed counts instead of a
wall-clock loop:

```sh
build/powerbench --workload l2 --warmup-seconds 0 --idle-seconds 0 \
  --warmup-batches 8 --fixed-batches 1 --batch-launches 1 --iterations 1024 \
  --profile-region
```

`--profile-region` brackets `measure` with `cudaProfilerStart` / `cudaProfilerStop`.
Start Nsight Compute with `--profile-from-start off` so initialization and warmup
kernels are excluded without relying on launch counts. For example:

```sh
ncu --profile-from-start off --replay-mode application --cache-control none \
  --set full --csv build/powerbench --workload l2 --warmup-seconds 0 \
  --idle-seconds 0 --warmup-batches 8 --fixed-batches 1 --batch-launches 1 \
  --iterations 1024 --profile-region
```

With a paired plan, the profiler command retains `--paired-reference` only to
allocate the same reference buffers. `--profile-region` suppresses every reference
arm and its warmup; only treatment `measure` is profiled.
`paired_reference_context_allocated` in the result must match the energy run.

Application replay and cache-control preservation allow the application warmup
before each pass. Eight batches are a starting point; measured hit rate, finite
footprint and stable clocks must still establish warmup adequacy. Filter the
intended microbenchmark kernel when appropriate. For GEMM retain all kernels
inside the measure region, since cuBLAS may launch multiple internal kernels per
invocation. `kernel_launches` is null for `gemm`; `gemm_invocations` is reported
separately. The first measured latency-probe nonce is reset after timed warmup,
so profiler replay begins from the same deterministic sequence.

Profiling results are counter diagnostics and must not be used as
uninstrumented energy measurements. Device metadata includes process ID and
CUDA compile/runtime/driver/cuBLAS versions to bind evidence to the actual
process and software stack. Real GPU execution is required to validate cache
residency, effective traffic, register spills, Tensor utilization, stationarity
and board or HBM energy.
