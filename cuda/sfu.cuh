// Native SFU instruction experiments. The long register recurrence has no
// global/shared/local load/store instructions. Immutable kernel parameters may
// be constant-bank operands. Initialization and one final hash store are separate.
constexpr const char* kSfuInputPolicy = "feedback_xor_iteration_mantissa_0p5_1_v1";

int sfu_operation(const std::string& workload) {
  if (workload == "sfu_ex2") return 0;
  if (workload == "sfu_tanh") return 1;
  if (workload == "sfu_rsqrt") return 2;
  if (workload == "sfu_rcp") return 3;
  if (workload == "sfu_lg2") return 4;
  return 5;
}
const char* sfu_opcode(int operation) {
  constexpr const char* names[] = {"ex2.approx.ftz.f32", "tanh.approx.f32",
    "rsqrt.approx.ftz.f32", "rcp.approx.ftz.f32", "lg2.approx.ftz.f32", "sqrt.approx.ftz.f32"};
  return names[operation];
}

template<int Operation, bool Control>
__device__ __forceinline__ uint32_t sfu_apply(float operand) {
  uint32_t result;
  if constexpr (Control) {
    // ptxas can coalesce this MOV. The reference is the retained common
    // register loop without MUFU, not a promised physical MOV instruction.
    asm volatile("mov.b32 %0, %1;" : "=r"(result) : "f"(operand));
  } else if constexpr (Operation == 0) {
    asm volatile("{ .reg .f32 y; ex2.approx.ftz.f32 y, %1; mov.b32 %0, y; }" : "=r"(result) : "f"(operand));
  } else if constexpr (Operation == 1) {
#if __CUDA_ARCH__ >= 750
    asm volatile("{ .reg .f32 y; tanh.approx.f32 y, %1; mov.b32 %0, y; }" : "=r"(result) : "f"(operand));
#else
    // Keep fatbin linkage explicit without silently emulating unsupported TANH.
    // The host rejects this workload on sm70 before validation or measurement.
    asm volatile("trap;");
    result = 0;
#endif
  } else if constexpr (Operation == 2) {
    asm volatile("{ .reg .f32 y; rsqrt.approx.ftz.f32 y, %1; mov.b32 %0, y; }" : "=r"(result) : "f"(operand));
  } else if constexpr (Operation == 3) {
    asm volatile("{ .reg .f32 y; rcp.approx.ftz.f32 y, %1; mov.b32 %0, y; }" : "=r"(result) : "f"(operand));
  } else if constexpr (Operation == 4) {
    asm volatile("{ .reg .f32 y; lg2.approx.ftz.f32 y, %1; mov.b32 %0, y; }" : "=r"(result) : "f"(operand));
  } else {
    asm volatile("{ .reg .f32 y; sqrt.approx.ftz.f32 y, %1; mov.b32 %0, y; }" : "=r"(result) : "f"(operand));
  }
  return result;
}

__host__ __device__ __forceinline__ uint32_t sfu_initial_bits(uint64_t lane, uint64_t seed, int chain) {
  return (random_word(lane ^ seed ^ (uint64_t(chain + 1) * 0x9e3779b97f4a7c15ULL)) & 0x007fffffU) | 0x3f000000U;
}

template<int Operation, bool Control, int Chains>
__global__ void sfu_register_kernel(uint32_t* sink, uint64_t lanes,
    uint64_t iterations, uint64_t seed) {
  const uint64_t first = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = uint64_t(gridDim.x) * blockDim.x;
  // A fixed grid may visit multiple lanes. A partial final auto CTA performs
  // only real lanes; padded threads issue no counted primitive or sink store.
  for (uint64_t lane = first; lane < lanes; lane += step) {
    float state[Chains];
    #pragma unroll
    for (int chain = 0; chain < Chains; ++chain)
      state[chain] = __uint_as_float(sfu_initial_bits(lane, seed, chain));
    #pragma unroll 1
    for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
      #pragma unroll
      for (int chain = 0; chain < Chains; ++chain) {
        uint32_t result = sfu_apply<Operation, Control>(state[chain]);
        // Shared with the reference. Every next operand is a positive normal
        // FP32 value in [0.5,1), avoiding overflow and degenerate feedback.
        state[chain] = __uint_as_float(((result ^ uint32_t(iteration)) & 0x007fffffU) | 0x3f000000U);
      }
    }
    uint32_t hash = 2166136261U;
    #pragma unroll
    for (int chain = 0; chain < Chains; ++chain)
      hash = (hash ^ __float_as_uint(state[chain])) * 16777619U;
    sink[lane] = hash;
  }
}

// One-step checks run outside all power/throughput phases. They validate the
// actual native primitive, not a CPU simulation of approximate bit feedback.
template<int Operation>
__global__ void sfu_validation_kernel(uint32_t* sample, uint64_t lanes, uint64_t seed, int chains) {
  unsigned index = threadIdx.x;
  if (index >= 8) return;
  uint64_t lane = uint64_t(index) * (lanes - 1) / 7;
  uint32_t bits = sfu_initial_bits(lane, seed, index % chains);
  sample[2 * index] = bits;
  sample[2 * index + 1] = sfu_apply<Operation, false>(__uint_as_float(bits));
}

struct SfuCheck {
  uint64_t checked_values = 0;
  double max_absolute_error = 0, max_relative_error = 0;
  bool passed = true;
};
SfuCheck check_sfu(int operation, uint32_t* sample, const Options& o) {
  #define SFU_CHECK_CASE(OP) case OP: sfu_validation_kernel<OP><<<1, 32>>>(sample, o.sfu_lanes, o.seed, o.sfu_chains); break
  switch (operation) {
    SFU_CHECK_CASE(0); SFU_CHECK_CASE(1); SFU_CHECK_CASE(2);
    SFU_CHECK_CASE(3); SFU_CHECK_CASE(4); SFU_CHECK_CASE(5);
  }
  #undef SFU_CHECK_CASE
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  std::array<uint32_t, 16> host{};
  CUDA_CHECK(cudaMemcpy(host.data(), sample, sizeof(host), cudaMemcpyDeviceToHost));
  SfuCheck result;
  for (unsigned i = 0; i < 8; ++i) {
    float x, y; std::memcpy(&x, &host[2 * i], sizeof(float)); std::memcpy(&y, &host[2 * i + 1], sizeof(float));
    double expected = operation == 0 ? std::exp2(double(x)) : operation == 1 ? std::tanh(double(x)) :
      operation == 2 ? 1.0 / std::sqrt(double(x)) : operation == 3 ? 1.0 / double(x) :
      operation == 4 ? std::log2(double(x)) : std::sqrt(double(x));
    double error = std::abs(double(y) - expected), relative = error / std::max(std::abs(expected), 1e-30);
    result.max_absolute_error = std::max(result.max_absolute_error, error);
    result.max_relative_error = std::max(result.max_relative_error, relative);
    result.passed = result.passed && std::isfinite(x) && std::isfinite(y) && x >= 0.5f && x < 1.0f
      && error <= 2e-6 + 2e-5 * std::abs(expected);
    ++result.checked_values;
  }
  return result;
}

template<int Chains, bool Control>
KernelResources sfu_resources(int operation, int threads) {
  if constexpr (Control) return query_kernel_resources(sfu_register_kernel<0, true, Chains>, threads);
  #define SFU_RESOURCE_CASE(OP) case OP: return query_kernel_resources(sfu_register_kernel<OP, false, Chains>, threads)
  switch (operation) {
    SFU_RESOURCE_CASE(0); SFU_RESOURCE_CASE(1); SFU_RESOURCE_CASE(2);
    SFU_RESOURCE_CASE(3); SFU_RESOURCE_CASE(4); SFU_RESOURCE_CASE(5);
  }
  #undef SFU_RESOURCE_CASE
  throw std::runtime_error("unknown SFU primitive");
}
template<int Chains, bool Control>
void launch_sfu(int operation, uint32_t* sink, const Options& o) {
  if constexpr (Control) {
    sfu_register_kernel<0, true, Chains><<<o.blocks, o.threads>>>(sink, o.sfu_lanes, o.iterations, o.seed);
    return;
  }
  #define SFU_LAUNCH_CASE(OP) case OP: sfu_register_kernel<OP, false, Chains><<<o.blocks, o.threads>>>(sink, o.sfu_lanes, o.iterations, o.seed); break
  switch (operation) {
    SFU_LAUNCH_CASE(0); SFU_LAUNCH_CASE(1); SFU_LAUNCH_CASE(2);
    SFU_LAUNCH_CASE(3); SFU_LAUNCH_CASE(4); SFU_LAUNCH_CASE(5);
  }
  #undef SFU_LAUNCH_CASE
}
void print_sfu_resources(const KernelResources& r, const Options& o, const cudaDeviceProp& p) {
  std::cout << "{\"registers_per_thread\":" << r.attributes.numRegs
    << ",\"local_bytes_per_thread\":" << r.attributes.localSizeBytes
    << ",\"static_shared_bytes_per_block\":" << r.attributes.sharedSizeBytes
    << ",\"dynamic_shared_bytes_per_block\":0,\"max_active_blocks_per_sm\":" << r.max_active_blocks_per_sm
    << ",\"max_active_warps_per_sm\":" << r.max_active_blocks_per_sm * (o.threads / p.warpSize)
    << ",\"occupancy_upper_bound_fraction\":" << double(r.max_active_blocks_per_sm * o.threads) / p.maxThreadsPerMultiProcessor
    << ",\"grid_average_blocks_per_sm\":" << double(o.blocks) / p.multiProcessorCount
    << ",\"scope\":\"compiled resources and theoretical residency bound; not achieved occupancy or matched register pressure\"}";
}
void print_sfu_metadata(const Options& o, int operation) {
  std::cout << ",\"sfu_primitive\":" << quote(o.workload.substr(4))
    << ",\"sfu_lanes\":" << o.sfu_lanes << ",\"sfu_chains\":" << o.sfu_chains
    << ",\"sfu_input_policy\":" << quote(kSfuInputPolicy)
    << ",\"sfu_ptx_opcode\":" << quote(sfu_opcode(operation))
    << ",\"sfu_approximate\":true,\"sfu_flush_to_zero\":" << (operation == 1 ? "false" : "true")
    << ",\"sfu_exponent_base\":" << (operation == 0 || operation == 4 ? "2" : "null")
    << ",\"math_implementation\":\"ptx_approx_register_v1\",\"grid_mode\":" << quote(o.grid_mode)
    << ",\"block_completion_count_source\":\"synchronized_completed_launches\"";
}
void print_sfu_epochs(const RunTiming& timing, const Options& o, int operation, bool control) {
  std::cout << "[";
  for (size_t i = 0; i < timing.epochs.size(); ++i) {
    const WorkEpoch& epoch = timing.epochs[i];
    uint64_t launches = epoch.batches * o.batch_launches;
    long double slots = (long double)launches * o.sfu_lanes * o.iterations * o.sfu_chains;
    std::cout << (i ? "," : "") << "{\"host_monotonic_start_ns\":" << epoch.start_ns
      << ",\"host_monotonic_end_ns\":" << epoch.end_ns
      << ",\"start_s\":" << double(epoch.start_ns) * 1e-9 << ",\"end_s\":" << double(epoch.end_ns) * 1e-9
      << ",\"batches\":" << epoch.batches << ",\"kernel_launches\":" << launches
      << ",\"admitted_blocks\":" << launches * o.blocks
      << ",\"sfu_instructions\":" << (control ? 0 : slots)
      << ",\"reference_loop_slots\":" << (control ? slots : 0)
      << ",\"operations\":" << (control ? 0 : slots) << ",\"logical_bytes\":0,\"elements\":0"
      << ",\"counts_exact\":true,\"counter_readback_ns\":0";
    print_sfu_metadata(o, operation);
    std::cout << "}";
  }
  std::cout << "]";
}
uint64_t sfu_sink_checksum(const uint32_t* sink, uint64_t lanes) {
  const uint64_t count = std::min(lanes, uint64_t(64));
  uint64_t hash = 0;
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t lane = count == 1 ? 0 : i * (lanes - 1) / (count - 1);
    uint32_t word;
    CUDA_CHECK(cudaMemcpy(&word, sink + lane, sizeof(word), cudaMemcpyDeviceToHost));
    hash = hash * 1315423911ULL + word;
  }
  return hash;
}

void experiment_sfu(Options o, const cudaDeviceProp& p) {
  const int operation = sfu_operation(o.workload);
  if (p.major < 7 || o.threads > p.maxThreadsPerBlock) throw std::runtime_error("register SFU requires compute capability >=7.0 and supported threads");
  if (operation == 1 && p.major * 10 + p.minor < 75)
    throw std::runtime_error("native sfu_tanh requires compute capability >=7.5; V100/sm70 is unsupported and is not emulated");
  if (o.grid_mode == "auto") {
    const uint64_t blocks = o.sfu_lanes / o.threads + (o.sfu_lanes % o.threads != 0);
    if (blocks > 1000000 || blocks > uint64_t(p.maxGridSize[0]))
      throw std::runtime_error("SFU auto grid exceeds practical/hardware limit; choose an explicit fixed grid");
    if (o.blocks_explicit && uint64_t(o.blocks) != blocks)
      throw std::runtime_error("explicit --blocks must equal the SFU Q-derived auto grid; use --grid-mode fixed for a block sweep");
    o.blocks = int(blocks);
  }
  if (o.blocks <= 0 || o.blocks > 1000000 || o.blocks > p.maxGridSize[0]) throw std::runtime_error("SFU blocks exceed practical/hardware limit");
  if (!o.iterations) o.iterations = 16384;
  if (!o.batch_launches) o.batch_launches = 1;
  const bool paired_context = o.paired_reference;
  const bool paired_reference = paired_context && !o.profile_region;
  if ((long double)o.sfu_lanes * 4 * (paired_context ? 2 : 1) + 262144 > (long double)p.totalGlobalMem * 0.7)
    throw std::runtime_error("SFU sink/context allocations exceed 70% of device memory");
  Buffer<uint32_t> sink(checked_size(o.sfu_lanes, sizeof(uint32_t), "SFU sink"));
  Buffer<uint32_t> reference_sink(checked_size(paired_context ? o.sfu_lanes : 1, sizeof(uint32_t), "SFU reference sink"));
  Buffer<uint32_t> validation(16);
  CUDA_CHECK(cudaMemset(sink.ptr, 0, size_t(o.sfu_lanes) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(reference_sink.ptr, 0, size_t(paired_context ? o.sfu_lanes : 1) * sizeof(uint32_t)));
  KernelResources resources, reference_resources;
  #define SFU_SETUP_CASE(C) case C: resources = sfu_resources<C, false>(operation, o.threads); reference_resources = sfu_resources<C, true>(operation, o.threads); break
  switch (o.sfu_chains) { SFU_SETUP_CASE(1); SFU_SETUP_CASE(4); SFU_SETUP_CASE(8); }
  #undef SFU_SETUP_CASE
  if (resources.attributes.localSizeBytes || resources.attributes.sharedSizeBytes ||
      reference_resources.attributes.localSizeBytes || reference_resources.attributes.sharedSizeBytes)
    throw std::runtime_error("register SFU/control compiled with local/shared memory; reject spill-contaminated experiment");
  auto launch = [&](bool control) {
    for (int batch = 0; batch < o.batch_launches; ++batch) {
      #define SFU_BATCH_CASE(C) case C: if (control) launch_sfu<C, true>(operation, reference_sink.ptr, o); else launch_sfu<C, false>(operation, sink.ptr, o); break
      switch (o.sfu_chains) { SFU_BATCH_CASE(1); SFU_BATCH_CASE(4); SFU_BATCH_CASE(8); }
      #undef SFU_BATCH_CASE
    }
  };
  auto treatment_launch = [&] { launch(false); };
  auto reference_launch = [&] { launch(true); };
  SfuCheck numerical = check_sfu(operation, validation.ptr, o);
  if (!numerical.passed) throw std::runtime_error("native SFU one-step output differs from CPU double reference");
  std::cout << "{\"type\":\"treatment_protocol\",\"kind\":" << quote(paired_reference ? "paired_active_reference" : "powered_idle_bracket")
    << ",\"order\":" << (paired_reference ? quote(o.reference_order) : "null")
    << ",\"phase_order\":" << (paired_reference ? (o.reference_order == "AB" ? "[\"active_reference\",\"measure\"]" : "[\"measure\",\"active_reference\"]") : "[\"measure\"]")
    << ",\"same_process\":true,\"same_allocations\":true,\"same_clock_policy\":true,\"launch_geometry_matched\":" << (paired_reference ? "true" : "false")
    << ",\"paired_reference_context_allocated\":" << (paired_context ? "true" : "false")
    << ",\"reference_workload\":" << (paired_reference ? "\"control\"" : "null")
    << ",\"reference_kind\":" << (paired_reference ? "\"register_loop_without_sfu\"" : "null")
    << ",\"reference_iterations\":" << (paired_reference ? o.iterations : 0)
    << ",\"arm_seconds\":" << o.seconds << ",\"arm_warmup_seconds\":" << o.warmup_seconds
    << ",\"reference_scope\":\"same Q, chains, initialization, bit-remap, loop, hash epilogue, grid and batching; SFU primitive replaced by PTX MOV which may be coalesced; different operand trajectories, instruction latency, register allocation and occupancy may remain; board-power operational proxy, not an isolated SFU rail\"}" << std::endl;
  run_phase("warmup", o.warmup_seconds, o.warmup_batches, treatment_launch);
  idle_phase("idle_pre", o.idle_seconds);
  RunTiming timing, reference_timing;
  std::function<uint64_t()> no_counter = [] { return uint64_t(0); };
  auto treatment_arm = [&] {
    if (paired_reference) run_phase("warmup_treatment", o.warmup_seconds, o.warmup_batches, treatment_launch);
    timing = run_phase("measure", o.seconds, o.fixed_batches, treatment_launch, no_counter, o.profile_region);
  };
  auto reference_arm = [&] {
    run_phase("warmup_reference", o.warmup_seconds, o.warmup_batches, reference_launch);
    reference_timing = run_phase("active_reference", o.seconds, o.fixed_batches, reference_launch, no_counter);
  };
  if (paired_reference && o.reference_order == "AB") reference_arm();
  treatment_arm();
  if (paired_reference && o.reference_order == "BA") reference_arm();
  idle_phase("idle_post", o.idle_seconds);
  SfuCheck post = check_sfu(operation, validation.ptr, o);
  numerical.checked_values += post.checked_values;
  numerical.max_absolute_error = std::max(numerical.max_absolute_error, post.max_absolute_error);
  numerical.max_relative_error = std::max(numerical.max_relative_error, post.max_relative_error);
  numerical.passed = numerical.passed && post.passed;
  const uint64_t checksum = sfu_sink_checksum(sink.ptr, o.sfu_lanes);
  const uint64_t allocated_bytes = (o.sfu_lanes + (paired_context ? o.sfu_lanes : 1) + 16) * sizeof(uint32_t);
  auto print_arm = [&](const RunTiming& t, bool control) {
    const uint64_t launches = t.batches * o.batch_launches;
    const long double slots = (long double)launches * o.sfu_lanes * o.iterations * o.sfu_chains;
    std::cout << std::setprecision(17) << "{\"type\":" << quote(control ? "active_reference_result" : "result")
      << ",\"workload\":" << quote(control ? "control" : o.workload) << ",\"access\":\"read\""
      << ",\"reference_kind\":" << (control ? "\"register_loop_without_sfu\"" : "null")
      << ",\"kernel_implementation_version\":" << quote(control ? "sfu_register_control_v1" : "sfu_register_recurrence_v1");
    print_sfu_metadata(o, operation);
    std::cout << ",\"duration_s\":" << t.device_s << ",\"host_duration_s\":" << t.host_s
      << ",\"batches\":" << t.batches << ",\"batch_launches\":" << o.batch_launches
      << ",\"iterations_per_launch\":" << o.iterations << ",\"iterations_per_batch\":" << o.iterations * o.batch_launches
      << ",\"kernel_launches\":" << launches << ",\"workload_launches\":" << launches
      << ",\"fixed_batches\":" << o.fixed_batches << ",\"warmup_batches\":" << o.warmup_batches
      << ",\"mean_batch_duration_s\":" << t.device_s / t.batches
      << ",\"duration_scope\":\"CUDA event elapsed experiment window including gaps; not summed kernel busy time\""
      << ",\"blocks\":" << o.blocks << ",\"threads\":" << o.threads << ",\"launch_geometry_applies\":true"
      << ",\"admitted_blocks\":" << launches * o.blocks
      << ",\"sfu_instructions\":" << (control ? 0 : slots) << ",\"reference_loop_slots\":" << (control ? slots : 0)
      << ",\"operations\":" << (control ? 0 : slots) << ",\"logical_bytes\":0,\"elements\":0"
      << ",\"operation_unit\":" << quote(control ? "not target work; common register-loop slots" : "scalar-lane native SFU instructions; not FLOPs or complete-function elements")
      << ",\"throughput_ops_s\":" << (control ? 0 : slots / t.device_s) << ",\"throughput_bytes_s\":0"
      << ",\"working_set_bytes\":0,\"requested_working_set_bytes\":0,\"allocated_bytes\":" << allocated_bytes
      << ",\"seed\":" << o.seed << ",\"checksum\":" << (control ? sfu_sink_checksum(reference_sink.ptr, o.sfu_lanes) : checksum)
      << ",\"checksum_kind\":\"distributed_per_lane_register_fnv_hash_sample\""
      << ",\"checksum_scope\":\"up to 64 distributed integer sink values after measurement; each lane hashes all chains; not a CPU recurrence or issued-instruction proof\""
      << ",\"kernel_resources\":";
    print_sfu_resources(control ? reference_resources : resources, o, p);
    std::cout << ",\"control_kernel_resources\":";
    print_sfu_resources(reference_resources, o, p);
    std::cout << ",\"numerical_validation\":{\"checked_values\":" << (control ? 0 : numerical.checked_values)
      << ",\"max_absolute_error\":" << (control ? 0 : numerical.max_absolute_error)
      << ",\"max_relative_error\":" << (control ? 0 : numerical.max_relative_error)
      << ",\"absolute_tolerance\":0.000002,\"relative_tolerance\":0.00002"
      << ",\"scope\":\"native primitive one-step outputs on eight distributed lane/chain inputs vs CPU double before and after phases; not full approximate recurrence/hash equivalence; all recurrence operands are bounded positive normal FP32 [0.5,1) by bit construction\"}"
      << ",\"sanity\":{\"finite_output_sample\":" << (control || numerical.passed ? "true" : "false")
      << ",\"numerical_validation_passed\":" << (control ? "null" : numerical.passed ? "true" : "false")
      << ",\"requested_sm_coverage_complete\":true}"
      << ",\"profile_region\":" << (o.profile_region ? "true" : "false")
      << ",\"paired_reference_context_allocated\":" << (paired_context ? "true" : "false")
      << ",\"sfu_scope\":\"native approximate register instruction recurrence; common XOR/mantissa remap, loop and register issue remain; one 4B/lane hash epilogue outside hot loop; no physical SFU rail attribution\""
      << ",\"active_sm_ids\":[],\"missing_requested_sm_ids\":[],\"active_sm_distribution_status\":\"not_measured_register_sfu\""
      << ",\"sm_distribution_scope\":\"no per-CTA SMID/admission telemetry; synchronized launch completion proves grid completion, not physical SM coverage\""
      << ",\"measure_epochs\":";
    print_sfu_epochs(t, o, operation, control);
    std::cout << ",\"auxiliary_work\":{\"logical_result_output_bytes\":" << (long double)launches * o.sfu_lanes * 4
      << ",\"sm_admission_atomic_updates\":0,\"host_counter_readback_bytes\":0"
      << ",\"scope\":\"one final 4B integer hash store per logical lane/launch, outside recurrence loop; not observed physical cache/DRAM traffic; shared initialization, bit remap, loop and epilogue register instructions remain\"}"
      << ",\"epoch_scope\":\"approximately one second, exact synchronized completed batches; Q/counts and completed CTAs derived from launch geometry, no admission counters or device counter readbacks\""
      << ",\"cache_policy\":\"no global input buffer or hot-loop global/shared/local load/store instructions; immutable kernel parameters may be constant-bank operands; final integer sink stores only\""
      << ",\"input_precision\":\"fp32\",\"accumulator_precision\":\"uint32 feedback bits\"}" << std::endl;
  };
  if (paired_reference) print_arm(reference_timing, true);
  print_arm(timing, false);
  if (!numerical.passed) throw std::runtime_error("native SFU one-step output failed CPU double validation after measurement");
}
