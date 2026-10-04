// Steady GPU loads for energy experiments. The host runner owns clock policy,
// power sampling, baseline pairing, sensor qualification and attribution.
// Counts here describe issued *logical* work, never measured cache/DRAM traffic.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <time.h>

namespace {
constexpr unsigned kSmSlots = 4096;
constexpr int kMaxAccumulators = 8;

void check_cuda(cudaError_t code, const char* where) {
  if (code != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(code));
}
void check_blas(cublasStatus_t code, const char* where) {
  if (code != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(where) + ": cuBLAS status " + std::to_string(code));
}
#define CUDA_CHECK(x) check_cuda((x), #x)
#define BLAS_CHECK(x) check_blas((x), #x)

uint64_t monotonic_ns() {
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC) failed");
  return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}

std::string quote(const std::string& value) {
  std::ostringstream s; s << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') s << '\\' << c;
    else if (c < 32) s << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else s << c;
  }
  s << '"'; return s.str();
}
void phase(const char* name, const char* event) {
  std::cout << "{\"type\":\"phase\",\"phase\":" << quote(name)
            << ",\"event\":" << quote(event) << ",\"host_monotonic_ns\":"
            << monotonic_ns() << "}" << std::endl;
}

struct Options {
  int device = 0, blocks = 0, threads = 256, accumulators = 4;
  int gemm_m = 4096, gemm_n = 4096, gemm_k = 4096;
  double seconds = 10, warmup_seconds = 3, idle_seconds = 6;
  uint64_t working_set_bytes = 0, stride_elements = 1, iterations = 0;
  int batch_launches = 0;
  uint64_t fixed_batches = 0, warmup_batches = 0;
  uint64_t offset_bytes = 0, seed = 1;
  std::string workload = "tensor", access = "read";
  std::vector<unsigned> sm_ids;
  bool describe = false;
};

uint64_t parse_u64(const std::string& value, const char* flag) {
  if (value.empty() || value.front() == '-') throw std::runtime_error(std::string(flag) + " requires an unsigned integer");
  size_t used = 0;
  uint64_t out = std::stoull(value, &used, 10);
  if (used != value.size()) throw std::runtime_error(std::string(flag) + " requires an integer");
  return out;
}
int parse_int(const std::string& value, const char* flag) {
  uint64_t out = parse_u64(value, flag);
  if (out > uint64_t(std::numeric_limits<int>::max())) throw std::runtime_error(std::string(flag) + " is too large");
  return int(out);
}
double parse_time(const std::string& value, const char* flag) {
  size_t used = 0; double out = std::stod(value, &used);
  if (used != value.size() || !std::isfinite(out) || out < 0)
    throw std::runtime_error(std::string(flag) + " requires a finite nonnegative number");
  return out;
}
void usage() {
  std::cout << "powerbench --describe [--device N]\n"
    "powerbench --workload tensor|gemm|l1|l2|l2_latency|hbm|control [options]\n"
    "  --device N --seconds 10 --warmup-seconds 3 --idle-seconds 6\n"
    "  --blocks N --threads 256 --iterations N --tensor-accumulators 1..8\n"
    "  --batch-launches N (default 16 microkernels, 1 GEMM)\n"
    "  --fixed-batches N --warmup-batches N (profiler mode; override timed loops)\n"
    "  --working-set-bytes N --stride-elements N --offset-bytes N\n"
    "  --access read|write|copy --sm-ids 0,1,... --seed N\n"
    "  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096\n"
    "Memory stride is in 32-bit words; footprint is total input bytes.\n"
    "L1 splits that footprint into disjoint per-block slices and supports read only.\n"
    "SM filters admit dispatched blocks, and do not disable SMs or select GPCs.\n"
    "cuBLAS GEMM does not support SM filters or requested launch geometry.\n"
    "stdout is JSONL, except for this --help text.\n";
}
Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    if (flag == "--help" || flag == "-h") { usage(); std::exit(0); }
    if (flag == "--describe") { o.describe = true; continue; }
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
    std::string value = argv[++i];
    if (flag == "--device") o.device = parse_int(value, flag.c_str());
    else if (flag == "--blocks") o.blocks = parse_int(value, flag.c_str());
    else if (flag == "--threads") o.threads = parse_int(value, flag.c_str());
    else if (flag == "--batch-launches") o.batch_launches = parse_int(value, flag.c_str());
    else if (flag == "--fixed-batches") o.fixed_batches = parse_u64(value, flag.c_str());
    else if (flag == "--warmup-batches") o.warmup_batches = parse_u64(value, flag.c_str());
    else if (flag == "--seconds") o.seconds = parse_time(value, flag.c_str());
    else if (flag == "--warmup-seconds") o.warmup_seconds = parse_time(value, flag.c_str());
    else if (flag == "--idle-seconds") o.idle_seconds = parse_time(value, flag.c_str());
    else if (flag == "--iterations") o.iterations = parse_u64(value, flag.c_str());
    else if (flag == "--working-set-bytes") o.working_set_bytes = parse_u64(value, flag.c_str());
    else if (flag == "--stride-elements") o.stride_elements = parse_u64(value, flag.c_str());
    else if (flag == "--offset-bytes") o.offset_bytes = parse_u64(value, flag.c_str());
    else if (flag == "--seed") o.seed = parse_u64(value, flag.c_str());
    else if (flag == "--tensor-accumulators") o.accumulators = parse_int(value, flag.c_str());
    else if (flag == "--workload") o.workload = value;
    else if (flag == "--access") o.access = value;
    else if (flag == "--gemm-m") o.gemm_m = parse_int(value, flag.c_str());
    else if (flag == "--gemm-n") o.gemm_n = parse_int(value, flag.c_str());
    else if (flag == "--gemm-k") o.gemm_k = parse_int(value, flag.c_str());
    else if (flag == "--sm-ids") {
      std::stringstream ss(value); std::string item;
      while (std::getline(ss, item, ',')) {
        uint64_t id = parse_u64(item, "--sm-ids");
        if (id >= kSmSlots) throw std::runtime_error("SM ID is outside supported telemetry range");
        o.sm_ids.push_back(unsigned(id));
      }
      if (o.sm_ids.empty()) throw std::runtime_error("--sm-ids cannot be empty");
      std::sort(o.sm_ids.begin(), o.sm_ids.end());
      o.sm_ids.erase(std::unique(o.sm_ids.begin(), o.sm_ids.end()), o.sm_ids.end());
    } else throw std::runtime_error("unknown option " + flag);
  }
  if (o.seconds <= 0) throw std::runtime_error("--seconds must be positive");
  if (o.stride_elements == 0 || o.stride_elements > (1ULL << 32)) throw std::runtime_error("invalid --stride-elements");
  if (o.accumulators < 1 || o.accumulators > kMaxAccumulators) throw std::runtime_error("--tensor-accumulators must be 1..8");
  if (o.threads < 32 || o.threads > 1024 || o.threads % 32) throw std::runtime_error("--threads must be a multiple of 32 between 32 and 1024");
  if (o.offset_bytes % 4 || o.working_set_bytes % 4) throw std::runtime_error("memory offset/footprint must be multiples of 4 bytes");
  if (o.iterations > (1ULL << 32)) throw std::runtime_error("--iterations must be <= 2^32");
  if (o.batch_launches > 65536) throw std::runtime_error("--batch-launches must be <= 65536");
  if (o.workload != "tensor" && o.workload != "gemm" && o.workload != "l1" && o.workload != "l2" && o.workload != "l2_latency" && o.workload != "hbm" && o.workload != "control")
    throw std::runtime_error("unknown workload " + o.workload);
  if (o.access != "read" && o.access != "write" && o.access != "copy") throw std::runtime_error("unknown memory access " + o.access);
  if (o.workload == "l1" && o.access != "read") throw std::runtime_error("L1 global-store/copy attribution is unsupported; use L1 read");
  if (o.workload == "l2_latency" && o.access != "read") throw std::runtime_error("dependent latency probes support read only");
  if (o.workload == "l2_latency" && o.stride_elements != 1) throw std::runtime_error("latency probes use randomized dependent links; --stride-elements must be 1");
  if (o.workload == "gemm" && !o.sm_ids.empty()) throw std::runtime_error("cuBLAS GEMM cannot honor --sm-ids");
  if (o.gemm_m <= 0 || o.gemm_n <= 0 || o.gemm_k <= 0) throw std::runtime_error("GEMM dimensions must be positive");
  return o;
}

template<class T> struct Buffer {
  T* ptr = nullptr;
  explicit Buffer(size_t count) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max(size_t(1), count) * sizeof(T))); }
  ~Buffer() { if (ptr) cudaFree(ptr); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
struct Event {
  cudaEvent_t event{};
  Event() { CUDA_CHECK(cudaEventCreate(&event)); }
  ~Event() { cudaEventDestroy(event); }
};
struct Blas {
  cublasHandle_t handle{};
  Blas() { BLAS_CHECK(cublasCreate(&handle)); BLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)); }
  ~Blas() { cublasDestroy(handle); }
};

__device__ __forceinline__ unsigned sm_id() {
  unsigned value; asm volatile("mov.u32 %0, %%smid;" : "=r"(value)); return value;
}
__device__ __forceinline__ uint32_t random_word(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return uint32_t(x ^ (x >> 31));
}
__global__ void init_words(uint32_t* data, uint64_t n, uint64_t seed) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += uint64_t(gridDim.x) * blockDim.x)
    data[i] = random_word(i ^ seed);
}
__global__ void init_halves(__half* data, uint64_t n, uint64_t seed) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += uint64_t(gridDim.x) * blockDim.x)
    data[i] = __float2half((float(int(random_word(i ^ seed) & 65535U)) - 32768.0f) / 131072.0f);
}

// No scheduling guarantee: the hardware may dispatch blocks only to a subset
// of requested SMs. Every admitted block records its actual physical SM ID.
__device__ __forceinline__ bool admit(const unsigned char* mask, bool filtered, unsigned long long* blocks_per_sm) {
  unsigned id = sm_id();
  bool ok = id < kSmSlots && (!filtered || mask[id]);
  if (ok && threadIdx.x == 0) atomicAdd(blocks_per_sm + id, 1ULL);
  return ok;
}
__device__ __forceinline__ uint32_t load_ca(const uint32_t* p) {
  uint32_t value; asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory"); return value;
}
__device__ __forceinline__ uint32_t load_cg(const uint32_t* p) {
  uint32_t value; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory"); return value;
}
__device__ __forceinline__ void store_cg(uint32_t* p, uint32_t value) {
  asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(p), "r"(value) : "memory");
}
__device__ __forceinline__ uint64_t wrap(uint64_t x, uint64_t n, bool pow2) { return pow2 ? (x & (n - 1)) : (x % n); }
// All inputs have been normalized to [0,n). Avoid 64-bit division in the hot
// loop even when a 20/25 MiB L2 footprint is not a power of two.
__device__ __forceinline__ uint64_t add_wrap(uint64_t a, uint64_t b, uint64_t n) {
  uint64_t sum = a + b; return sum >= n ? sum - n : sum;
}

template<bool L1, int Access> // 0 = read, 1 = write, 2 = copy
__global__ void memory_kernel(const uint32_t* input, uint32_t* output, uint32_t* sink,
    uint64_t total_words, uint64_t slice_words, uint64_t stride, uint64_t iterations,
    const unsigned char* mask, bool filtered, unsigned long long* blocks_per_sm) {
  if (!admit(mask, filtered, blocks_per_sm)) return;
  const uint64_t n = L1 ? slice_words : total_words;
  const uint64_t start = L1 ? uint64_t(blockIdx.x) * slice_words : 0;
  const uint64_t lanes = L1 ? blockDim.x : uint64_t(gridDim.x) * blockDim.x;
  const uint64_t tid = L1 ? threadIdx.x : uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool pow2 = (n & (n - 1)) == 0;
  // Lanes access consecutive words for stride=1. Four streams are separated by
  // a whole participating grid, keeping each load coalesced at sector level.
  uint64_t base = wrap(tid * stride, n, pow2);
  uint64_t delta1 = wrap(lanes * stride, n, pow2);
  uint64_t delta2 = wrap(lanes * 2 * stride, n, pow2);
  uint64_t delta3 = wrap(lanes * 3 * stride, n, pow2);
  uint64_t advance = wrap(lanes * 4 * stride, n, pow2);
  uint32_t a = random_word(tid), b = a ^ 0x7f4a7c15U, c = a ^ 0xbf58476dU, d = a ^ 0x94d049bbU;
  for (uint64_t i = 0; i < iterations; ++i) {
    uint64_t j0 = start + base;
    uint64_t j1 = start + add_wrap(base, delta1, n);
    uint64_t j2 = start + add_wrap(base, delta2, n);
    uint64_t j3 = start + add_wrap(base, delta3, n);
    if constexpr (Access != 1) {
      uint32_t x0 = L1 ? load_ca(input + j0) : load_cg(input + j0);
      uint32_t x1 = L1 ? load_ca(input + j1) : load_cg(input + j1);
      uint32_t x2 = L1 ? load_ca(input + j2) : load_cg(input + j2);
      uint32_t x3 = L1 ? load_ca(input + j3) : load_cg(input + j3);
      if constexpr (Access == 2) {
        store_cg(output + j0, x0); store_cg(output + j1, x1);
        store_cg(output + j2, x2); store_cg(output + j3, x3);
      }
      // A dependency per stream keeps all inline loads observable; four streams
      // expose independent loads. Hash ALU and address generation are overhead.
      a ^= x0; b ^= x1; c ^= x2; d ^= x3;
    } else {
      // Changing, thread-dependent data resists compression and avoids repeated
      // identical stores. Overlap is permitted; it does not change payload count.
      a = a * 1664525U + 1013904223U; b ^= a + uint32_t(i);
      c = c * 22695477U + 1U; d ^= c + uint32_t(tid);
      store_cg(output + j0, a); store_cg(output + j1, b);
      store_cg(output + j2, c); store_cg(output + j3, d);
    }
    base = add_wrap(base, advance, n);
  }
  sink[uint64_t(blockIdx.x) * blockDim.x + threadIdx.x] = a ^ b ^ c ^ d;
}

__global__ void control_kernel(uint32_t* sink, uint64_t iterations,
    const unsigned char* mask, bool filtered, unsigned long long* blocks_per_sm) {
  if (!admit(mask, filtered, blocks_per_sm)) return;
  uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint32_t a = random_word(tid), b = a + 1U, c = a + 2U, d = a + 3U;
  for (uint64_t i = 0; i < iterations; ++i) {
    asm volatile("add.u32 %0, %0, %4;\n\txor.b32 %1, %1, %0;\n\tadd.u32 %2, %2, %1;\n\txor.b32 %3, %3, %2;"
      : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(uint32_t(i)));
  }
  sink[tid] = a ^ b ^ c ^ d;
}

template<int Accumulators>
__global__ void tensor_kernel(const __half* a, const __half* b, float* output,
    uint64_t iterations, const unsigned char* mask,
    bool filtered, unsigned long long* blocks_per_sm) {
  if (!admit(mask, filtered, blocks_per_sm)) return;
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[Accumulators];
  wmma::load_matrix_sync(fa, a, 16);
  wmma::load_matrix_sync(fb, b, 16);
  #pragma unroll
  for (int j = 0; j < Accumulators; ++j) wmma::fill_fragment(fc[j], 0.0f);
  for (uint64_t i = 0; i < iterations; ++i) {
    #pragma unroll
    for (int j = 0; j < Accumulators; ++j) wmma::mma_sync(fc[j], fa, fb, fc[j]);
  }
  uint64_t warp = uint64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  #pragma unroll
  for (int j = 0; j < Accumulators; ++j)
    wmma::store_matrix_sync(output + (warp * Accumulators + j) * 256, fc[j], 16, wmma::mem_row_major);
}

__global__ void latency_kernel(const uint32_t* links, uint32_t* sink,
    uint64_t words, uint64_t iterations, uint64_t nonce, const unsigned char* mask, bool filtered,
    unsigned long long* blocks_per_sm, unsigned long long* cycles_per_sm,
    unsigned long long* loads_per_sm) {
  if (!admit(mask, filtered, blocks_per_sm) || threadIdx.x != 0) return;
  // One active lane per admitted block; dependent links prevent memory-level
  // parallelism within a chain. Multiple blocks/SM can still overlap, so compare
  // identical geometry and use a validated single-SM probe for locality work.
  uint32_t index = random_word(nonce ^ (uint64_t(blockIdx.x) << 32)) % words;
  unsigned long long begin = clock64();
  for (uint64_t i = 0; i < iterations; ++i) index = load_cg(links + index);
  unsigned long long elapsed = clock64() - begin;
  sink[uint64_t(blockIdx.x) * blockDim.x] = index;
  unsigned id = sm_id();
  atomicAdd(cycles_per_sm + id, elapsed);
  atomicAdd(loads_per_sm + id, static_cast<unsigned long long>(iterations));
}

std::string uuid_string(const cudaUUID_t& uuid) {
  std::ostringstream out; out << "GPU-" << std::hex << std::setfill('0');
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
    out << std::setw(2) << unsigned(static_cast<unsigned char>(uuid.bytes[i]));
  }
  return out.str();
}
void describe(const Options& o, const cudaDeviceProp& p) {
  char pci[64]{}; CUDA_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), o.device));
  int runtime = 0, driver = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
  int max_blocks_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_blocks_sm, cudaDevAttrMaxBlocksPerMultiprocessor, o.device));
  std::cout << "{\"type\":\"device\",\"cuda_ordinal\":" << o.device
    << ",\"name\":" << quote(p.name) << ",\"uuid\":" << quote(uuid_string(p.uuid))
    << ",\"pci_bus_id\":" << quote(pci) << ",\"cc\":" << quote(std::to_string(p.major) + "." + std::to_string(p.minor))
    << ",\"compute_capability_major\":" << p.major << ",\"compute_capability_minor\":" << p.minor
    << ",\"sm_count\":" << p.multiProcessorCount << ",\"warp_size\":" << p.warpSize
    << ",\"l2_bytes\":" << p.l2CacheSize << ",\"total_memory_bytes\":" << p.totalGlobalMem
    << ",\"max_threads_per_sm\":" << p.maxThreadsPerMultiProcessor
    << ",\"max_blocks_per_sm\":" << max_blocks_sm << ",\"registers_per_sm\":" << p.regsPerMultiprocessor
    << ",\"shared_memory_per_sm_bytes\":" << p.sharedMemPerMultiprocessor
    << ",\"nominal_max_sm_clock_khz\":" << p.clockRate << ",\"nominal_max_memory_clock_khz\":" << p.memoryClockRate
    << ",\"memory_bus_width_bits\":" << p.memoryBusWidth
    << ",\"cuda_runtime_version\":" << runtime << ",\"cuda_driver_version\":" << driver << "}" << std::endl;
}

struct RunTiming { double device_s = 0, host_s = 0; uint64_t batches = 0; };
template<class Launch> RunTiming run_phase(const char* name, double seconds, uint64_t fixed_batches, Launch launch) {
  CUDA_CHECK(cudaDeviceSynchronize());
  phase(name, "start");
  uint64_t begin = monotonic_ns();
  Event start, stop, batch_done;
  CUDA_CHECK(cudaEventRecord(start.event));
  uint64_t batches = 0;
  // Synchronize once per batch. --iterations makes a batch long enough to avoid
  // host launch overhead dominating; report measured batch count and duration.
  while (fixed_batches ? batches < fixed_batches : (batches == 0 ? seconds > 0 : double(monotonic_ns() - begin) * 1e-9 < seconds)) {
    launch(); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(batch_done.event));
    CUDA_CHECK(cudaEventSynchronize(batch_done.event));
    ++batches;
  }
  CUDA_CHECK(cudaEventRecord(stop.event));
  CUDA_CHECK(cudaEventSynchronize(stop.event));
  uint64_t end = monotonic_ns();
  float elapsed_ms = 0; CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start.event, stop.event));
  phase(name, "end");
  return {double(elapsed_ms) * 1e-3, double(end - begin) * 1e-9, batches};
}
void idle_phase(const char* name, double seconds) {
  CUDA_CHECK(cudaDeviceSynchronize()); phase(name, "start");
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  phase(name, "end");
}

size_t checked_size(uint64_t n, uint64_t unit, const char* name) {
  if (n > uint64_t(std::numeric_limits<size_t>::max()) / unit) throw std::runtime_error(std::string(name) + " size overflow");
  return size_t(n);
}

void experiment(Options o, const cudaDeviceProp& p) {
  if (!o.blocks) o.blocks = 2 * p.multiProcessorCount;
  if (o.blocks > 1000000) throw std::runtime_error("--blocks exceeds practical safety limit");
  if (o.threads > p.maxThreadsPerBlock) throw std::runtime_error("thread count exceeds this device limit");
  if (p.major < 7) throw std::runtime_error("benchmark requires compute capability >= 7.0");
  const bool latency = o.workload == "l2_latency";
  const bool memory = o.workload == "l1" || o.workload == "l2" || latency || o.workload == "hbm";
  if (!o.iterations) o.iterations = o.workload == "gemm" ? 1 : 1024;
  if (!o.batch_launches) o.batch_launches = o.workload == "gemm" ? 1 : 16;
  if (!o.working_set_bytes) {
    if (o.workload == "l1") o.working_set_bytes = uint64_t(o.blocks) * 16 * 1024;
    else if (o.workload == "l2" || latency) o.working_set_bytes = std::max(uint64_t(4), uint64_t(p.l2CacheSize) / 2 / 4 * 4);
    else o.working_set_bytes = std::max(uint64_t(512) * 1024 * 1024, uint64_t(p.l2CacheSize) * 8);
  }
  uint64_t words = memory ? o.working_set_bytes / 4 : 1;
  uint64_t slice = o.workload == "l1" ? words / o.blocks : 0;
  if (memory && words == 0) throw std::runtime_error("memory footprint must contain at least one word");
  if (latency && words > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("pointer chain exceeds uint32 node-index range");
  if (o.workload == "l1" && !slice) throw std::runtime_error("L1 footprint must contain at least one word per block");
  if (o.workload == "l1") words = slice * o.blocks;
  uint64_t offset_words = memory ? o.offset_bytes / 4 : 0;
  if (words > std::numeric_limits<uint64_t>::max() - offset_words) throw std::runtime_error("allocation size overflow");
  if (memory && words + offset_words > uint64_t(p.totalGlobalMem) / 4 / (o.access == "read" ? 1 : 2) * 3 / 4)
    throw std::runtime_error("requested buffers exceed 75% of device capacity");
  uint64_t lanes = uint64_t(o.blocks) * o.threads;
  uint64_t tensor_out_count = o.workload == "tensor" ? lanes / 32 * o.accumulators * 256 : 1;
  uint64_t gemm_a_count = o.workload == "gemm" ? uint64_t(o.gemm_m) * o.gemm_k : 256;
  uint64_t gemm_b_count = o.workload == "gemm" ? uint64_t(o.gemm_k) * o.gemm_n : 256;
  uint64_t gemm_c_count = o.workload == "gemm" ? uint64_t(o.gemm_m) * o.gemm_n : tensor_out_count;
  Buffer<uint32_t> input(checked_size(words + offset_words, sizeof(uint32_t), "input"));
  Buffer<uint32_t> output(checked_size(memory && o.access != "read" ? words + offset_words : 1, sizeof(uint32_t), "output"));
  Buffer<uint32_t> sink(checked_size(lanes, sizeof(uint32_t), "sink"));
  Buffer<__half> a(checked_size(gemm_a_count, sizeof(__half), "GEMM A"));
  Buffer<__half> b(checked_size(gemm_b_count, sizeof(__half), "GEMM B"));
  Buffer<float> c(checked_size(gemm_c_count, sizeof(float), "GEMM C"));
  Buffer<unsigned char> mask(kSmSlots);
  Buffer<unsigned long long> sm_blocks(kSmSlots);
  Buffer<unsigned long long> sm_cycles(kSmSlots);
  Buffer<unsigned long long> sm_loads(kSmSlots);
  std::vector<unsigned char> mask_host(kSmSlots, o.sm_ids.empty() ? 1 : 0);
  for (unsigned id : o.sm_ids) mask_host[id] = 1;
  CUDA_CHECK(cudaMemcpy(mask.ptr, mask_host.data(), kSmSlots, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(sink.ptr, 0, size_t(lanes) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(c.ptr, 0, size_t(gemm_c_count) * sizeof(float)));
  CUDA_CHECK(cudaMemset(sm_blocks.ptr, 0, kSmSlots * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(sm_cycles.ptr, 0, kSmSlots * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(sm_loads.ptr, 0, kSmSlots * sizeof(unsigned long long)));
  init_words<<<std::min(o.blocks, 4096), 256>>>(input.ptr, words + offset_words, o.seed);
  if (memory && o.access != "read") init_words<<<std::min(o.blocks, 4096), 256>>>(output.ptr, words + offset_words, o.seed ^ 0x9e3779b9ULL);
  init_halves<<<std::min(o.blocks, 4096), 256>>>(a.ptr, gemm_a_count, o.seed);
  init_halves<<<std::min(o.blocks, 4096), 256>>>(b.ptr, gemm_b_count, o.seed ^ 12345ULL);
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  if (latency) {
    // Build a full randomized cycle, rather than random links that can collapse
    // into a short cache-resident subcycle. This setup is outside every phase.
    std::vector<uint32_t> order(static_cast<size_t>(words));
    std::iota(order.begin(), order.end(), 0U);
    std::mt19937_64 random(o.seed); std::shuffle(order.begin(), order.end(), random);
    std::vector<uint32_t> links(static_cast<size_t>(words));
    for (size_t i = 0; i < order.size(); ++i) links[order[i]] = order[(i + 1) % order.size()];
    CUDA_CHECK(cudaMemcpy(input.ptr + offset_words, links.data(), size_t(words) * sizeof(uint32_t), cudaMemcpyHostToDevice));
  }
  // Set a stable carveout preference for L1; actual shared/L1 partition and hit
  // rate still require counters. No shared memory is consumed by this kernel.
  if (o.workload == "l1")
    CUDA_CHECK(cudaFuncSetAttribute(memory_kernel<true, 0>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  Blas blas;
  const bool filtered = !o.sm_ids.empty();
  uint64_t launch_nonce = o.seed;
  auto single_launch = [&] {
    ++launch_nonce;
    if (o.workload == "tensor") {
      // Compile-time fragment indexing keeps the independent accumulator set
      // eligible for registers. Inspect ptxas/Nsight spill counters to qualify.
      #define TENSOR_CASE(N) case N: tensor_kernel<N><<<o.blocks, o.threads>>>(a.ptr, b.ptr, c.ptr, o.iterations, mask.ptr, filtered, sm_blocks.ptr); break
      switch (o.accumulators) {
        TENSOR_CASE(1); TENSOR_CASE(2); TENSOR_CASE(3); TENSOR_CASE(4);
        TENSOR_CASE(5); TENSOR_CASE(6); TENSOR_CASE(7); TENSOR_CASE(8);
      }
      #undef TENSOR_CASE
    } else if (o.workload == "gemm") {
      float alpha = 1.0f, beta = 0.0f;
      for (uint64_t i = 0; i < o.iterations; ++i)
        BLAS_CHECK(cublasGemmEx(blas.handle, CUBLAS_OP_N, CUBLAS_OP_N,
          o.gemm_m, o.gemm_n, o.gemm_k, &alpha, a.ptr, CUDA_R_16F, o.gemm_m,
          b.ptr, CUDA_R_16F, o.gemm_k, &beta, c.ptr, CUDA_R_32F, o.gemm_m,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (o.workload == "control") {
      control_kernel<<<o.blocks, o.threads>>>(sink.ptr, o.iterations, mask.ptr, filtered, sm_blocks.ptr);
    } else if (latency) {
      latency_kernel<<<o.blocks, o.threads>>>(input.ptr + offset_words, sink.ptr, words, o.iterations, launch_nonce,
        mask.ptr, filtered, sm_blocks.ptr, sm_cycles.ptr, sm_loads.ptr);
    } else {
      const uint32_t* in = input.ptr + offset_words; uint32_t* out = output.ptr + (o.access == "read" ? 0 : offset_words);
      if (o.workload == "l1") memory_kernel<true, 0><<<o.blocks, o.threads>>>(in, out, sink.ptr, words, slice, o.stride_elements, o.iterations, mask.ptr, filtered, sm_blocks.ptr);
      else if (o.access == "read") memory_kernel<false, 0><<<o.blocks, o.threads>>>(in, out, sink.ptr, words, 0, o.stride_elements, o.iterations, mask.ptr, filtered, sm_blocks.ptr);
      else if (o.access == "write") memory_kernel<false, 1><<<o.blocks, o.threads>>>(in, out, sink.ptr, words, 0, o.stride_elements, o.iterations, mask.ptr, filtered, sm_blocks.ptr);
      else memory_kernel<false, 2><<<o.blocks, o.threads>>>(in, out, sink.ptr, words, 0, o.stride_elements, o.iterations, mask.ptr, filtered, sm_blocks.ptr);
    }
  };
  auto launch = [&] {
    // Queue several microkernels before waiting on the host. The actual counts
    // and elapsed window include launch gaps; they are not profiler busy time.
    for (int i = 0; i < o.batch_launches; ++i) single_launch();
  };
  run_phase("warmup", o.warmup_seconds, o.warmup_batches, launch);
  idle_phase("idle_pre", o.idle_seconds);
  CUDA_CHECK(cudaMemset(sm_blocks.ptr, 0, kSmSlots * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(sm_cycles.ptr, 0, kSmSlots * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(sm_loads.ptr, 0, kSmSlots * sizeof(unsigned long long)));
  RunTiming timing = run_phase("measure", o.seconds, o.fixed_batches, launch);
  idle_phase("idle_post", o.idle_seconds);
  std::vector<unsigned long long> observed(kSmSlots);
  CUDA_CHECK(cudaMemcpy(observed.data(), sm_blocks.ptr, kSmSlots * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  unsigned long long admitted_blocks = 0; std::vector<unsigned> active;
  for (unsigned id = 0; id < kSmSlots; ++id) if (observed[id]) { active.push_back(id); admitted_blocks += observed[id]; }
  if (o.workload != "gemm" && !admitted_blocks) throw std::runtime_error("no blocks were admitted by the SM filter");
  std::vector<unsigned> missing;
  for (unsigned id : o.sm_ids) if (!observed[id]) missing.push_back(id);
  long double operations = 0, logical_bytes = 0;
  if (o.workload == "tensor") operations = (long double)admitted_blocks * (o.threads / 32) * o.iterations * o.accumulators * 8192;
  else if (o.workload == "gemm") operations = (long double)timing.batches * o.batch_launches * o.iterations * 2 * o.gemm_m * o.gemm_n * o.gemm_k;
  else if (latency) { operations = (long double)admitted_blocks * o.iterations; logical_bytes = operations * 4; }
  else if (memory) {
    operations = (long double)admitted_blocks * o.threads * o.iterations * 4;
    logical_bytes = operations * 4 * (o.access == "copy" ? 2 : 1);
  }
  uint64_t checksum = 0; bool finite = true;
  const char* checksum_kind = "uint32_sink_sample";
  if (o.workload == "tensor" || o.workload == "gemm") {
    std::array<float, 256> sample{}; size_t count = std::min(size_t(256), size_t(gemm_c_count));
    CUDA_CHECK(cudaMemcpy(sample.data(), c.ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) { uint32_t bits; std::memcpy(&bits, &sample[i], sizeof(bits)); checksum = checksum * 1315423911ULL + bits; finite = finite && std::isfinite(sample[i]); }
    checksum_kind = "fp32_output_sample_hash";
  } else {
    std::vector<uint32_t> sample(std::min(size_t(1024), size_t(lanes)));
    CUDA_CHECK(cudaMemcpy(sample.data(), sink.ptr, sample.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (uint32_t word : sample) checksum = checksum * 1315423911ULL + word;
    // Read-only repeated XOR can cancel by design. This verifies a defined output
    // transfer, not a full memory correctness test or validation of cache residency.
  }
  uint64_t allocated_bytes = (words + offset_words + (memory && o.access != "read" ? words + offset_words : 1) + lanes) * 4
    + (gemm_a_count + gemm_b_count) * 2 + gemm_c_count * 4 + kSmSlots * 25;
  uint64_t reachable_words = memory ? (latency ? words : (o.workload == "l1" ? slice / std::gcd(slice, o.stride_elements) * o.blocks : words / std::gcd(words, o.stride_elements))) : 0;
  // Sector footprint is a full-period bound, not observed traffic. For total
  // word counts divisible by 8 and offset aligned to 32B it is exact for the
  // strided address set; otherwise use a conservative upper bound.
  uint64_t sector_bound = memory ? std::min((words + (o.offset_bytes % 32) / 4 + 7) / 8 * 32, reachable_words * 32) : 0;
  bool sector_exact = memory && words % 8 == 0 && o.offset_bytes % 32 == 0 && o.workload != "l1";
  if (sector_exact) sector_bound = latency ? words * 4 : (std::gcd(words, o.stride_elements) <= 8 ? words * 4 : reachable_words * 32);
  std::vector<unsigned long long> cycles(kSmSlots), loads(kSmSlots);
  if (latency) {
    CUDA_CHECK(cudaMemcpy(cycles.data(), sm_cycles.ptr, kSmSlots * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(loads.data(), sm_loads.ptr, kSmSlots * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  }
  std::cout << std::setprecision(17)
    << "{\"type\":\"result\",\"workload\":" << quote(o.workload) << ",\"access\":" << quote(o.access)
    << ",\"duration_s\":" << timing.device_s << ",\"host_duration_s\":" << timing.host_s
    << ",\"batches\":" << timing.batches << ",\"iterations_per_launch\":" << o.iterations
    << ",\"iterations_per_batch\":" << o.iterations * o.batch_launches
    << ",\"batch_launches\":" << o.batch_launches << ",\"workload_launches\":" << timing.batches * o.batch_launches
    << ",\"kernel_launches\":" << (o.workload == "gemm" ? "null" : std::to_string(timing.batches * o.batch_launches))
    << ",\"gemm_invocations\":" << (o.workload == "gemm" ? timing.batches * o.batch_launches * o.iterations : 0)
    << ",\"fixed_batches\":" << o.fixed_batches << ",\"warmup_batches\":" << o.warmup_batches
    << ",\"mean_batch_duration_s\":" << timing.device_s / timing.batches
    << ",\"duration_scope\":\"CUDA event elapsed experiment window including gaps; not summed kernel busy time\""
    << ",\"operations\":" << operations << ",\"operation_unit\":" << quote(memory ? "32-bit memory accesses (copy pairs count once)" : o.workload == "control" ? "not attributed" : "dense FP FLOPs (FMA=2)")
    << ",\"logical_bytes\":" << logical_bytes
    << ",\"throughput_ops_s\":" << operations / timing.device_s
    << ",\"throughput_bytes_s\":" << logical_bytes / timing.device_s
    << ",\"blocks\":" << o.blocks << ",\"threads\":" << o.threads
    << ",\"launch_geometry_applies\":" << (o.workload == "gemm" ? "false" : "true")
    << ",\"admitted_blocks\":" << admitted_blocks
    << ",\"working_set_bytes\":" << (memory ? words * 4 : 0)
    << ",\"allocated_bytes\":" << allocated_bytes << ",\"l1_bytes_per_block\":" << slice * 4
    << ",\"logical_reachable_bytes\":" << reachable_words * 4
    << ",\"potential_cache_sector_bytes\":" << sector_bound
    << ",\"potential_sector_footprint_exact\":" << (sector_exact ? "true" : "false")
    << ",\"reachable_footprint_scope\":\"potential over complete stride/chain cycle; finite batches and SM admission may touch less; sector count is a bound when unaligned\""
    << ",\"stride_elements\":" << o.stride_elements << ",\"offset_bytes\":" << o.offset_bytes
    << ",\"tensor_accumulators\":" << o.accumulators
    << ",\"gemm_m\":" << o.gemm_m << ",\"gemm_n\":" << o.gemm_n << ",\"gemm_k\":" << o.gemm_k
    << ",\"seed\":" << o.seed << ",\"checksum\":" << checksum << ",\"checksum_kind\":" << quote(checksum_kind)
    << ",\"sanity\":{\"finite_output_sample\":" << (finite ? "true" : "false")
    << ",\"requested_sm_coverage_complete\":" << (missing.empty() ? "true" : "false") << "}"
    << ",\"active_sm_ids\":[";
  for (size_t i = 0; i < active.size(); ++i) std::cout << (i ? "," : "") << active[i];
  std::cout << "],\"missing_requested_sm_ids\":[";
  for (size_t i = 0; i < missing.size(); ++i) std::cout << (i ? "," : "") << missing[i];
  std::cout << "],\"allocated_sm_working_set\":{\"meaning\":\"per-block L1 slice times mean dispatched blocks per measured kernel launch; not concurrent residency\",\"bytes_per_sm\":{";
  if (o.workload == "l1")
    for (size_t i = 0; i < active.size(); ++i) std::cout << (i ? "," : "") << quote(std::to_string(active[i])) << ":" << (long double)observed[active[i]] * slice * 4 / (timing.batches * o.batch_launches);
  std::cout << "}},\"latency_probe\":{\"enabled\":" << (latency ? "true" : "false")
    << ",\"scope\":\"dependent .cg chain, one lane per admitted block; cycles include address and loop overhead; no near/far label without verified mapping\",\"per_sm\":{";
  if (latency) {
    for (size_t i = 0; i < active.size(); ++i) {
      unsigned id = active[i];
      std::cout << (i ? "," : "") << quote(std::to_string(id)) << ":{\"loads\":" << loads[id]
        << ",\"cycles\":" << cycles[id] << ",\"cycles_per_access\":" << (loads[id] ? double(cycles[id]) / loads[id] : 0) << "}";
    }
  }
  std::cout << "}},\"cache_policy\":" << quote(o.workload == "l1" ? "ld.global.ca; prefer maximum L1 carveout" : memory ? "ld/st.global.cg; L1 bypass; L2/DRAM residency requires counters" : "not applicable")
    << ",\"memory_count_scope\":\"logical requested payload; excludes sector inflation, writeback, initialization, telemetry and sink transfers\""
    << ",\"sm_filter_scope\":\"best-effort dispatched-block admission; no physical SM disable or GPC mapping; GEMM SM IDs unavailable\""
    << ",\"control_scope\":\"integer issue-loop reference, not matched cache/tensor dynamic power or transistor static power\""
    << ",\"tensor_scope\":\"WMMA register-operand reuse with identical small matrices across warps; compare randomized cuBLAS GEMM for sustained dense peak\""
    << ",\"sparsity\":\"dense\",\"input_precision\":\"fp16 for tensor/gemm; uint32 memory payload otherwise\",\"accumulator_precision\":\"fp32\"}" << std::endl;
  if (!finite) throw std::runtime_error("non-finite Tensor/GEMM output sample");
  if (!missing.empty()) throw std::runtime_error("requested SM coverage incomplete; reject this trial");
}
} // namespace

int main(int argc, char** argv) {
  try {
    Options o = parse_options(argc, argv);
    CUDA_CHECK(cudaSetDevice(o.device));
    cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p, o.device));
    describe(o, p);
    if (!o.describe) experiment(o, p);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "powerbench: " << error.what() << std::endl;
    return 1;
  }
}
