// Steady GPU loads for energy experiments. The host runner owns clock policy,
// power sampling, baseline pairing, sensor qualification and attribution.
// Counts here describe issued *logical* work, never measured cache/DRAM traffic.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <math_constants.h>
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
#include <functional>
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
#include <unistd.h>

namespace {
constexpr unsigned kSmSlots = 4096;
constexpr int kMaxAccumulators = 8;
bool nonlinear_workload(const std::string& name) {
  return name == "exp" || name == "tanh" || name == "silu" || name == "rmsnorm" || name == "softmax";
}

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
  int row_width = 0;
  double seconds = 10, warmup_seconds = 3, idle_seconds = 6;
  uint64_t working_set_bytes = 0, stride_elements = 1, iterations = 0;
  int batch_launches = 0;
  uint64_t fixed_batches = 0, warmup_batches = 0;
  uint64_t offset_bytes = 0, seed = 1;
  std::string workload = "tensor", access = "read", reference_order = "AB";
  std::vector<unsigned> sm_ids;
  bool describe = false, profile_region = false, paired_reference = false;
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
    "powerbench --workload tensor|gemm|l1|l2|l2_latency|hbm|control|exp|tanh|silu|rmsnorm|softmax [options]\n"
    "  --device N --seconds 10 --warmup-seconds 3 --idle-seconds 6\n"
    "  --blocks N --threads 256 --iterations N --tensor-accumulators 1..8\n"
    "  --batch-launches N (default 16 microkernels, 1 GEMM)\n"
    "  --fixed-batches N --warmup-batches N (profiler mode; override timed loops)\n"
    "  --profile-region (cudaProfilerStart/Stop bracket measure only; suppresses paired reference)\n"
    "  --paired-reference --reference-order AB|BA (A=issue-loop reference, B=treatment)\n"
    "  --working-set-bytes N --stride-elements N --offset-bytes N\n"
    "  --access read|write|copy --sm-ids 0,1,... --seed N\n"
    "  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096\n"
    "  --row-width 1024 (RMSNorm/Softmax, FP32 full row operations)\n"
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
    if (flag == "--profile-region") { o.profile_region = true; continue; }
    if (flag == "--paired-reference") { o.paired_reference = true; continue; }
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
    std::string value = argv[++i];
    if (flag == "--device") o.device = parse_int(value, flag.c_str());
    else if (flag == "--blocks") o.blocks = parse_int(value, flag.c_str());
    else if (flag == "--threads") o.threads = parse_int(value, flag.c_str());
    else if (flag == "--row-width") {
      o.row_width = parse_int(value, flag.c_str());
      if (!o.row_width) throw std::runtime_error("--row-width must be positive");
    }
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
    else if (flag == "--reference-order") o.reference_order = value;
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
  if (o.reference_order != "AB" && o.reference_order != "BA") throw std::runtime_error("--reference-order must be AB or BA");
  if (o.paired_reference && !o.profile_region && o.workload != "control" && (o.seconds < 10 || o.warmup_seconds < 1 || o.idle_seconds < 6))
    throw std::runtime_error("paired energy arms require seconds>=10, warmup-seconds>=1 and idle-seconds>=6");
  if (o.stride_elements == 0 || o.stride_elements > (1ULL << 32)) throw std::runtime_error("invalid --stride-elements");
  if (o.accumulators < 1 || o.accumulators > kMaxAccumulators) throw std::runtime_error("--tensor-accumulators must be 1..8");
  if (o.threads < 32 || o.threads > 1024 || o.threads % 32) throw std::runtime_error("--threads must be a multiple of 32 between 32 and 1024");
  if (o.offset_bytes % 4 || o.working_set_bytes % 4) throw std::runtime_error("memory offset/footprint must be multiples of 4 bytes");
  if (o.iterations > (1ULL << 32)) throw std::runtime_error("--iterations must be <= 2^32");
  if (o.batch_launches > 65536) throw std::runtime_error("--batch-launches must be <= 65536");
  if (o.workload != "tensor" && o.workload != "gemm" && o.workload != "l1" && o.workload != "l2" && o.workload != "l2_latency" && o.workload != "hbm" && o.workload != "control" && !nonlinear_workload(o.workload))
    throw std::runtime_error("unknown workload " + o.workload);
  if (o.access != "read" && o.access != "write" && o.access != "copy") throw std::runtime_error("unknown memory access " + o.access);
  if (o.workload == "l1" && o.access != "read") throw std::runtime_error("L1 global-store/copy attribution is unsupported; use L1 read");
  if (o.workload == "l2_latency" && o.access != "read") throw std::runtime_error("dependent latency probes support read only");
  if (o.workload == "l2_latency" && o.stride_elements != 1) throw std::runtime_error("latency probes use randomized dependent links; --stride-elements must be 1");
  if (o.workload == "gemm" && !o.sm_ids.empty()) throw std::runtime_error("cuBLAS GEMM cannot honor --sm-ids");
  if (o.gemm_m <= 0 || o.gemm_n <= 0 || o.gemm_k <= 0) throw std::runtime_error("GEMM dimensions must be positive");
  const bool rowwise = o.workload == "rmsnorm" || o.workload == "softmax";
  if (rowwise && !o.row_width) o.row_width = 1024;
  if ((rowwise && o.row_width > 65536) || (!rowwise && o.row_width))
    throw std::runtime_error("--row-width 1..65536 applies only to RMSNorm/Softmax");
  if (nonlinear_workload(o.workload) && (!o.sm_ids.empty() || o.offset_bytes || o.stride_elements != 1 || o.access != "read"))
    throw std::runtime_error("nonlinear workloads require a full grid, offset=0, stride=1 and access=read");
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
      // Changing, thread-dependent data resists compression. The host rounds
      // write/copy footprints to lanes*stride multiples, so distinct threads
      // own disjoint addresses across every wrap; no conflicting plain stores.
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

#include "nonlinear.cuh"

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

// Query compiled resources outside the measured phases. This is an occupancy
// upper bound for one homogeneous kernel, never a measurement of achieved
// occupancy, concurrent CTA count, Tensor utilization or cache residency.
struct KernelResources {
  bool available = false;
  cudaFuncAttributes attributes{};
  int max_active_blocks_per_sm = 0;
  size_t dynamic_shared_bytes_per_block = 0;
};
template<class Kernel>
KernelResources query_kernel_resources(Kernel kernel, int threads, size_t dynamic_shared_bytes = 0) {
  KernelResources result;
  CUDA_CHECK(cudaFuncGetAttributes(&result.attributes, kernel));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &result.max_active_blocks_per_sm, kernel, threads, dynamic_shared_bytes));
  result.available = true;
  result.dynamic_shared_bytes_per_block = dynamic_shared_bytes;
  return result;
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
  int runtime = 0, driver = 0, cublas_version = 0;
  { Blas probe; BLAS_CHECK(cublasGetVersion(probe.handle, &cublas_version)); }
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
  int max_blocks_sm = 0, nominal_sm_clock_khz = 0, nominal_memory_clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_blocks_sm, cudaDevAttrMaxBlocksPerMultiprocessor, o.device));
  // CUDA 13 removed clockRate/memoryClockRate from cudaDeviceProp. The
  // attributes remain supported in CUDA 12/13 and preserve our JSON schema.
  CUDA_CHECK(cudaDeviceGetAttribute(&nominal_sm_clock_khz, cudaDevAttrClockRate, o.device));
  CUDA_CHECK(cudaDeviceGetAttribute(&nominal_memory_clock_khz, cudaDevAttrMemoryClockRate, o.device));
  std::cout << "{\"type\":\"device\",\"process_id\":" << getpid() << ",\"cuda_ordinal\":" << o.device
    << ",\"name\":" << quote(p.name) << ",\"uuid\":" << quote(uuid_string(p.uuid))
    << ",\"pci_bus_id\":" << quote(pci) << ",\"cc\":" << quote(std::to_string(p.major) + "." + std::to_string(p.minor))
    << ",\"compute_capability_major\":" << p.major << ",\"compute_capability_minor\":" << p.minor
    << ",\"sm_count\":" << p.multiProcessorCount << ",\"warp_size\":" << p.warpSize
    << ",\"l2_bytes\":" << p.l2CacheSize << ",\"total_memory_bytes\":" << p.totalGlobalMem
    << ",\"max_threads_per_sm\":" << p.maxThreadsPerMultiProcessor
    << ",\"max_blocks_per_sm\":" << max_blocks_sm << ",\"registers_per_sm\":" << p.regsPerMultiprocessor
    << ",\"shared_memory_per_sm_bytes\":" << p.sharedMemPerMultiprocessor
    << ",\"nominal_max_sm_clock_khz\":" << nominal_sm_clock_khz << ",\"nominal_max_memory_clock_khz\":" << nominal_memory_clock_khz
    << ",\"memory_bus_width_bits\":" << p.memoryBusWidth
    << ",\"cuda_runtime_version\":" << runtime << ",\"cuda_driver_version\":" << driver
    << ",\"cuda_compile_version\":" << CUDART_VERSION << ",\"cublas_version\":" << cublas_version << "}" << std::endl;
}

struct WorkEpoch {
  uint64_t start_ns = 0, end_ns = 0, batches = 0;
  uint64_t admitted_blocks = 0, counter_readback_ns = 0;
};
struct RunTiming {
  double device_s = 0, host_s = 0;
  uint64_t batches = 0;
  std::vector<WorkEpoch> epochs;
};
template<class Launch> RunTiming run_phase(const char* name, double seconds,
    uint64_t fixed_batches, Launch launch,
    const std::function<uint64_t()>& snapshot_admissions = {}, bool profile_region = false) {
  CUDA_CHECK(cudaDeviceSynchronize());
  phase(name, "start");
  Event start, stop, batch_done;
  // The profiler region includes every actual measure kernel, including all
  // internal cuBLAS kernels, and excludes initialization/warmup. Nsight must
  // launch with --profile-from-start off. Counter replay is a separate run.
  if (profile_region) CUDA_CHECK(cudaProfilerStart());
  uint64_t begin = monotonic_ns();
  CUDA_CHECK(cudaEventRecord(start.event));
  uint64_t batches = 0, epoch_begin = begin, epoch_batches = 0, previous_admissions = 0;
  RunTiming result;
  auto capture_epoch = [&] {
    uint64_t copy_begin = monotonic_ns();
    uint64_t cumulative = snapshot_admissions ? snapshot_admissions() : 0;
    uint64_t copy_end = monotonic_ns();
    // Windows include counter readback gaps, making their work and host power
    // denominators agree. No stdout is emitted inside the measured window.
    result.epochs.push_back({epoch_begin, copy_end, batches - epoch_batches,
      cumulative - previous_admissions, copy_end - copy_begin});
    epoch_begin = copy_end; epoch_batches = batches; previous_admissions = cumulative;
  };
  // Synchronize once per batch. --iterations makes a batch long enough to avoid
  // host launch overhead dominating; report measured batch count and duration.
  while (fixed_batches ? batches < fixed_batches : (batches == 0 ? seconds > 0 : double(monotonic_ns() - begin) * 1e-9 < seconds)) {
    launch(); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(batch_done.event));
    CUDA_CHECK(cudaEventSynchronize(batch_done.event));
    ++batches;
    if (snapshot_admissions && monotonic_ns() - epoch_begin >= 1000000000ULL) capture_epoch();
  }
  CUDA_CHECK(cudaEventRecord(stop.event));
  CUDA_CHECK(cudaEventSynchronize(stop.event));
  if (snapshot_admissions && batches > epoch_batches) capture_epoch();
  uint64_t end = monotonic_ns();
  if (profile_region) CUDA_CHECK(cudaProfilerStop());
  float elapsed_ms = 0; CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start.event, stop.event));
  phase(name, "end");
  result.device_s = double(elapsed_ms) * 1e-3;
  result.host_s = double(end - begin) * 1e-9; result.batches = batches;
  return result;
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
  const bool nonlinear = nonlinear_workload(o.workload);
  const bool rowwise = o.workload == "rmsnorm" || o.workload == "softmax";
  if (!o.iterations) o.iterations = o.workload == "gemm" || nonlinear ? 1 : 1024;
  if (!o.batch_launches) o.batch_launches = o.workload == "gemm" ? 1 : 16;
  if (!o.working_set_bytes) {
    if (nonlinear) o.working_set_bytes = uint64_t(o.blocks) * (rowwise ? o.row_width : o.threads * 128) * 4;
    else if (o.workload == "l1") o.working_set_bytes = uint64_t(o.blocks) * 16 * 1024;
    else if (o.workload == "l2" || latency) o.working_set_bytes = std::max(uint64_t(4), uint64_t(p.l2CacheSize) / 2 / 4 * 4);
    else o.working_set_bytes = std::max(uint64_t(512) * 1024 * 1024, uint64_t(p.l2CacheSize) * 8);
  }
  const uint64_t requested_working_set_bytes = o.working_set_bytes;
  const uint64_t lanes = uint64_t(o.blocks) * o.threads;
  const bool paired_context = o.paired_reference && o.workload != "control";
  const bool paired_reference = paired_context && !o.profile_region;
  uint64_t words = memory || nonlinear ? o.working_set_bytes / 4 : 1;
  if (nonlinear && (!words || words % (uint64_t(o.blocks) * (rowwise ? o.row_width : 1))))
    throw std::runtime_error("nonlinear footprint must contain equal complete block slices/rows");
  if (nonlinear && ((long double)words * 8 + uint64_t(o.row_width) * 4 + lanes * 8 + 262144 > (long double)p.totalGlobalMem * 0.7))
    throw std::runtime_error("nonlinear input/output/gamma and reference allocations exceed 70% of device memory");
  if (memory && o.access != "read") {
    // x=(tid+k*lanes)*stride modulo words. A whole lanes*stride tile
    // makes x/stride modulo words/stride preserve tid modulo lanes.
    // Consequently different threads never race on the same output word.
    if (lanes > std::numeric_limits<uint64_t>::max() / o.stride_elements)
      throw std::runtime_error("write/copy ownership tile overflows");
    const uint64_t tile_words = lanes * o.stride_elements;
    words = words / tile_words * tile_words;
    if (!words) throw std::runtime_error("write/copy footprint must fit at least blocks*threads*stride 32-bit words");
  }
  uint64_t slice = o.workload == "l1" ? words / o.blocks : 0;
  if (memory && words == 0) throw std::runtime_error("memory footprint must contain at least one word");
  if (latency && words > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("pointer chain exceeds uint32 node-index range");
  if (o.workload == "l1" && !slice) throw std::runtime_error("L1 footprint must contain at least one word per block");
  if (o.workload == "l1") words = slice * o.blocks;
  uint64_t offset_words = memory ? o.offset_bytes / 4 : 0;
  if (words > std::numeric_limits<uint64_t>::max() - offset_words) throw std::runtime_error("allocation size overflow");
  if (memory && words + offset_words > uint64_t(p.totalGlobalMem) / 4 / (o.access == "read" ? 1 : 2) * 3 / 4)
    throw std::runtime_error("requested buffers exceed 75% of device capacity");
  uint64_t tensor_out_count = o.workload == "tensor" ? lanes / 32 * o.accumulators * 256 : 1;
  uint64_t gemm_a_count = o.workload == "gemm" ? uint64_t(o.gemm_m) * o.gemm_k : 256;
  uint64_t gemm_b_count = o.workload == "gemm" ? uint64_t(o.gemm_k) * o.gemm_n : 256;
  uint64_t gemm_c_count = o.workload == "gemm" ? uint64_t(o.gemm_m) * o.gemm_n : tensor_out_count;
  Buffer<uint32_t> input(checked_size(words + offset_words, sizeof(uint32_t), "input"));
  Buffer<uint32_t> output(checked_size((memory && o.access != "read") || nonlinear ? words + offset_words : 1, sizeof(uint32_t), "output"));
  const uint64_t gamma_count = o.workload == "rmsnorm" ? o.row_width : 1;
  Buffer<uint32_t> gamma(checked_size(gamma_count, sizeof(uint32_t), "gamma"));
  Buffer<uint32_t> sink(checked_size(lanes, sizeof(uint32_t), "sink"));
  Buffer<__half> a(checked_size(gemm_a_count, sizeof(__half), "GEMM A"));
  Buffer<__half> b(checked_size(gemm_b_count, sizeof(__half), "GEMM B"));
  Buffer<float> c(checked_size(gemm_c_count, sizeof(float), "GEMM C"));
  Buffer<unsigned char> mask(kSmSlots);
  Buffer<unsigned long long> sm_blocks(kSmSlots);
  Buffer<unsigned long long> sm_cycles(kSmSlots);
  Buffer<unsigned long long> sm_loads(kSmSlots);
  // Separate reference sinks/counters preserve treatment outputs and admission
  // counts even when the randomized order runs the reference second. Both
  // arms retain exactly the same allocated context and buffers.
  Buffer<uint32_t> reference_sink(checked_size(paired_context ? lanes : 1, sizeof(uint32_t), "reference sink"));
  Buffer<unsigned long long> reference_sm_blocks(paired_context ? kSmSlots : 1);
  std::vector<unsigned char> mask_host(kSmSlots, o.sm_ids.empty() ? 1 : 0);
  for (unsigned id : o.sm_ids) mask_host[id] = 1;
  CUDA_CHECK(cudaMemcpy(mask.ptr, mask_host.data(), kSmSlots, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(sink.ptr, 0, size_t(lanes) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(c.ptr, 0, size_t(gemm_c_count) * sizeof(float)));
  CUDA_CHECK(cudaMemset(sm_blocks.ptr, 0, kSmSlots * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(sm_cycles.ptr, 0, kSmSlots * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(sm_loads.ptr, 0, kSmSlots * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(reference_sink.ptr, 0, size_t(paired_context ? lanes : 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(reference_sm_blocks.ptr, 0, size_t(paired_context ? kSmSlots : 1) * sizeof(unsigned long long)));
  init_words<<<std::min(o.blocks, 4096), 256>>>(input.ptr, words + offset_words, o.seed);
  if (nonlinear) {
    // Large-offset logits exercise max-subtracted Softmax without overflow.
    init_nonlinear<<<std::min(o.blocks, 4096), 256>>>(input.ptr, words, o.seed, o.workload == "softmax" ? 100.0f : 0.0f, false);
    init_nonlinear<<<std::min(o.blocks, 4096), 256>>>(gamma.ptr, gamma_count, o.seed ^ 12345ULL, 0.0f, true);
    CUDA_CHECK(cudaMemset(output.ptr, 0, size_t(words) * sizeof(uint32_t)));
  }
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
  KernelResources resources;
  if (o.workload == "tensor") {
    #define TENSOR_RESOURCE_CASE(N) case N: resources = query_kernel_resources(tensor_kernel<N>, o.threads); break
    switch (o.accumulators) {
      TENSOR_RESOURCE_CASE(1); TENSOR_RESOURCE_CASE(2); TENSOR_RESOURCE_CASE(3); TENSOR_RESOURCE_CASE(4);
      TENSOR_RESOURCE_CASE(5); TENSOR_RESOURCE_CASE(6); TENSOR_RESOURCE_CASE(7); TENSOR_RESOURCE_CASE(8);
    }
    #undef TENSOR_RESOURCE_CASE
  } else if (o.workload == "l1") resources = query_kernel_resources(memory_kernel<true, 0>, o.threads);
  else if (o.workload == "l2" || o.workload == "hbm") {
    if (o.access == "read") resources = query_kernel_resources(memory_kernel<false, 0>, o.threads);
    else if (o.access == "write") resources = query_kernel_resources(memory_kernel<false, 1>, o.threads);
    else resources = query_kernel_resources(memory_kernel<false, 2>, o.threads);
  } else if (latency) resources = query_kernel_resources(latency_kernel, o.threads);
  else if (o.workload == "control") resources = query_kernel_resources(control_kernel, o.threads);
  else if (o.workload == "exp") resources = query_kernel_resources(pointwise_nonlinear_kernel<0>, o.threads);
  else if (o.workload == "tanh") resources = query_kernel_resources(pointwise_nonlinear_kernel<1>, o.threads);
  else if (o.workload == "silu") resources = query_kernel_resources(pointwise_nonlinear_kernel<2>, o.threads);
  else if (o.workload == "softmax") resources = query_kernel_resources(row_nonlinear_kernel<true>, o.threads, o.threads * sizeof(float));
  else if (o.workload == "rmsnorm") resources = query_kernel_resources(row_nonlinear_kernel<false>, o.threads, o.threads * sizeof(float));
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
    } else if (nonlinear) {
      const uint64_t block_words = words / o.blocks;
      if (o.workload == "exp") pointwise_nonlinear_kernel<0><<<o.blocks, o.threads>>>(input.ptr, output.ptr, block_words, o.iterations, mask.ptr, sm_blocks.ptr);
      else if (o.workload == "tanh") pointwise_nonlinear_kernel<1><<<o.blocks, o.threads>>>(input.ptr, output.ptr, block_words, o.iterations, mask.ptr, sm_blocks.ptr);
      else if (o.workload == "silu") pointwise_nonlinear_kernel<2><<<o.blocks, o.threads>>>(input.ptr, output.ptr, block_words, o.iterations, mask.ptr, sm_blocks.ptr);
      else if (o.workload == "softmax") row_nonlinear_kernel<true><<<o.blocks, o.threads, o.threads * sizeof(float)>>>(input.ptr, gamma.ptr, output.ptr, block_words / o.row_width, o.row_width, o.iterations, mask.ptr, sm_blocks.ptr);
      else row_nonlinear_kernel<false><<<o.blocks, o.threads, o.threads * sizeof(float)>>>(input.ptr, gamma.ptr, output.ptr, block_words / o.row_width, o.row_width, o.iterations, mask.ptr, sm_blocks.ptr);
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
  NonlinearCheck precheck;
  if (nonlinear) {
    single_launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    precheck = check_nonlinear(o, input.ptr, output.ptr, gamma.ptr, words);
    if (!precheck.passed) throw std::runtime_error("nonlinear output differs from CPU double reference before measurement");
  }
  std::cout << "{\"type\":\"treatment_protocol\",\"kind\":"
    << quote(paired_reference ? "paired_active_reference" : "powered_idle_bracket")
    << ",\"order\":" << (paired_reference ? quote(o.reference_order) : "null")
    << ",\"phase_order\":" << (paired_reference ? (o.reference_order == "AB" ? "[\"active_reference\",\"measure\"]" : "[\"measure\",\"active_reference\"]") : "[\"measure\"]")
    << ",\"same_process\":true,\"same_allocations\":true,\"same_clock_policy\":true"
    << ",\"paired_reference_context_allocated\":" << (paired_context ? "true" : "false")
    << ",\"launch_geometry_matched\":" << (paired_reference && o.workload != "gemm" ? "true" : "false")
    << ",\"reference_workload\":" << (paired_reference ? "\"control\"" : "null")
    << ",\"reference_kind\":" << (paired_reference ? "\"issue_loop\"" : "null")
    << ",\"reference_iterations\":" << (paired_reference ? o.iterations : 0)
    << ",\"arm_seconds\":" << o.seconds << ",\"arm_warmup_seconds\":" << o.warmup_seconds
    << ",\"reference_scope\":\"paired integer issue-loop operational contrast; launch geometry, iterations, SM admission mask and batching match custom kernels; instruction mix, residency, register pressure, occupancy and duration per launch are not counterfactual equivalents; not isolated component energy\"}"
    << std::endl;
  run_phase("warmup", o.warmup_seconds, o.warmup_batches, launch);
  idle_phase("idle_pre", o.idle_seconds);
  std::vector<unsigned long long> epoch_observed(kSmSlots), reference_observed(kSmSlots);
  auto snapshot_admissions = [&]() -> uint64_t {
    if (o.workload == "gemm") return 0;
    CUDA_CHECK(cudaMemcpy(epoch_observed.data(), sm_blocks.ptr, kSmSlots * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return std::accumulate(epoch_observed.begin(), epoch_observed.end(), uint64_t(0));
  };
  auto snapshot_reference_admissions = [&]() -> uint64_t {
    CUDA_CHECK(cudaMemcpy(reference_observed.data(), reference_sm_blocks.ptr, kSmSlots * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return std::accumulate(reference_observed.begin(), reference_observed.end(), uint64_t(0));
  };
  RunTiming timing, reference_timing;
  auto treatment_arm = [&]() {
    // Each crossover arm prepares its own target state after the prior arm.
    // Warmup is outside every energy/counter measurement window.
    if (paired_reference) run_phase("warmup_treatment", o.warmup_seconds, o.warmup_batches, launch);
    CUDA_CHECK(cudaMemset(sm_blocks.ptr, 0, kSmSlots * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(sm_cycles.ptr, 0, kSmSlots * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(sm_loads.ptr, 0, kSmSlots * sizeof(unsigned long long)));
    // A timed warmup must not shift the first measured pointer-chain nonce.
    launch_nonce = o.seed;
    timing = run_phase("measure", o.seconds, o.fixed_batches, launch, snapshot_admissions, o.profile_region);
  };
  auto reference_launch = [&]() {
    for (int i = 0; i < o.batch_launches; ++i)
      control_kernel<<<o.blocks, o.threads>>>(reference_sink.ptr, o.iterations, mask.ptr, filtered, reference_sm_blocks.ptr);
  };
  auto reference_arm = [&]() {
    run_phase("warmup_reference", o.warmup_seconds, o.warmup_batches, reference_launch);
    CUDA_CHECK(cudaMemset(reference_sm_blocks.ptr, 0, kSmSlots * sizeof(unsigned long long)));
    reference_timing = run_phase("active_reference", o.seconds, o.fixed_batches, reference_launch, snapshot_reference_admissions);
  };
  if (paired_reference && o.reference_order == "AB") reference_arm();
  treatment_arm();
  if (paired_reference && o.reference_order == "BA") reference_arm();
  idle_phase("idle_post", o.idle_seconds);
  if (paired_reference) {
    CUDA_CHECK(cudaMemcpy(reference_observed.data(), reference_sm_blocks.ptr, kSmSlots * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    uint64_t reference_admitted = std::accumulate(reference_observed.begin(), reference_observed.end(), uint64_t(0));
    if (!reference_admitted) throw std::runtime_error("no active-reference blocks were admitted by the SM filter");
    std::vector<unsigned> missing_reference;
    for (unsigned id : o.sm_ids) if (!reference_observed[id]) missing_reference.push_back(id);
    std::cout << std::setprecision(17)
      << "{\"type\":\"active_reference_result\",\"workload\":\"control\",\"reference_kind\":\"issue_loop\""
      << ",\"duration_s\":" << reference_timing.device_s << ",\"host_duration_s\":" << reference_timing.host_s
      << ",\"batches\":" << reference_timing.batches << ",\"batch_launches\":" << o.batch_launches
      << ",\"iterations_per_launch\":" << o.iterations << ",\"blocks\":" << o.blocks << ",\"threads\":" << o.threads
      << ",\"admitted_blocks\":" << reference_admitted << ",\"operations\":0,\"logical_bytes\":0"
      << ",\"operation_unit\":\"not FLOPs or target bytes; integer issue-loop reference\",\"measure_epochs\":[";
    for (size_t i = 0; i < reference_timing.epochs.size(); ++i) {
      const WorkEpoch& epoch = reference_timing.epochs[i];
      std::cout << (i ? "," : "") << "{\"host_monotonic_start_ns\":" << epoch.start_ns
        << ",\"host_monotonic_end_ns\":" << epoch.end_ns
        << ",\"start_s\":" << double(epoch.start_ns) * 1e-9 << ",\"end_s\":" << double(epoch.end_ns) * 1e-9
        << ",\"batches\":" << epoch.batches << ",\"admitted_blocks\":" << epoch.admitted_blocks
        << ",\"kernel_launches\":" << epoch.batches * o.batch_launches
        << ",\"operations\":0,\"logical_bytes\":0,\"counts_exact\":true,\"counter_readback_ns\":" << epoch.counter_readback_ns << "}";
    }
    std::cout << "],\"sanity\":{\"requested_sm_coverage_complete\":" << (missing_reference.empty() ? "true" : "false")
      << "},\"scope\":\"active integer issue-loop reference measured in same CUDA process and allocated context; no target-operation normalization and no physical component attribution\"}" << std::endl;
    if (!missing_reference.empty()) throw std::runtime_error("requested active-reference SM coverage incomplete; reject this trial");
  }
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
  else if (nonlinear) {
    operations = (long double)admitted_blocks * (words / o.blocks) * o.iterations;
    logical_bytes = operations * (rowwise ? 16 : 8);
  }
  uint64_t checksum = 0; bool finite = true;
  NonlinearCheck numerical;
  const char* checksum_kind = "uint32_sink_sample";
  if (nonlinear) {
    numerical = check_nonlinear(o, input.ptr, output.ptr, gamma.ptr, words);
    numerical.checked_values += precheck.checked_values;
    numerical.passed = numerical.passed && precheck.passed;
    numerical.max_absolute_error = std::max(numerical.max_absolute_error, precheck.max_absolute_error);
    numerical.max_relative_error = std::max(numerical.max_relative_error, precheck.max_relative_error);
    finite = numerical.passed; checksum = numerical.checksum;
    checksum_kind = "validated_fp32_nonlinear_output_sample_hash";
  } else if (o.workload == "tensor" || o.workload == "gemm") {
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
  uint64_t allocated_bytes = (words + offset_words + ((memory && o.access != "read") || nonlinear ? words + offset_words : 1) + lanes + gamma_count) * 4
    + (gemm_a_count + gemm_b_count) * 2 + gemm_c_count * 4 + kSmSlots * 25
    + (paired_context ? lanes * 4 + kSmSlots * sizeof(unsigned long long) : 12);
  uint64_t reachable_words = memory ? (latency ? words : (o.workload == "l1" ? slice / std::gcd(slice, o.stride_elements) * o.blocks : words / std::gcd(words, o.stride_elements))) : 0;
  // Sector footprint is a full-period bound, not observed traffic. For total
  // word counts divisible by 8 and offset aligned to 32B it is exact for the
  // strided address set; otherwise use a conservative upper bound.
  uint64_t sector_bound = memory ? std::min((words + (o.offset_bytes % 32) / 4 + 7) / 8 * 32, reachable_words * 32) : 0;
  bool sector_exact = memory && words % 8 == 0 && o.offset_bytes % 32 == 0 && o.workload != "l1";
  if (sector_exact) sector_bound = latency ? words * 4 : (std::gcd(words, o.stride_elements) <= 8 ? words * 4 : reachable_words * 32);
  // Finite work matters because every non-latency memory launch restarts at
  // the same addresses. An unfiltered launch visits q*stride mod n for a
  // contiguous q interval of length lanes*4*iterations. Full-period capacity
  // alone must never qualify an HBM trial.
  const uint64_t region_words = o.workload == "l1" ? slice : words;
  const uint64_t region_lanes = o.workload == "l1" ? uint64_t(o.threads) : lanes;
  uint64_t finite_words = 0;
  bool finite_words_exact = memory && !filtered && !latency;
  if (memory) {
    if (latency) finite_words = std::min(words, uint64_t(std::min((long double)words, (long double)admitted_blocks * o.iterations)));
    else finite_words = std::min(region_words / std::gcd(region_words, o.stride_elements),
      region_lanes * 4 * o.iterations) * (o.workload == "l1" ? o.blocks : 1);
  }
  uint64_t finite_sector_bound = std::min(sector_bound, finite_words * 32);
  bool finite_sector_exact = finite_words_exact && o.workload != "l1" && o.stride_elements == 1;
  if (finite_sector_exact) finite_sector_bound = (finite_words + (o.offset_bytes % 32) / 4 + 7) / 8 * 32;
  else if (finite_words_exact && finite_words == reachable_words && sector_exact) finite_sector_exact = true;
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
    << ",\"operations\":" << operations << ",\"operation_unit\":" << quote(nonlinear ? "output elements of the complete function; not FLOPs or SFU instructions" : memory ? "32-bit memory accesses (copy pairs count once)" : o.workload == "control" ? "not attributed" : "dense FP FLOPs (FMA=2)")
    << ",\"logical_bytes\":" << logical_bytes
    << ",\"throughput_ops_s\":" << operations / timing.device_s
    << ",\"throughput_bytes_s\":" << logical_bytes / timing.device_s
    << ",\"blocks\":" << o.blocks << ",\"threads\":" << o.threads
    << ",\"launch_geometry_applies\":" << (o.workload == "gemm" ? "false" : "true")
    << ",\"admitted_blocks\":" << admitted_blocks
    << ",\"working_set_bytes\":" << (memory || nonlinear ? words * 4 : 0)
    << ",\"requested_working_set_bytes\":" << (memory || nonlinear ? requested_working_set_bytes : 0)
    << ",\"write_copy_thread_addresses_disjoint\":" << (memory && o.access != "read" ? "true" : "null")
    << ",\"allocated_bytes\":" << allocated_bytes << ",\"l1_bytes_per_block\":" << slice * 4
    << ",\"logical_reachable_bytes\":" << reachable_words * 4
    << ",\"potential_cache_sector_bytes\":" << sector_bound
    << ",\"potential_sector_footprint_exact\":" << (sector_exact ? "true" : "false")
    << ",\"reachable_footprint_scope\":\"potential over complete stride/chain cycle; finite batches and SM admission may touch less; sector count is a bound when unaligned\""
    << ",\"finite_launch_reachable_bytes_upper_bound\":" << finite_words * 4
    << ",\"finite_launch_reachable_bytes_exact\":" << (finite_words_exact ? "true" : "false")
    << ",\"finite_launch_sector_bytes_upper_bound\":" << finite_sector_bound
    << ",\"finite_launch_sector_bytes_exact\":" << (finite_sector_exact ? "true" : "false")
    << ",\"memory_address_sequence_restarts_each_launch\":" << (memory && !latency ? "true" : "false")
    << ",\"finite_footprint_scope\":\"one unfiltered memory launch; filtered values are bounds over entire launched grid; latency value bounds all measured dependent probes\""
    << ",\"stride_elements\":" << o.stride_elements << ",\"offset_bytes\":" << o.offset_bytes
    << ",\"tensor_accumulators\":" << o.accumulators
    << ",\"gemm_m\":" << o.gemm_m << ",\"gemm_n\":" << o.gemm_n << ",\"gemm_k\":" << o.gemm_k
    << ",\"seed\":" << o.seed << ",\"checksum\":" << checksum << ",\"checksum_kind\":" << quote(checksum_kind)
    << ",\"kernel_resources\":";
  if (!resources.available) std::cout << "null";
  else std::cout << "{\"registers_per_thread\":" << resources.attributes.numRegs
    << ",\"local_bytes_per_thread\":" << resources.attributes.localSizeBytes
    << ",\"static_shared_bytes_per_block\":" << resources.attributes.sharedSizeBytes
    << ",\"dynamic_shared_bytes_per_block\":" << resources.dynamic_shared_bytes_per_block
    << ",\"max_active_blocks_per_sm\":" << resources.max_active_blocks_per_sm
    << ",\"max_active_warps_per_sm\":" << resources.max_active_blocks_per_sm * (o.threads / p.warpSize)
    << ",\"occupancy_upper_bound_fraction\":" << double(resources.max_active_blocks_per_sm * o.threads) / p.maxThreadsPerMultiProcessor
    << ",\"grid_average_blocks_per_sm\":" << double(o.blocks) / p.multiProcessorCount
    << ",\"l1_slice_bytes_per_sm_at_occupancy_bound\":" << (o.workload == "l1" ? uint64_t(resources.max_active_blocks_per_sm) * slice * 4 : 0)
    << ",\"scope\":\"CUDA compiled resources and theoretical residency limit; grid-average and L1 slice capacity are diagnostic bounds, not actual concurrent scheduling, occupancy, cache hit rate or Tensor utilization\"}";
  std::cout
    << ",\"row_width\":" << (rowwise ? o.row_width : 0)
    << ",\"elements\":" << (nonlinear ? operations : 0)
    << ",\"row_evaluations\":" << (rowwise ? operations / o.row_width : 0)
    << ",\"math_implementation\":" << (nonlinear ? "\"cuda_fp32_standard_streaming_v1\"" : "null")
    << ",\"rms_epsilon\":" << (o.workload == "rmsnorm" ? "0.00001" : "null")
    << ",\"affine_gamma\":" << (o.workload == "rmsnorm" ? "true" : "false")
    << ",\"nonlinear_input_distribution\":" << (nonlinear ? quote(o.workload == "softmax" ? "deterministic uniform FP32 [96,104); stable max subtraction" : "deterministic uniform FP32 [-4,4)") : "null")
    << ",\"nonlinear_scope\":" << (nonlinear ? "\"complete FP32 function including global input/output and row reductions; repeated applications reload fixed input; no pure SFU or FLOP attribution\"" : "null")
    << ",\"numerical_validation\":{\"checked_values\":" << numerical.checked_values
    << ",\"max_absolute_error\":" << numerical.max_absolute_error << ",\"max_relative_error\":" << numerical.max_relative_error
    << ",\"absolute_tolerance\":0.000002,\"relative_tolerance\":0.0002,\"scope\":\"CPU double reference on distributed block slices/complete rows before warmup and after measurement; Softmax row sums also checked; not exhaustive\"}"
    << ",\"sanity\":{\"finite_output_sample\":" << (finite ? "true" : "false")
    << ",\"numerical_validation_passed\":" << (nonlinear ? (numerical.passed ? "true" : "false") : "null")
    << ",\"requested_sm_coverage_complete\":" << (missing.empty() ? "true" : "false") << "}"
    << ",\"profile_region\":" << (o.profile_region ? "true" : "false")
    << ",\"paired_reference_context_allocated\":" << (paired_context ? "true" : "false")
    << ",\"measure_epochs\":[";
  for (size_t i = 0; i < timing.epochs.size(); ++i) {
    const WorkEpoch& epoch = timing.epochs[i];
    long double epoch_operations = 0, epoch_bytes = 0;
    if (o.workload == "tensor") epoch_operations = (long double)epoch.admitted_blocks * (o.threads / 32) * o.iterations * o.accumulators * 8192;
    else if (o.workload == "gemm") epoch_operations = (long double)epoch.batches * o.batch_launches * o.iterations * 2 * o.gemm_m * o.gemm_n * o.gemm_k;
    else if (latency) { epoch_operations = (long double)epoch.admitted_blocks * o.iterations; epoch_bytes = epoch_operations * 4; }
    else if (memory) { epoch_operations = (long double)epoch.admitted_blocks * o.threads * o.iterations * 4; epoch_bytes = epoch_operations * 4 * (o.access == "copy" ? 2 : 1); }
    else if (nonlinear) { epoch_operations = (long double)epoch.admitted_blocks * (words / o.blocks) * o.iterations; epoch_bytes = epoch_operations * (rowwise ? 16 : 8); }
    std::cout << (i ? "," : "") << "{\"host_monotonic_start_ns\":" << epoch.start_ns
      << ",\"host_monotonic_end_ns\":" << epoch.end_ns
      << ",\"start_s\":" << double(epoch.start_ns) * 1e-9 << ",\"end_s\":" << double(epoch.end_ns) * 1e-9
      << ",\"batches\":" << epoch.batches << ",\"admitted_blocks\":" << epoch.admitted_blocks
      << ",\"kernel_launches\":" << (o.workload == "gemm" ? "null" : std::to_string(epoch.batches * o.batch_launches))
      << ",\"gemm_invocations\":" << (o.workload == "gemm" ? epoch.batches * o.batch_launches * o.iterations : 0)
      << ",\"operations\":" << epoch_operations << ",\"logical_bytes\":" << epoch_bytes
      << ",\"elements\":" << (nonlinear ? epoch_operations : 0) << ",\"row_evaluations\":" << (rowwise ? epoch_operations / o.row_width : 0)
      << ",\"counts_exact\":true,\"counter_readback_ns\":" << epoch.counter_readback_ns << "}";
  }
  const long double output_payload_bytes = o.workload == "tensor" ?
    (long double)admitted_blocks * (o.threads / 32) * o.accumulators * 256 * 4 :
    o.workload == "gemm" ? (long double)timing.batches * o.batch_launches * o.iterations * gemm_c_count * 4 :
    nonlinear ? 0 : latency ? (long double)admitted_blocks * 4 : (long double)admitted_blocks * o.threads * 4;
  std::cout << "],\"auxiliary_work\":{\"logical_result_output_bytes\":" << output_payload_bytes
    << ",\"logical_tensor_operand_load_bytes\":" << (o.workload == "tensor" ? (long double)admitted_blocks * (o.threads / 32) * 1024 : 0)
    << ",\"sm_admission_atomic_updates\":" << admitted_blocks
    << ",\"latency_telemetry_atomic_updates\":" << (latency ? (long double)admitted_blocks * 2 : 0)
    << ",\"host_counter_readback_bytes\":" << (o.workload == "gemm" ? 0 : timing.epochs.size() * kSmSlots * sizeof(unsigned long long))
    << ",\"scope\":\"issued auxiliary payload, not physical cache or DRAM bytes; excludes possible compiler spills; GEMM output payload is algorithmic minimum, internal kernels may differ\"}"
    << ",\"epoch_scope\":\"approximately one second, exact completed batch counts; host windows include measured admission-counter readback overhead; no interpolated partial-batch counts\""
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
  std::cout << "}},\"cache_policy\":" << quote(o.workload == "l1" ? "ld.global.ca; prefer maximum L1 carveout" : memory || nonlinear ? "ld/st.global.cg; L1 bypass; L2/DRAM residency requires counters" : "not applicable")
    << ",\"memory_count_scope\":\"logical requested payload; excludes sector inflation, writeback, initialization, telemetry and sink transfers\""
    << ",\"sm_filter_scope\":\"best-effort dispatched-block admission; no physical SM disable or GPC mapping; GEMM SM IDs unavailable\""
    << ",\"control_scope\":\"integer issue-loop reference, not matched cache/tensor dynamic power or transistor static power\""
    << ",\"tensor_scope\":\"WMMA register-operand reuse with identical small matrices across warps; compare randomized cuBLAS GEMM for sustained dense peak\""
    << ",\"sparsity\":\"dense\",\"input_precision\":" << quote(nonlinear ? "fp32" : "fp16 for tensor/gemm; uint32 memory payload otherwise") << ",\"accumulator_precision\":\"fp32\"}" << std::endl;
  if (!finite) throw std::runtime_error("non-finite or numerically incorrect output sample");
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
