// Complete FP32 nonlinear applications. Included inside gpu_bench's namespace
// after the volatile .cg load/store and SM-admission helpers.
constexpr float kRmsEpsilon = 1e-5f;

__global__ void init_nonlinear(uint32_t* input, uint64_t words, uint64_t seed,
                              float shift, bool gamma) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += uint64_t(gridDim.x) * blockDim.x) {
    uint32_t bits = random_word(i ^ seed) & 65535U;
    float x = gamma ? 0.75f + float(bits) / 131072.0f
                    : (float(int(bits) - 32768) / 8192.0f) + shift;
    input[i] = __float_as_uint(x);
  }
}

__device__ __forceinline__ float nonlinear_load(const uint32_t* p) {
  // Volatile inline loads prevent hoisting repeated identical input evaluations
  // out of --iterations; every counted application loads and stores its output.
  return __uint_as_float(load_cg(p));
}

template<int Function> // 0=EXP, 1=TANH, 2=SiLU; standard CUDA FP32 math.
__global__ void pointwise_nonlinear_kernel(const uint32_t* input, uint32_t* output,
    uint64_t slice_words, uint64_t iterations, const unsigned char* mask,
    unsigned long long* blocks_per_sm) {
  if (!admit(mask, false, blocks_per_sm)) return;
  uint64_t base = uint64_t(blockIdx.x) * slice_words;
  for (uint64_t repeat = 0; repeat < iterations; ++repeat) {
    for (uint64_t j = threadIdx.x; j < slice_words; j += blockDim.x) {
      float x = nonlinear_load(input + base + j);
      float y;
      if constexpr (Function == 0) y = expf(x);
      else if constexpr (Function == 1) y = tanhf(x);
      else y = x / (1.0f + expf(-x));
      store_cg(output + base + j, __float_as_uint(y));
    }
  }
}

template<bool Maximum>
__device__ float nonlinear_reduce(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  // Also handles non-power-of-two block sizes such as 96 threads.
  for (unsigned active = blockDim.x; active > 1; active = (active + 1) / 2) {
    unsigned half = (active + 1) / 2;
    if (threadIdx.x < active / 2) {
      float other = shared[threadIdx.x + half];
      shared[threadIdx.x] = Maximum ? fmaxf(shared[threadIdx.x], other)
                                  : shared[threadIdx.x] + other;
    }
    __syncthreads();
  }
  float result = shared[0];
  __syncthreads(); // Everyone consumes the result before the next reduction.
  return result;
}

template<bool Softmax>
__global__ void row_nonlinear_kernel(const uint32_t* input, const uint32_t* gamma,
    uint32_t* output, uint64_t rows_per_block, int width, uint64_t iterations,
    const unsigned char* mask, unsigned long long* blocks_per_sm) {
  if (!admit(mask, false, blocks_per_sm)) return;
  extern __shared__ float shared[];
  for (uint64_t repeat = 0; repeat < iterations; ++repeat) {
    for (uint64_t row = 0; row < rows_per_block; ++row) {
      uint64_t base = (uint64_t(blockIdx.x) * rows_per_block + row) * width;
      float maximum = 0.0f;
      if constexpr (Softmax) {
        float local = -CUDART_INF_F;
        for (int col = threadIdx.x; col < width; col += blockDim.x)
          local = fmaxf(local, nonlinear_load(input + base + col));
        maximum = nonlinear_reduce<true>(local, shared);
      }
      float local_sum = 0.0f;
      for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float x = nonlinear_load(input + base + col);
        local_sum += Softmax ? expf(x - maximum) : x * x;
      }
      float sum = nonlinear_reduce<false>(local_sum, shared);
      float scale = Softmax ? 1.0f / sum : rsqrtf(sum / float(width) + kRmsEpsilon);
      for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float x = nonlinear_load(input + base + col);
        float y = Softmax ? expf(x - maximum) * scale
                         : x * scale * nonlinear_load(gamma + col);
        store_cg(output + base + col, __float_as_uint(y));
      }
      __syncthreads();
    }
  }
}

struct NonlinearCheck {
  uint64_t checked_values = 0, checksum = 0;
  double max_absolute_error = 0, max_relative_error = 0;
  bool passed = true;
};

NonlinearCheck check_nonlinear(const Options& o, const uint32_t* input,
    const uint32_t* output, const uint32_t* gamma, uint64_t words) {
  NonlinearCheck result;
  const bool rowwise = o.workload == "rmsnorm" || o.workload == "softmax";
  const uint64_t slice = words / o.blocks;
  const int samples = std::min(o.blocks, 8);
  std::vector<float> weights(rowwise ? o.row_width : 1, 1.0f);
  if (o.workload == "rmsnorm")
    CUDA_CHECK(cudaMemcpy(weights.data(), gamma, weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
  auto compare = [&](float actual, double expected) {
    double error = std::abs(double(actual) - expected);
    double relative = error / std::max(std::abs(expected), 1e-30);
    result.max_absolute_error = std::max(result.max_absolute_error, error);
    result.max_relative_error = std::max(result.max_relative_error, relative);
    result.passed = result.passed && std::isfinite(actual) && std::isfinite(expected)
      && error <= 2e-6 + 2e-4 * std::abs(expected);
    uint32_t bits; std::memcpy(&bits, &actual, sizeof(bits));
    result.checksum = result.checksum * 1315423911ULL + bits;
    ++result.checked_values;
  };
  for (int sample = 0; sample < samples; ++sample) {
    uint64_t block = samples == 1 ? 0 : uint64_t(sample) * (o.blocks - 1) / (samples - 1);
    if (rowwise) {
      uint64_t rows = slice / o.row_width;
      for (uint64_t row : std::vector<uint64_t>{0, rows - 1}) {
        uint64_t base = block * slice + row * o.row_width;
        std::vector<float> x(o.row_width), y(o.row_width);
        CUDA_CHECK(cudaMemcpy(x.data(), input + base, x.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y.data(), output + base, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double maximum = *std::max_element(x.begin(), x.end()), sum = 0, ysum = 0;
        for (float value : x) sum += o.workload == "softmax" ? std::exp(double(value) - maximum) : double(value) * value;
        for (int col = 0; col < o.row_width; ++col) {
          double expected = o.workload == "softmax" ? std::exp(double(x[col]) - maximum) / sum
            : double(x[col]) / std::sqrt(sum / o.row_width + double(kRmsEpsilon)) * weights[col];
          compare(y[col], expected);
          ysum += y[col];
        }
        if (o.workload == "softmax") result.passed = result.passed && std::abs(ysum - 1.0) <= 2e-4;
        if (rows == 1) break;
      }
    } else {
      for (uint64_t offset : std::vector<uint64_t>{0, slice / 2, slice - 1}) {
        float x, y; uint64_t index = block * slice + offset;
        CUDA_CHECK(cudaMemcpy(&x, input + index, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&y, output + index, sizeof(float), cudaMemcpyDeviceToHost));
        compare(y, o.workload == "exp" ? std::exp(double(x)) : o.workload == "tanh" ? std::tanh(double(x)) : double(x) / (1.0 + std::exp(-double(x))));
      }
    }
  }
  return result;
}
