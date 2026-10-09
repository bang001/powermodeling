#pragma once
#include <cstdint>
#include <limits>

// Keep these helpers usable by the CPU boundary tests without a CUDA driver.
#ifdef __CUDACC__
#define POWERBENCH_READ_INLINE __host__ __device__ __forceinline__
#else
#define POWERBENCH_READ_INLINE inline
#endif

inline bool memory_read_uses_uint32(uint64_t region_words, uint64_t iterations) {
  return region_words <= std::numeric_limits<uint32_t>::max() &&
         iterations <= std::numeric_limits<uint32_t>::max();
}

POWERBENCH_READ_INLINE uint32_t memory_read_next32(uint32_t position,
                                                 uint32_t advance,
                                                 uint32_t boundary) {
  return position >= boundary ? position - boundary : position + advance;
}

#undef POWERBENCH_READ_INLINE
