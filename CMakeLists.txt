cmake_minimum_required(VERSION 3.22)
# CUDA 12 supports all three requested generations. CUDA 13 removed offline
# compilation for Volta; configure with CUDA 12 when building one portable binary.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
endif()
project(powermodeling LANGUAGES CXX CUDA)
find_package(CUDAToolkit 12 REQUIRED)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13 AND "70" IN_LIST CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CUDA 13 cannot build sm_70. Use CUDA 12 for the V100/A100/H100 binary, or set CMAKE_CUDA_ARCHITECTURES=80;90.")
endif()

add_executable(powerbench cuda/gpu_bench.cu)
target_compile_features(powerbench PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(powerbench PRIVATE CUDA::cudart CUDA::cublas)
target_compile_options(powerbench PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo;--ptxas-options=-v>")
set_target_properties(powerbench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
install(TARGETS powerbench RUNTIME DESTINATION bin)
