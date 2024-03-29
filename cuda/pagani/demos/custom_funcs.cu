#include <iostream>

#include "common/cuda/cudaMemoryUtil.h"
#include "common/cuda/thrust_utils.cuh"
#include "common/cuda/custom_functions.cuh"

template <typename T, size_t size>
T*
make_gpu_arr(std::array<T, size> arr)
{
  int* d_arr = quad::cuda_malloc_managed<int>(arr.size());
  quad::cuda_memcpy_to_device<int>(d_arr, arr.data(), arr.size());
  return d_arr;
}

int
main()
{

  constexpr size_t size = 512;
  std::array<int, size> arr;
  std::fill(arr.begin(), arr.end(), 3.9);

  using MinMax = std::pair<int, int>;
  int* scan_output = quad::cuda_malloc<int>(size);
  int* d_arr = make_gpu_arr<int, size>(arr);

  MinMax res = min_max<int>(d_arr, size);

  int reduction_res = custom_reduce_atomics<int>(d_arr, size);

  int dot_res = custom_inner_product_atomics<int, int>(d_arr, d_arr, size);

  exclusive_scan<int, true>(d_arr, size, scan_output);
  return 0;
}
