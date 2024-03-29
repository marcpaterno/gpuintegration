#ifndef REGION_ESTIMATES_CUH
#define REGION_ESTIMATES_CUH

#include <iostream>
#include "common/cuda/cudaMemoryUtil.h"

template <typename T, size_t ndim>
class Region_estimates {
public:
  Region_estimates() {}

  Region_estimates(size_t num_regions)
  {
    integral_estimates = quad::cuda_malloc<T>(num_regions);
    error_estimates = quad::cuda_malloc<T>(num_regions);
    size = num_regions;
  }

  void
  reallocate(size_t num_regions)
  {
    cudaFree(integral_estimates);
    cudaFree(error_estimates);
    integral_estimates = quad::cuda_malloc<T>(num_regions);
    error_estimates = quad::cuda_malloc<T>(num_regions);
    size = num_regions;
  }

  ~Region_estimates()
  {
    cudaFree(integral_estimates);
    cudaFree(error_estimates);
  }

  T* integral_estimates = nullptr;
  T* error_estimates = nullptr;
  size_t size = 0;
};
#endif