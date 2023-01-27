#ifndef HYBRID_CUH
#define HYBRID_CUH

#include <iostream>

// #include "cuda/pagani/quad/GPUquad/Sub_regions.cuh"
#include "cuda/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "cuda/pagani/quad/GPUquad/Region_estimates.cuh"
#include "cuda/pagani/quad/GPUquad/Phases.cuh"

template <typename T, size_t ndim>
void
two_level_errorest_and_relerr_classify(
  Region_estimates<T, ndim>& current_iter_raw_estimates,
  const Region_estimates<T, ndim>& prev_iter_two_level_estimates,
  const Region_characteristics<ndim>& reg_classifiers,
  T epsrel,
  bool relerr_classification = true)
{

  size_t num_regions = current_iter_raw_estimates.size;
  size_t block_size = 64;
  size_t numBlocks =
    num_regions / block_size + ((num_regions % block_size) ? 1 : 0);
  bool forbid_relerr_classification = !relerr_classification;
  if (prev_iter_two_level_estimates.size == 0) {
    return;
  }

  T* new_two_level_errorestimates = quad::cuda_malloc<T>(num_regions);
  quad::RefineError<T><<<numBlocks, block_size>>>(
    current_iter_raw_estimates.integral_estimates,
    current_iter_raw_estimates.error_estimates,
    prev_iter_two_level_estimates.integral_estimates,
    prev_iter_two_level_estimates.error_estimates,
    new_two_level_errorestimates,
    reg_classifiers.active_regions,
    num_regions,
    epsrel,
    forbid_relerr_classification);

  cudaDeviceSynchronize();
  cudaFree(current_iter_raw_estimates.error_estimates);
  current_iter_raw_estimates.error_estimates = new_two_level_errorestimates;
}

template <typename T, size_t ndim>
void
computute_two_level_errorest(
  Region_estimates<T, ndim>& current_iter_raw_estimates,
  const Region_estimates<T, ndim>& prev_iter_two_level_estimates,
  Region_characteristics<ndim>& reg_classifiers,
  bool relerr_classification = true)
{

  size_t num_regions = current_iter_raw_estimates.size;
  T epsrel = 1.e-3;
  size_t block_size = 64;
  size_t numBlocks =
    num_regions / block_size + ((num_regions % block_size) ? 1 : 0);
  bool forbid_relerr_classification = !relerr_classification;
  if (prev_iter_two_level_estimates.size == 0) {
    return;
  }

  T* new_two_level_errorestimates = quad::cuda_malloc<T>(num_regions);
  quad::RefineError<T><<<numBlocks, block_size>>>(
    current_iter_raw_estimates.integral_estimates,
    current_iter_raw_estimates.error_estimates,
    prev_iter_two_level_estimates.integral_estimates,
    prev_iter_two_level_estimates.error_estimates,
    new_two_level_errorestimates,
    reg_classifiers.active_regions,
    num_regions,
    epsrel,
    forbid_relerr_classification);

  cudaDeviceSynchronize();
  cudaFree(current_iter_raw_estimates.error_estimates);
  current_iter_raw_estimates.error_estimates = new_two_level_errorestimates;
}
#endif