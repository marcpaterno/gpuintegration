#ifndef REGION_ESTIMATES_CUH
#define REGION_ESTIMATES_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include "dpct-exp/common/cuda/cudaMemoryUtil.h"

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
	/*dpct::device_ext& dev_ct1 = dpct::get_current_device();
	sycl::queue& q_ct1 = dev_ct1.default_queue();
    sycl::free(integral_estimates, q_ct1);
    sycl::free(error_estimates, q_ct1);
    integral_estimates = quad::cuda_malloc<T>(num_regions);
    error_estimates = quad::cuda_malloc<T>(num_regions);
    size = num_regions;*/
	auto q_ct1 = sycl::queue(sycl::gpu_selector());
    sycl::free(integral_estimates, q_ct1);
    sycl::free(error_estimates, q_ct1);
    integral_estimates = quad::cuda_malloc<double>(num_regions);
    error_estimates = quad::cuda_malloc<double>(num_regions);
    size = num_regions;
  }

  ~Region_estimates()
  {
	dpct::device_ext& dev_ct1 = dpct::get_current_device();
	sycl::queue& q_ct1 = dev_ct1.default_queue();
    sycl::free(integral_estimates, q_ct1);
    sycl::free(error_estimates, q_ct1);
  }

  T* integral_estimates = nullptr;
  T* error_estimates = nullptr;
  size_t size = 0;
};
#endif