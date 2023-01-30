#ifndef REGION_ESTIMATES_CUH
#define REGION_ESTIMATES_CUH

#include <CL/sycl.hpp>
#include <iostream>
#include "common/oneAPI/cudaMemoryUtil.h"

template<size_t ndim>
class Region_estimates{
    public:
    
    Region_estimates(){}
    
    Region_estimates(size_t num_regions){
      device_init(num_regions);
    }
	
	Region_estimates(const Region_estimates<ndim>& other){
		device_init(other.size);
		quad::cuda_memcpy_device_to_device<double>(integral_estimates, other.integral_estimates, other.size);
		quad::cuda_memcpy_device_to_device<double>(error_estimates, other.error_estimates, other.size);
	}
	
	void
	device_init(size_t num_regions){
		integral_estimates = quad::cuda_malloc<double>(num_regions);  
        error_estimates = quad::cuda_malloc<double>(num_regions);  
        size = num_regions;
	}

    void
    reallocate(size_t num_regions) {
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      sycl::free(integral_estimates, q_ct1);
        sycl::free(error_estimates, q_ct1);
        integral_estimates = quad::cuda_malloc<double>(num_regions);  
        error_estimates = quad::cuda_malloc<double>(num_regions);  
        size = num_regions;
    }

    ~Region_estimates() {
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      sycl::free(integral_estimates, q_ct1);
        sycl::free(error_estimates, q_ct1);
    }
    
    double* integral_estimates = nullptr;
    double* error_estimates = nullptr;
    size_t size = 0;
};


#endif
