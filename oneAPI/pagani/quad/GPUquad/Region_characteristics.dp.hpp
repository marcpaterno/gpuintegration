#ifndef REGION_CHARACTERISTICS_CUH
#define REGION_CHARACTERISTICS_CUH

#include <CL/sycl.hpp>
#include <iostream>
#include "common/oneAPI/cudaMemoryUtil.h"

template<size_t ndim>
class Region_characteristics{
    public:
    Region_characteristics(size_t num_regions){
        device_init(num_regions);
    }
	
	Region_characteristics(const Region_characteristics<ndim>& other){
		 device_init(other.size);
		quad::cuda_memcpy_device_to_device<double>(active_regions, other.active_regions, other.size);
		quad::cuda_memcpy_device_to_device<int>(sub_dividing_dim, other.sub_dividing_dim, other.size);
	}
	
	void
	device_init(size_t num_regions){
		 active_regions = quad::cuda_malloc<double>(num_regions);  
        sub_dividing_dim = quad::cuda_malloc<int>(num_regions);  
        size = num_regions;
	}

    ~Region_characteristics() {
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      sycl::free(active_regions, q_ct1);
        sycl::free(sub_dividing_dim, q_ct1);
    }
    
    size_t size = 0;
    double* active_regions = nullptr;
    int* sub_dividing_dim = nullptr;
};


#endif
