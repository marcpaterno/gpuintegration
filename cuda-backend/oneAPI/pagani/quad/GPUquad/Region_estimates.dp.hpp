#ifndef REGION_ESTIMATES_CUH
#define REGION_ESTIMATES_CUH

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/quad/util/mem_util.dp.hpp"

template<size_t ndim>
class Region_estimates{
    public:
    
    Region_estimates(){}
    
    Region_estimates(size_t num_regions){
        integral_estimates = cuda_malloc<double>(num_regions);  
        error_estimates = cuda_malloc<double>(num_regions);  
        size = num_regions;
    }

    void
    reallocate(size_t num_regions) {
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      sycl::free(integral_estimates, q_ct1);
        sycl::free(error_estimates, q_ct1);
        integral_estimates = cuda_malloc<double>(num_regions);  
        error_estimates = cuda_malloc<double>(num_regions);  
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
