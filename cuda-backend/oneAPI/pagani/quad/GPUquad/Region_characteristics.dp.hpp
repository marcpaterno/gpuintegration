#ifndef REGION_CHARACTERISTICS_CUH
#define REGION_CHARACTERISTICS_CUH

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>

//helper routines
#include "oneAPI/pagani/quad/util/mem_util.dp.hpp"

template<size_t ndim>
class Region_characteristics{
    public:
    Region_characteristics(size_t num_regions){
        active_regions = cuda_malloc<double>(num_regions*ndim);  
        sub_dividing_dim = cuda_malloc<int>(num_regions*ndim);  
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
