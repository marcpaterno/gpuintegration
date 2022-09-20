#ifndef ONEAPI_REGION_ESTIMATES_CUH
#define ONEAPI_REGION_ESTIMATES_CUH

#include <iostream>
#include <CL/sycl.hpp>


template<size_t ndim>
class Region_estimates{
 public:
  
  Region_estimates(sycl::queue&q){_q = &q;}
  Region_estimates(sycl::queue& q, size_t num_regions);
  ~Region_estimates();
  
  void reallocate(sycl::queue& q, size_t num_regions);

  sycl::queue* _q;
  double* integral_estimates = nullptr;
  double* error_estimates = nullptr;
  size_t size = 0;
};

template<size_t ndim>
Region_estimates<ndim>::Region_estimates(sycl::queue& q, size_t num_regions){
    integral_estimates = sycl::malloc_device<double>(num_regions, q);  
    error_estimates = sycl::malloc_device<double>(num_regions, q);  
    size = num_regions;
    _q = &q;
}

template<size_t ndim>
Region_estimates<ndim>::~Region_estimates(){
    sycl::free(integral_estimates, *_q);
    sycl::free(error_estimates, *_q);
}

template<size_t ndim>
void
Region_estimates<ndim>::reallocate(sycl::queue& q,size_t num_regions){
    sycl::free(integral_estimates, *_q);
    sycl::free(error_estimates, *_q);
    
    integral_estimates = sycl::malloc_device<double>(num_regions, *_q);  
    error_estimates = sycl::malloc_device<double>(num_regions, *_q);  
    size = num_regions;
}
#endif

