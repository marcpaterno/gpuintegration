#ifndef ONEAPI_REGION_CHARACTERISTICS_CUH
#define ONEAPI_REGION_CHARACTERISTICS_CUH

#include <iostream>

// helper routines
#include <CL/sycl.hpp>

template <size_t ndim>
class Region_characteristics {

public:
  Region_characteristics(sycl::queue& q, size_t num_regions);
  ~Region_characteristics();
  Region_characteristics() {}

  sycl::queue* _q;
  size_t size = 0;
  double* active_regions = nullptr;
  int* sub_dividing_dim = nullptr;
};

template <size_t ndim>
Region_characteristics<ndim>::Region_characteristics(sycl::queue& q,
                                                     size_t num_regions)
{
  active_regions = sycl::malloc_device<double>(num_regions, q);
  sub_dividing_dim = sycl::malloc_device<int>(num_regions, q);
  size = num_regions;
  _q = &q;
}

template <size_t ndim>
Region_characteristics<ndim>::~Region_characteristics()
{
  sycl::free(active_regions, *_q);
  sycl::free(sub_dividing_dim, *_q);
}

#endif
