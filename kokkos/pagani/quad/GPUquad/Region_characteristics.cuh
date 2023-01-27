#ifndef REGION_CHARACTERISTICS_CUH
#define REGION_CHARACTERISTICS_CUH

#include <iostream>

// helper routines
#include "common/cuda/cudaMemoryUtil.h"

template <size_t ndim>
class Region_characteristics {
public:
  Region_characteristics(size_t num_regions)
  {
    active_regions = quad::cuda_malloc<int>(num_regions);
    sub_dividing_dim = quad::cuda_malloc<int>(num_regions);
    size = num_regions;
  }

  ~Region_characteristics()
  {}

  size_t size = 0;
  ViewVectorInt active_regions;
  ViewVectorInt sub_dividing_dim;
};

#endif