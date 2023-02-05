#ifndef KOKKOS_REGION_CHARACTERISTICS_CUH
#define KOKKOS_REGION_CHARACTERISTICS_CUH

#include <iostream>

// helper routines
#include "common/kokkos/cudaMemoryUtil.h"

template <size_t ndim>
class Region_characteristics {
public:
  Region_characteristics(size_t num_regions) { device_init(num_regions); }

  Region_characteristics(const Region_characteristics<ndim>& other)
  {
    device_init(other.num_regions);
    Kokkos::deep_copy(active_regions, other.active_regions);
    Kokkos::deep_copy(sub_dividing_dim, other.sub_dividing_dim);
  }

  void
  device_init(size_t num_regions)
  {
    active_regions = quad::cuda_malloc<int>(num_regions);
    sub_dividing_dim = quad::cuda_malloc<int>(num_regions);
    size = num_regions;
  }

  size_t size = 0;
  ViewVectorInt active_regions;
  ViewVectorInt sub_dividing_dim;
};

#endif