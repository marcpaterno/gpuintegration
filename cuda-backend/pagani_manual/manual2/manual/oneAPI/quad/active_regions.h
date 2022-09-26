#ifndef ONE_API_ACTIVE_REGIONS_H
#define ONE_API_ACTIVE_REGIONS_H

//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>

#include "oneAPI/quad/Region_characteristics.h"
#include "oneAPI/quad/Finished_estimates.h"
#include "oneAPI/quad/util/MemoryUtil.h"

// this is placed on its own header because of header inclusion issues with
// <oneapi/dpl/execution> and the tbb library this is the order that works for
// the test using this, the order of the catch.hpp  header needs to be after
// execution & async?? why??

/*
#define CATCH_CONFIG_MAIN
#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include "externals/catch2/catch.hpp"
#include <CL/sycl.hpp>
#include "oneAPI/quad/Region_characteristics.h"
#include "oneAPI/quad/Finished_estimates.h"

does not require linking with tbb for compilation

*/
template <size_t ndim>
size_t
get_num_active_regions(sycl::queue& q, Region_characteristics<ndim>& regs)
{
  const size_t num_regions = regs.size;
  double* active_regions = regs.active_regions;
  double* scanned_array = sycl::malloc_device<double>(num_regions, q);

  dpl::experimental::exclusive_scan_async(
    oneapi::dpl::execution::make_device_policy(q),
    active_regions,
    active_regions + num_regions,
    scanned_array,
    0.)
    .wait();
  size_t num_active = scanned_array[num_regions - 1];
  if (active_regions[num_regions - 1] == 1)
    num_active++;
  sycl::free(scanned_array, q);
  return static_cast<size_t>(num_active);
}

#endif
