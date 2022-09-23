#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include "oneAPI/quad/Cubature_rules.h"
#include "oneAPI/quad/Region_estimates.h"
#include "oneAPI/quad/Sub_regions.h"
#include "oneAPI/quad/GPUquad/Rule.h"
#include "oneAPI/quad/GPUquad/Phases.h"
#include "oneAPI/quad/Cubature_rules.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <array>
#include "oneAPI/quad/Workspace.h"

#include "oneAPI/demos/demo_utils.h"

#include "oneapi/mkl.hpp"
#include "oneapi/tbb.h"
#include <limits>

 class GENZ_3_3D {
  public:
    double
    operator()(double x, double y, double z)
    {
      return sycl::pown(1 + 3 * x + 2 * y + z, -4);
    }
  };

int
main()
{    
  sycl::queue q(sycl::gpu_selector(), sycl::property::queue::enable_profiling{});
  GENZ_3_3D integrand;  
  constexpr size_t ndim = 3;
  constexpr int warp_size = 32;    
  double true_value = 0.010846560846560846561;
  double init_epsrel = 1.e-3;
  double min_epsrel = 1.e-6;
  double epsabs = 1.e-12;

  double* lows = malloc_shared<double>(ndim, q);  
  double* highs = malloc_shared<double>(ndim, q);  
  
  for(size_t dim = 0; dim < ndim; ++dim){
    lows[dim] = 0.;
    highs[dim] = 1.;
  }
    
  double epsrel = 1.e-3;
  bool status = false;
  do{
      status = time_and_call_pagani<GENZ_3_3D, ndim>(q, "oneAPI", "f3", integrand, true_value, lows, highs, epsrel, epsabs);
      epsrel /= 5;
    }while(epsrel >= 1.e-9);
  
  sycl::free(lows, q);
  sycl::free(highs, q);
  return 0;
}


