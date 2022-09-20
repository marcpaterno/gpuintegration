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

 class GENZ_6_6D {
  public:
    double
    operator()(double u, double v, double w, double x, double y, double z)
    {
      if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
        return 0.;
      else
        return sycl::exp(10 * z + 9 * y + 8 * x + 7 * w + 6 * v +
                   5 * u) /*/1.5477367885091207413e8*/;
    }
};

int
main()
{    
  sycl::queue q;  
  ShowDevice(q);  
  GENZ_6_6D integrand;  
  constexpr size_t ndim = 6;
  constexpr int warp_size = 32;    
  double true_value = 1.5477367885091207413e8;
  double init_epsrel = 1.e-3;
  double min_epsrel = 1.e-6;
  double epsabs = 1.e-20;
  double* lows = malloc_shared<double>(ndim, q);  
  double* highs = malloc_shared<double>(ndim, q);  
  
  for(size_t dim = 0; dim < ndim; ++dim){
    lows[dim] = 0.;
    highs[dim] = 1.;
  }
    
  double epsrel = 1.e-3;
  bool status = false;

    do{
            status = time_and_call_pagani<GENZ_6_6D, ndim>(q, "oneAPI", "f6,", integrand, true_value, lows, highs, epsrel, epsabs);
            epsrel /= 5;
  }while(epsrel >= 1.e-9);
  
    
  sycl::free(lows, q);
  sycl::free(highs, q);  
  return 0;
}


