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

class GENZ_3_8D {
  public:
    double
    operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
    {
      return pow(1 + 8 * s + 7 * t + 6 * u + 5 * v + 4 * w + 3 * x + 2 * y + z,
                 -9);
    }
  };

int
main()
{    
  sycl::queue q;  
  ShowDevice(q);  
  
  GENZ_3_8D integrand;
   
  constexpr size_t ndim = 8;
    
  double true_value = 2.2751965817917756076e-10;
  
  double* lows = malloc_shared<double>(ndim, q);  
  double* highs = malloc_shared<double>(ndim, q);  
  
  for(size_t dim = 0; dim < ndim; ++dim){
    lows[dim] = 0.;
    highs[dim] = 1.;
  }
    
  double epsrel = 1.e-3;
  double epsabs = 1.e-20;
    
  bool status = false;

  do{
            status = time_and_call_pagani<GENZ_3_8D, ndim>(q, "oneAPI", "f3", integrand, true_value, lows, highs, epsrel, epsabs);
            epsrel /= 5;
      
  }while(epsrel >= 1.e-9);
  
  sycl::free(lows, q);
  sycl::free(highs, q);  
    
  return 0;
}


