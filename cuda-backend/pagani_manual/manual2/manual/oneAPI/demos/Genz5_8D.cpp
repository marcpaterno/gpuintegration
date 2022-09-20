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

using namespace quad;

class GENZ_5_8D {
  public:
    double
    operator()(double x,
               double y,
               double z,
               double k,
               double m,
               double n,
               double p,
               double q)
    {
      double beta = .5;
      double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                  10. * fabs(z - beta) - 10. * fabs(k - beta) -
                  10. * fabs(m - beta) - 10. * fabs(n - beta) -
                  10. * fabs(p - beta) - 10. * fabs(q - beta);
      return sycl::exp(t1);
    }
  };


int
main()
{    
  sycl::queue q;  
  ShowDevice(q);  
  GENZ_5_8D integrand;  
  constexpr size_t ndim = 8;
  constexpr int warp_size = 32;  
  double true_value = 2.425217625641885e-06;
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
        status = time_and_call_pagani<GENZ_5_8D, ndim>(q, "oneAPI", "f5", integrand, true_value, lows, highs, epsrel, epsabs);
        epsrel /= 5;
  }while(epsrel >= 1.e-9);
  
  sycl::free(lows, q);
  sycl::free(highs, q);  
  return 0;
}


