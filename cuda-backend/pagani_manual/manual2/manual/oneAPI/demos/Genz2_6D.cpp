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

  class GENZ_2_6D {
  public:
    double
    operator()(double x, double y, double z, double k, double l, double m)
    {
        const double a = 50.;
        const double b = .5;

        const double term_1 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(x - b, 2.));
        const double term_2 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(y - b, 2.));
        const double term_3 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(z - b, 2.));
        const double term_4 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(k - b, 2.));
        const double term_5 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(l - b, 2.));
        const double term_6 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(m - b, 2.));
        
        double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
        return val;
    }
  };

int
main()
{    
  sycl::queue q;  
  ShowDevice(q);  
  GENZ_2_6D integrand;  
  constexpr size_t ndim = 6;
  constexpr int warp_size = 32;    
  double true_value = 1.286889807581113e+13;
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
            status = time_and_call_pagani<GENZ_2_6D, ndim>(q, "oneAPI", "f2", integrand, true_value, lows, highs, epsrel, epsabs);
            epsrel /= 5;
  }while(epsrel >= 1.e-9);
  
    
  sycl::free(lows, q);
  sycl::free(highs, q);  
  return 0;
}


