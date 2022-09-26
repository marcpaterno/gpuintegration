#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>


using namespace quad;

namespace detail {
  class GENZ_3_3D {
  public:
    double
    operator()(double x, double y, double z)
    {
		//return -.1;
	  //double expo = -4;
      return sycl::pown(1 + 3 * x + 2 * y + z, -4);
    }
  };
}

int
main()
{
  ShowDevice(dpct::get_default_queue());
  double epsrel = 1.e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value = 0.010846560846560846561;
  detail::GENZ_3_3D integrand;
  
  constexpr int ndim = 3;
  bool relerr_classification = true;
  
  constexpr bool debug = false;
  while (clean_time_and_call<detail::GENZ_3_3D, ndim, false, debug>("3D f3",
                                                   integrand,
                                                   epsrel,
                                                   true_value,
                                                   "oneAPI_pagani",
                                                   std::cout,
                                                   relerr_classification) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
	break;
  }
    
     epsrel = 1.e-3;
    /*while (clean_time_and_call<detail::GENZ_3_3D, ndim, true>("3D f3",
                                                   integrand,
                                                   epsrel,
                                                   true_value,
                                                   "oneAPI_pagani",
                                                   std::cout,
                                                   relerr_classification) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
  }*/
}
