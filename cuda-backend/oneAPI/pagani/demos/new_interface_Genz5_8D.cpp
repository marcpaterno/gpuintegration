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
      double t1 = -10. * sycl::fabs(x - beta) - 10. * fabs(y - beta) -
                  10. * sycl::fabs(z - beta) - 10. * fabs(k - beta) -
                  10. * fabs(m - beta) - 10. * fabs(n - beta) -
                  10. * fabs(p - beta) - 10. * fabs(q - beta);
      return sycl::exp(t1);
    }
  };
}

int
main()
{
  ShowDevice(dpct::get_default_queue());
  double epsrel = 1.e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value = 2.425217625641885e-06;
  detail::GENZ_5_8D integrand;
  
  constexpr int ndim = 8;
  bool relerr_classification = true;
  
  while (clean_time_and_call<detail::GENZ_5_8D, ndim, false>("f5",
                                                   integrand,
                                                   epsrel,
                                                   true_value,
                                                   "oneAPI_pagani",
                                                   std::cout,
                                                   relerr_classification) == true &&
         epsrel > epsrel_min) {
		break;
    epsrel /= 5.0;
  }
    
   /*  epsrel = 1.e-3;
    while (clean_time_and_call<detail::GENZ_5_8D, ndim, true>("f5",
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
