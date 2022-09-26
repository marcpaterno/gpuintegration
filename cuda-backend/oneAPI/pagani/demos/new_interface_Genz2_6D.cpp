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
}

int
main()
{
  ShowDevice(dpct::get_default_queue());
  double epsrel = 1.e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value = 1.286889807581113e+13;
  detail::GENZ_2_6D integrand;
  
  constexpr int ndim = 6;
  bool relerr_classification = true;
  constexpr bool debug = false;
  while (clean_time_and_call<detail::GENZ_2_6D, ndim, false, debug>("f2",
                                                   integrand,
                                                   epsrel,
                                                   true_value,
                                                   "oneAPI_pagani",
                                                   std::cout,
                                                   relerr_classification) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
  }
    
   /* epsrel = 1.e-3;
    while (clean_time_and_call<detail::GENZ_2_6D, ndim, true>("f2",
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
