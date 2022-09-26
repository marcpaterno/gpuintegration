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
  class GENZ_4_5D {
  public:
    double
    operator()(double x, double y, double z, double w, double v)
    {
      // double alpha = 25.;
      double beta = .5;
      return sycl::exp(-1.0 * (pow(25, 2) * (x - beta) * (x - beta) +
                         pow(25, 2) * (y - beta) * (y - beta) +
                         pow(25, 2) * (z - beta) * (z - beta) +
                         pow(25, 2) * (w - beta) * (w - beta) +
                         pow(25, 2) * (v - beta) * (v - beta)));
    }
  };
}

int
main()
{
  double epsrel = 1.e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value = 1.79132603674879e-06;
  detail::GENZ_4_5D integrand;
  
  constexpr int ndim = 5;
  bool relerr_classification = true;
  
  while (clean_time_and_call<detail::GENZ_4_5D, ndim, false>("f4",
                                                   integrand,
                                                   epsrel,
                                                   true_value,
                                                   "gpucuhre",
                                                   std::cout,
                                                   relerr_classification) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
  }
    
    epsrel = 1.e-3;
     while (clean_time_and_call<detail::GENZ_4_5D, ndim, true>("f4",
                                                   integrand,
                                                   epsrel,
                                                   true_value,
                                                   "gpucuhre",
                                                   std::cout,
                                                   relerr_classification) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
  }
}
