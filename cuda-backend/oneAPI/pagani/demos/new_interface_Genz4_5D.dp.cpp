#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"

  class GENZ_4_5D {
  public:
    SYCL_EXTERNAL double
    operator()(double x, double y, double z, double w, double v)
    {
      // double alpha = 25.;
      double beta = .5;
      return sycl::exp(
        -1.0 *
        (25 * 25 * (x - beta) * (x - beta) + 25 * 25 * (y - beta) * (y - beta) +
         25 * 25 * (z - beta) * (z - beta) + 25 * 25 * (w - beta) * (w - beta) +
         25 * 25 * (v - beta) * (v - beta)));
    }
  };

int main(){
    double epsrel = 1.0e-3;
    double const epsrel_min = 1.0240000000000002e-10;
    constexpr int ndim = 5;
    GENZ_4_5D integrand;
    double true_value = 1.79132603674879e-06;
	
	
	
   /*while (clean_time_and_call<GENZ_4_5D, ndim, false>("f4",
                                           integrand,
                                           epsrel,
                                           true_value,
                                           "gpucuhre",
                                           std::cout) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }*/
    
    epsrel = 1.e-3;
   while (clean_time_and_call<GENZ_4_5D, ndim, true>("f4",
                                           integrand,
                                           epsrel,
                                           true_value,
                                           "gpucuhre",
                                           std::cout) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }
  }
    return 0;
}

