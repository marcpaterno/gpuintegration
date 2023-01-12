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
  ShowDevice(dpct::get_default_queue());
  double epsrel = 1.e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value = 1.5477367885091207413e8;
  GENZ_6_6D integrand;
  
  constexpr int ndim = 6;
  bool relerr_classification = true;
  
  quad::Volume<double, ndim>  vol;
  
  while (clean_time_and_call<GENZ_6_6D, ndim, false>("f6",
                                                   integrand,
                                                   epsrel,
                                                   true_value,
                                                   "oneAPI_pagani",
                                                   std::cout,
                                                   relerr_classification) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
  }
    
     epsrel = 1.e-3;
    while (clean_time_and_call<GENZ_6_6D, ndim, true>("f6",
                                                   integrand,
                                                   epsrel,
                                                   true_value,
                                                   "oneAPI_pagani",
                                                   std::cout,
                                                   relerr_classification) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
  }
}
