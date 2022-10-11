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
        -1.0 * (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2) +
                pow(25, 2) * pow(z - beta, 2) + pow(25, 2) * pow(w - beta, 2) +
                pow(25, 2) * pow(v - beta, 2)));
    }
  };


int main(){
    //double epsrel = 1.0e-16;
	double epsrel = 1.0e-3;
    double const epsrel_min = 1.0240000000000002e-10;
    constexpr int ndim = 5;
    GENZ_4_5D integrand;
    double true_value = 1.79132603674879e-06;
	quad::Volume<double, ndim> vol;
	
	for(int i=0; i < 10; ++i)
		call_cubature_rules<GENZ_4_5D, ndim>(integrand, vol);
   /*while (clean_time_and_call<GENZ_4_5D, ndim, false>("f4",
                                           integrand,
                                           epsrel,
                                           true_value,
                                           "gpucuhre",
                                           std::cout) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
	break;
  }*/
    
   /*while (clean_time_and_call<GENZ_4_5D, ndim, true>("f4",
                                           integrand,
                                           epsrel,
                                           true_value,
                                           "gpucuhre",
                                           std::cout) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
	break;
  }*/
    return 0;
}

