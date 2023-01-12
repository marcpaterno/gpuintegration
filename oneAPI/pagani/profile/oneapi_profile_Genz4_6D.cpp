//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"

class GENZ_4_6D {
  public:
    SYCL_EXTERNAL double
    operator()(double x, double y, double z, double w, double v, double b)
    {
      double beta = .5;
      return sycl::exp(
        -1.0 * sycl::pow(25., 2.) * ( 
				sycl::pow(x - beta, 2.) + sycl::pow(y - beta, 2.) +
                sycl::pow(z - beta, 2.) + sycl::pow(w - beta, 2.) +
                sycl::pow(v - beta, 2.) + sycl::pow(b - beta, 2.)));
    }
};

int main(){
    constexpr int ndim = 6;
    GENZ_4_6D integrand;
	quad::Volume<double, ndim> vol;
	
	call_cubature_rules<GENZ_4_6D, ndim>(integrand, vol);
  
    return 0;
}

