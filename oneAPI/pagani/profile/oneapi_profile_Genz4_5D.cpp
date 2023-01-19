//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include "oneAPI/integrands.hpp"

/*class GENZ_4_5D {
  public:
    SYCL_EXTERNAL double
    operator()(double x, double y, double z, double w, double v)
    {	  	
	  double beta = .5;
      return sycl::exp(
        -1.0 * (sycl::pow(25., 2.) * sycl::pow(x - beta, 2.) + sycl::pow(25., 2.) * sycl::pow(y - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(z - beta, 2.) + sycl::pow(25., 2.) * sycl::pow(w - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(v - beta, 2.)));
    }
  };*/

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
    constexpr int ndim = 8;
    F_4_8D integrand;
	quad::Volume<double, ndim> vol;
	
	call_cubature_rules<F_4_8D, ndim>(integrand, vol, num_repeats);
  
    return 0;
}
