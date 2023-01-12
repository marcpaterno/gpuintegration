//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"

class Simple_6D {
  public:
    SYCL_EXTERNAL double
    operator()(double x, double y, double z, double k, double l, double m)
    {
		double sum = 0.;
	for(int i=0; i < 1000; ++i)
		sum += (x*y*z*k*l*m)/(x/y/z/k/l/m);
	return sum;
    }
};

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
    constexpr int ndim = 6;
    Simple_6D integrand;
	quad::Volume<double, ndim> vol;
	call_cubature_rules<Simple_6D, ndim>(integrand, vol, num_repeats);
    return 0;
}

