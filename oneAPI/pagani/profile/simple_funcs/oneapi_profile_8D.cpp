#include <CL/sycl.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"


class Simple_8D {
  public:
    SYCL_EXTERNAL double
    operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
    {
		double sum = 0.;
	for(int i=0; i < 1000; ++i)
		sum += (x*y*z*w*v*u+t+s)/(x/y/w/v/u/t/s);
	return sum; 
    }
};

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
    constexpr int ndim = 8;
    Simple_8D integrand;
	quad::Volume<double, ndim> vol;
	call_cubature_rules<Simple_8D, ndim>(integrand, vol, num_repeats);
    return 0;
}
