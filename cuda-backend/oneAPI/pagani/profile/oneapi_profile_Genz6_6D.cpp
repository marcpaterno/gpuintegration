#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"

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

int main(){
    constexpr int ndim = 6;
    GENZ_6_6D integrand;
	quad::Volume<double, ndim> vol;
	
	call_cubature_rules<GENZ_6_6D, ndim>(integrand, vol);
  
    return 0;
}

