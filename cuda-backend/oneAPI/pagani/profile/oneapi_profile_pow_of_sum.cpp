#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"

class Pow_of_product {
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
		return sycl::pow(1. + 8. * s + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, 9.);
    }
};


int main(){
    constexpr int ndim = 8;
    Pow_of_product integrand;
	quad::Volume<double, ndim> vol;
	call_cubature_rules<Pow_of_product, ndim>(integrand, vol);
    return 0;
}

