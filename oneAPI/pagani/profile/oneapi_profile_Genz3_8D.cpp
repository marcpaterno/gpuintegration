#include <CL/sycl.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"


class GENZ_3_8D {
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
		return sycl::pow(1. + 8. * s + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
    }
};

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
    constexpr int ndim = 8;
    GENZ_3_8D integrand;
	quad::Volume<double, ndim> vol;
	call_cubature_rules<GENZ_3_8D, ndim>(integrand, vol, num_repeats);
    return 0;
}
