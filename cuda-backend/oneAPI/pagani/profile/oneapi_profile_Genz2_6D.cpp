#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"

class GENZ_2_6D {
  public:
    SYCL_EXTERNAL double
    operator()(double x, double y, double z, double k, double l, double m)
    {
        const double a = 50.;
        const double b = .5;

        const double term_1 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(x - b, 2.));
        const double term_2 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(y - b, 2.));
        const double term_3 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(z - b, 2.));
        const double term_4 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(k - b, 2.));
        const double term_5 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(l - b, 2.));
        const double term_6 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(m - b, 2.));
        
        double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
        return val;

    }
};

int main(){
    constexpr int ndim = 6;
    GENZ_2_6D integrand;
	quad::Volume<double, ndim> vol;
	call_cubature_rules<GENZ_2_6D, ndim>(integrand, vol);
    return 0;
}

