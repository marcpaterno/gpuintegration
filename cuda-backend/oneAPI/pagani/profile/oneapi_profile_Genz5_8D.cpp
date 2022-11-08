//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"

class GENZ_5_8D {
  public:
    double
       operator()(double x,
               double y,
               double z,
               double k,
               double m,
               double n,
               double p,
               double q)
    {
      double beta = .5;
      double t1 = -10. * sycl::fabs(x - beta) - 10. * fabs(y - beta) -
                  10. * sycl::fabs(z - beta) - 10. * fabs(k - beta) -
                  10. * fabs(m - beta) - 10. * fabs(n - beta) -
                  10. * fabs(p - beta) - 10. * fabs(q - beta);
      return sycl::exp(t1);
    }
};

int main(){
    constexpr int ndim = 8;
    GENZ_5_8D integrand;
	quad::Volume<double, ndim> vol;
	
	call_cubature_rules<GENZ_5_8D, ndim>(integrand, vol);
  
    return 0;
}

