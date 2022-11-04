#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"

class GENZ_3_3D {
  public:
    double
    operator()(double x, double y, double z)
    {
      return sycl::pown(1 + 3 * x + 2 * y + z, -4);
    }
};

int main(){
    constexpr int ndim = 3;
    GENZ_3_3D integrand;
	quad::Volume<double, ndim> vol;
	
	call_cubature_rules<GENZ_3_3D, ndim>(integrand, vol);
  
    return 0;
}

