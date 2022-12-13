#include <CL/sycl.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include "oneAPI/integrands.hpp"

int main(){
    constexpr int ndim = 3;
    Addition_3D integrand;
	quad::Volume<double, ndim> vol;
	
	call_cubature_rules<Addition_3D, ndim>(integrand, vol);
  
    return 0;
}

