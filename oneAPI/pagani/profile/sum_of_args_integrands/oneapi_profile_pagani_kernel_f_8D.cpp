//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include "oneAPI/integrands.hpp"
//i had initially forgotten the SYCL_EXTERNAL

int
main(int argc, char** argv)
{
    int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
    constexpr int ndim = 8;
    Addition_8D integrand;
	quad::Volume<double, ndim> vol;
	
	call_cubature_rules<Addition_8D, ndim>(integrand, vol, num_repeats);
  
    return 0;
}

