//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include "oneAPI/mcubes/demo_utils.dp.hpp"
#include "common/oneAPI/integrands.hpp"

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  sycl::queue q;
  quad::ShowDevice(q);
  call_cubature_rules<F_1_8D, 8>(num_repeats, "f1");
  /*call_cubature_rules<F_2_8D, 8>(num_repeats, "f2");
  call_cubature_rules<F_3_8D, 8>(num_repeats, "f3");
  call_cubature_rules<F_4_8D, 8>(num_repeats, "f4");
  call_cubature_rules<F_5_8D, 8>(num_repeats, "f5");
  call_cubature_rules<F_6_8D, 8>(num_repeats, "f6");
  call_cubature_rules<F_1_7D, 7>(num_repeats, "f1");
  call_cubature_rules<F_2_7D, 7>(num_repeats, "f2");
  call_cubature_rules<F_3_7D, 7>(num_repeats, "f3");
  call_cubature_rules<F_4_7D, 7>(num_repeats, "f4");
  call_cubature_rules<F_5_7D, 7>(num_repeats, "f5");
  call_cubature_rules<F_6_7D, 7>(num_repeats, "f6");
  call_cubature_rules<F_1_6D, 6>(num_repeats, "f1");
  call_cubature_rules<F_2_6D, 6>(num_repeats, "f2");
  call_cubature_rules<F_3_6D, 6>(num_repeats, "f3");
  call_cubature_rules<F_4_6D, 6>(num_repeats, "f4");
  call_cubature_rules<F_5_6D, 6>(num_repeats, "f5");
  call_cubature_rules<F_6_6D, 6>(num_repeats, "f6");
  
  call_cubature_rules<SinSum_8D, 8>(num_repeats, "sinsum");
  call_cubature_rules<SinSum_7D, 7>(num_repeats, "sinsum");
  call_cubature_rules<SinSum_6D, 6>(num_repeats, "sinsum");
  call_cubature_rules<SinSum_5D, 5>(num_repeats, "sinsum");
  call_cubature_rules<SinSum_4D, 4>(num_repeats, "sinsum");
  call_cubature_rules<SinSum_3D, 3>(num_repeats, "sinsum");
  
  call_cubature_rules<Addition_8D, 8>(num_repeats, "addition");
  call_cubature_rules<Addition_7D, 7>(num_repeats, "addition");
  call_cubature_rules<Addition_6D, 6>(num_repeats, "addition");
  call_cubature_rules<Addition_5D, 5>(num_repeats, "addition");
  call_cubature_rules<Addition_4D, 4>(num_repeats, "addition");
  call_cubature_rules<Addition_3D, 3>(num_repeats, "addition");*/
  return 0;
}

