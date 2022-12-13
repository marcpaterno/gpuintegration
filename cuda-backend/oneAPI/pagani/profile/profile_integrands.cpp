//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include "oneAPI/mcubes/demo_utils.dp.hpp"
#include "oneAPI/integrands.hpp"

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  call_cubature_rules<F_1_8D, 8>(num_repeats);
  call_cubature_rules<F_2_8D, 8>(num_repeats);
  call_cubature_rules<F_3_8D, 8>(num_repeats);
  call_cubature_rules<F_4_8D, 8>(num_repeats);
  call_cubature_rules<F_5_8D, 8>(num_repeats);
  call_cubature_rules<F_6_8D, 8>(num_repeats);
  call_cubature_rules<F_1_7D, 7>(num_repeats);
  call_cubature_rules<F_2_7D, 7>(num_repeats);
  call_cubature_rules<F_3_7D, 7>(num_repeats);
  call_cubature_rules<F_4_7D, 7>(num_repeats);
  call_cubature_rules<F_5_7D, 7>(num_repeats);
  call_cubature_rules<F_6_7D, 7>(num_repeats);
  call_cubature_rules<F_1_6D, 6>(num_repeats);
  call_cubature_rules<F_2_6D, 6>(num_repeats);
  call_cubature_rules<F_3_6D, 6>(num_repeats);
  call_cubature_rules<F_4_6D, 6>(num_repeats);
  call_cubature_rules<F_5_6D, 6>(num_repeats);
  call_cubature_rules<F_6_6D, 6>(num_repeats);

  call_cubature_rules<SinSum_8D, 8>(num_repeats);
  call_cubature_rules<SinSum_7D, 7>(num_repeats);
  call_cubature_rules<SinSum_6D, 6>(num_repeats);
  call_cubature_rules<SinSum_5D, 5>(num_repeats);
  call_cubature_rules<SinSum_4D, 4>(num_repeats);
  call_cubature_rules<SinSum_3D, 3>(num_repeats);
  
  call_cubature_rules<Addition_8D, 8>(num_repeats);
  call_cubature_rules<Addition_7D, 7>(num_repeats);
  call_cubature_rules<Addition_6D, 6>(num_repeats);
  call_cubature_rules<Addition_5D, 5>(num_repeats);
  call_cubature_rules<Addition_4D, 4>(num_repeats);
  call_cubature_rules<Addition_3D, 3>(num_repeats);  

  call_mcubes_kernel<F_1_8D, 8>(num_repeats);
  call_mcubes_kernel<F_2_8D, 8>(num_repeats);
  call_mcubes_kernel<F_3_8D, 8>(num_repeats);
  call_mcubes_kernel<F_4_8D, 8>(num_repeats);
  call_mcubes_kernel<F_5_8D, 8>(num_repeats);
  call_mcubes_kernel<F_6_8D, 8>(num_repeats);
  call_mcubes_kernel<F_1_7D, 7>(num_repeats);
  call_mcubes_kernel<F_2_7D, 7>(num_repeats);
  call_mcubes_kernel<F_3_7D, 7>(num_repeats);
  call_mcubes_kernel<F_4_7D, 7>(num_repeats);
  call_mcubes_kernel<F_5_7D, 7>(num_repeats);
  call_mcubes_kernel<F_6_7D, 7>(num_repeats);
  call_mcubes_kernel<F_1_6D, 6>(num_repeats);
  call_mcubes_kernel<F_2_6D, 6>(num_repeats);
  call_mcubes_kernel<F_3_6D, 6>(num_repeats);
  call_mcubes_kernel<F_4_6D, 6>(num_repeats);
  call_mcubes_kernel<F_5_6D, 6>(num_repeats);
  call_mcubes_kernel<F_6_6D, 6>(num_repeats);

  call_mcubes_kernel<SinSum_8D, 8>(num_repeats);
  call_mcubes_kernel<SinSum_7D, 7>(num_repeats);
  call_mcubes_kernel<SinSum_6D, 6>(num_repeats);
  call_mcubes_kernel<SinSum_5D, 5>(num_repeats);
  call_mcubes_kernel<SinSum_4D, 4>(num_repeats);
  call_mcubes_kernel<SinSum_3D, 3>(num_repeats);
  
  call_mcubes_kernel<Addition_8D, 8>(num_repeats);
  call_mcubes_kernel<Addition_7D, 7>(num_repeats);
  call_mcubes_kernel<Addition_6D, 6>(num_repeats);
  call_mcubes_kernel<Addition_5D, 5>(num_repeats);
  call_mcubes_kernel<Addition_4D, 4>(num_repeats);
  call_mcubes_kernel<Addition_3D, 3>(num_repeats);  

    return 0;
}

