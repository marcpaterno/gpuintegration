#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"
#include "kokkos/mcubes/demos/time_and_call.h"
#include "common/kokkos/integrands.cuh"

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr bool use_custom = true;
  call_cubature_rules<F_1_8D, 8, use_custom>(num_repeats);
  call_cubature_rules<F_2_8D, 8, use_custom>(num_repeats);
  call_cubature_rules<F_3_8D, 8, use_custom>(num_repeats);
  call_cubature_rules<F_4_8D, 8, use_custom>(num_repeats);
  call_cubature_rules<F_5_8D, 8, use_custom>(num_repeats);
  call_cubature_rules<F_6_8D, 8, use_custom>(num_repeats);
  call_cubature_rules<F_1_7D, 7, use_custom>(num_repeats);
  call_cubature_rules<F_2_7D, 7, use_custom>(num_repeats);
  call_cubature_rules<F_3_7D, 7, use_custom>(num_repeats);
  call_cubature_rules<F_4_7D, 7, use_custom>(num_repeats);
  call_cubature_rules<F_5_7D, 7, use_custom>(num_repeats);
  call_cubature_rules<F_6_7D, 7, use_custom>(num_repeats);
  call_cubature_rules<F_1_6D, 6, use_custom>(num_repeats);
  call_cubature_rules<F_2_6D, 6, use_custom>(num_repeats);
  call_cubature_rules<F_3_6D, 6, use_custom>(num_repeats);
  call_cubature_rules<F_4_6D, 6, use_custom>(num_repeats);
  call_cubature_rules<F_5_6D, 6, use_custom>(num_repeats);
  call_cubature_rules<F_6_6D, 6, use_custom>(num_repeats);

  call_cubature_rules<SinSum_8D, 8, use_custom>(num_repeats);
  call_cubature_rules<SinSum_7D, 7, use_custom>(num_repeats);
  call_cubature_rules<SinSum_6D, 6, use_custom>(num_repeats);
  call_cubature_rules<SinSum_5D, 5, use_custom>(num_repeats);
  call_cubature_rules<SinSum_4D, 4, use_custom>(num_repeats);
  call_cubature_rules<SinSum_3D, 3, use_custom>(num_repeats);
  
  call_cubature_rules<Addition_8D, 8, use_custom>(num_repeats);
  call_cubature_rules<Addition_7D, 7, use_custom>(num_repeats);
  call_cubature_rules<Addition_6D, 6, use_custom>(num_repeats);
  call_cubature_rules<Addition_5D, 5, use_custom>(num_repeats);
  call_cubature_rules<Addition_4D, 4, use_custom>(num_repeats);
  call_cubature_rules<Addition_3D, 3, use_custom>(num_repeats); 
  Kokkos::finalize();

  return 0;
}
