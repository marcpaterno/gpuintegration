#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"
#include "common/kokkos/integrands.cuh"

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 5;
  constexpr bool use_custom = true;
  F_4_5D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<F_4_5D, ndim, use_custom>(integrand, vol, num_repeats);
  std::cout << "done\n";
  Kokkos::finalize();
  return 0;
}
