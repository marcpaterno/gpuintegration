#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"
#include "common/kokkos/integrands.cuh"

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 6;
  F_2_6D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<F_2_6D, ndim>(integrand, vol, num_repeats);
  Kokkos::finalize();
  return 0;
}
