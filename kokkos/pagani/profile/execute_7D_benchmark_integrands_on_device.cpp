#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"
#include <array>
#include "common/kokkos/integrands.cuh"

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  size_t num_invocations = argc > 1 ? std::stoi(argv[1]) : 100000;
  constexpr int ndim = 7;
  double sum = 0.;
  sum += execute_integrand_at_points<F_1_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_2_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_3_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_4_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_5_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_6_7D, ndim>(num_invocations);
  printf("%.15e\n", sum);
  Kokkos::finalize();
  return 0;
}
