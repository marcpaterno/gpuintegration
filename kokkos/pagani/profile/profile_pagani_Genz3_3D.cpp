#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"

class F_3_3D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z)
  {
  return 0.1;
    return pow(1 + 3 * x + 2 * y + z, -4);
  }
};


int
main(int argc, char** argv)
{
  Kokkos::initialize();
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 3;
  F_3_3D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<F_3_3D, ndim>(integrand, vol, num_repeats);
  Kokkos::finalize();
  return 0;
}
