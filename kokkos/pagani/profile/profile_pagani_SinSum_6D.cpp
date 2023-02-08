#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"

class SinSum6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    return sin(x + y + z + k + l + m);
  }
};

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 6;
  SinSum6D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<SinSum6D, ndim>(integrand, vol, num_repeats);
  Kokkos::finalize();
  return 0;
}
