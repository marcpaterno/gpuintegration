#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"

class F_3_8D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t,
             double s)
  {
    // return x + y + z + w + v + u + t + s;
    return pow(1. + 8. * s + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x +
                 2. * y + z,
               -9.);
  }
};

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 8;
  F_3_8D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<F_3_8D, ndim>(integrand, vol, num_repeats);
  Kokkos::finalize();
  return 0;
}
