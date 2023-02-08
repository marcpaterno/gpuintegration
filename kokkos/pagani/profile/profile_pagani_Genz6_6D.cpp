#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"

class F_6_6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double u, double v, double w, double x, double y, double z)
  {
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u);
  }
};

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 100;
  constexpr int ndim = 6;
  F_6_6D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<F_6_6D, ndim>(integrand, vol, num_repeats);
  Kokkos::finalize();
  return 0;
}
