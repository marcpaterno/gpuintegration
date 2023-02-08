#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"

class F_5_8D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q)
  {
    double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta) -
                10. * fabs(p - beta) - 10. * fabs(q - beta);
    return exp(t1);
  }
};

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 8;
  F_5_8D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<F_5_8D, ndim>(integrand, vol, num_repeats);
  Kokkos::finalize();
  return 0;
}
