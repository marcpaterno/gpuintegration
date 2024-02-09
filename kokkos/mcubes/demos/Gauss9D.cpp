#include "time_and_call.h"
#include "common/kokkos/Volume.cuh"

class Gauss9D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o,
             double p)
  {
    double sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
                 pow(m, 2) + pow(n, 2) + pow(o, 2) + pow(p, 2);
    return exp(-1 * sum / (2 * pow(0.01, 2))) *
           (1 / pow(sqrt(2 * M_PI) * 0.01, 9));
  }
};

int
main()
{
  Kokkos::initialize();
  {
    double epsrel = 1.e-3;
    constexpr int ndim = 9;
    double ncall = 1.0e8;
    int titer = 15;
    int itmax = 10;
    int skip = 10;
    double true_value = 1.;
    VegasParams params(ncall, titer, itmax, skip);

    double lows[] = {-1., -1., -1., -1., -1., -1., -1., -1., -1.};
    double highs[] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
    quad::Volume<double, ndim> volume(lows, highs);
    Gauss9D integrand;

    PrintHeader();
    bool success = false;
    success = mcubes_time_and_call<Gauss9D, ndim>(
      integrand, epsrel, true_value, "f4 5D", params, &volume);
  }
  Kokkos::finalize();
  return 0;
}
