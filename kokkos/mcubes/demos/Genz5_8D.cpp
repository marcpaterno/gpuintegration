#include "time_and_call.h"
#include "common/kokkos/Volume.cuh"

class GENZ_5_8D {
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
main()
{
  Kokkos::initialize();
  {
    double epsrel = 1e-3;
    double epsrel_min = 1e-9;
    constexpr int ndim = 8;

    double ncall = 1.0e6;
    int titer = 100;
    int itmax = 20;
    int skip = 5;
    VegasParams params(ncall, titer, itmax, skip);

    double true_value = 2.42521762564189e-06;

    double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
    double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
    quad::Volume<double, ndim> volume(lows, highs);
    GENZ_5_8D integrand;

    PrintHeader();
    bool success = false;

    do {
      for (int run = 0; run < 10; run++) {
        success = mcubes_time_and_call<GENZ_5_8D, ndim>(
          integrand, epsrel, true_value, "f5 8D", params, &volume);
        if (!success)
          break;
      }
      epsrel /= 5.;

    } while (success == true && epsrel >= epsrel_min);
  }
  Kokkos::finalize();
  return 0;
}
