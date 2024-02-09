#include "time_and_call.h"
#include "common/kokkos/Volume.cuh"

class GENZ_4_5D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double w, double v)
  {
    double beta = .5;
    return exp(-1.0 *
               (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2) +
                pow(25, 2) * pow(z - beta, 2) + pow(25, 2) * pow(w - beta, 2) +
                pow(25, 2) * pow(v - beta, 2)));
  }
};

int
main()
{
  Kokkos::initialize();
  {
    double epsrel = 1e-3;
    double epsrel_min = 1e-9;
    constexpr int ndim = 5;

    double ncall = 1.0e6;
    int titer = 100;
    int itmax = 20;
    int skip = 5;
    VegasParams params(ncall, titer, itmax, skip);

    double true_value = 1.79132603674879e-06;

    double lows[] = {0., 0., 0., 0., 0.};
    double highs[] = {1., 1., 1., 1., 1.};
    quad::Volume<double, ndim> volume(lows, highs);
    GENZ_4_5D integrand;

    PrintHeader();
    bool success = false;

    do {
      for (int run = 0; run < 10; run++) {
        success = mcubes_time_and_call<GENZ_4_5D, ndim>(
          integrand, epsrel, true_value, "f4 5D", params, &volume);
        if (!success)
          break;
      }
      epsrel /= 5.;

    } while (success == true && epsrel >= epsrel_min);
  }
  Kokkos::finalize();
  return 0;
}
