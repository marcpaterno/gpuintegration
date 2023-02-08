#include "kokkos/mcubes/demos/time_and_call.h"
#include "common/kokkos/Volume.cuh"

class GENZ_3_8D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t,
             double s)
  {
    return pow(1 + 8 * s + 7 * t + 6 * u + 5 * v + 4 * w + 3 * x + 2 * y + z,
               -9);
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

    double true_value = 2.275196581791776e-10;

    double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
    double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
    quad::Volume<double, ndim> volume(lows, highs);
    GENZ_3_8D integrand;

    PrintHeader();
    bool success = false;

    do {
      for (int run = 0; run < 10; run++) {
        success = mcubes_time_and_call<GENZ_3_8D, ndim>(
          integrand, epsrel, true_value, "f3 8D", params, &volume);
        if (!success)
          break;
      }
      epsrel /= 5.;

    } while (success == true && epsrel >= epsrel_min);
  }
  Kokkos::finalize();
  return 0;
}
