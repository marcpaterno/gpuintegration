#include "cuda/mcubes/demos/demo_utils.cuh"
#include "cuda/mcubes/vegasT.cuh"

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
main(int argc, char** argv)
{
  double epsrel = 1e-3;
  double epsrel_min = 1e-9;
  constexpr int ndim = 8;

  double ncall = 1.0e6;
  int titer = 100;
  int itmax = 20;
  int skip = 5;
  VegasParams params(ncall, titer, itmax, skip);

  double true_value = 2.2751965817917756076e-10;

  double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  GENZ_3_8D integrand;
  PrintHeader();

  PrintHeader();

  size_t expID = 0;
  bool success = false;
  do {
    for (int run = 0; run < 100; run++) {
      success = mcubes_time_and_call<GENZ_3_8D, ndim>(
        integrand, epsrel, true_value, "f3 8D", params, &volume);
      if (!success)
        break;
    }
    epsrel /= 5.;
    expID++;
  } while (success == true && epsrel >= epsrel_min);

  return 0;
}
