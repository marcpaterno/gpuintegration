#include "cuda/mcubes/demos/demo_utils.cuh"
#include "cuda/mcubes/vegasT.cuh"

class GENZ_6_6D {
public:
  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y, double z)
  {
    if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return exp(10 * z + 9 * y + 8 * x + 7 * w + 6 * v +
                 5 * u) /*/1.5477367885091207413e8*/;
  }
};

int
main(int argc, char** argv)
{
  double epsrel = 1.e-3;
  double epsrel_min = 1e-9;
  constexpr int ndim = 6;

  double ncall = 1.0e6;
  int titer = 100;
  int itmax = 20;
  int skip = 5;
  VegasParams params(ncall, titer, itmax, skip);

  double true_value = 1.5477367885091207413e8;

  double lows[] = {0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  GENZ_6_6D integrand;

  PrintHeader();
  // std::array<double, 10> required_ncall =
  // {1.e7, 1.e7, 1.e7, 3.e9, 2.e9, 8.e9, 8.e9, 8.e9, 8.e9, 8.e9};

  bool success = false;
  // size_t expID = 0;
  do {
    params.ncall = ncall;
    for (int run = 0; run < 100; run++) {
      success = mcubes_time_and_call<GENZ_6_6D, ndim>(
        integrand, epsrel, true_value, "f6 6D", params, &volume);
      if (!success)
        break;
    }
    epsrel /= 5.;
  } while (epsrel >= epsrel_min && success == true);

  return 0;
}