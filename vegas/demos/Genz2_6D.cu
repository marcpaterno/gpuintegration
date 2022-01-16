#include "vegas/demos/demo_utils.cuh"
#include "vegas/vegasT.cuh"

class GENZ_2_6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    double a = 50.;
    double b = .5;

    double term_1 = 1. / ((1. / pow(a, 2)) + pow(x - b, 2));
    double term_2 = 1. / ((1. / pow(a, 2)) + pow(y - b, 2));
    double term_3 = 1. / ((1. / pow(a, 2)) + pow(z - b, 2));
    double term_4 = 1. / ((1. / pow(a, 2)) + pow(k - b, 2));
    double term_5 = 1. / ((1. / pow(a, 2)) + pow(l - b, 2));
    double term_6 = 1. / ((1. / pow(a, 2)) + pow(m - b, 2));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
    return val;
  }
};

int
main(int argc, char** argv)
{
  double epsrel = 1e-3;
  double epsrel_min = 1e-9;
  constexpr int ndim = 6;
  double ncall = 1.e8;
  int titer = 100;
  int itmax = 40;
  int skip = 10;
  VegasParams params(ncall, titer, itmax, skip);

  double true_value = 1.286889807581113e+13;

  double lows[] = {0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  GENZ_2_6D integrand;

  PrintHeader();
  bool success = false;
  size_t expID = 0;
  do {
    params.ncall = ncall;
    for (int run = 0; run < 100; run++) {
      success = mcubes_time_and_call<GENZ_2_6D, ndim>(
        integrand, epsrel, true_value, "f2 6D", params, &volume);
      if (!success)
        break;
    }
    epsrel /= 5.;
  } while (epsrel >= epsrel_min && success == true);

  return 0;
}
