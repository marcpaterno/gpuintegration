//#include <CL/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <dpct/dpct.hpp>
#include "oneAPI/dpct_latest/demo_utils.dp.hpp"
#include "oneAPI/dpct_latest/mcubes/vegasT.dp.hpp"


class GENZ_6_2D {
public:
  SYCL_EXTERNAL double operator()(double y, double z)
  {
    if (z > .9 || y > .8)
      return 0.;
    else
      return sycl::exp(10 * z + 9 * y);
  }
};

int
main(int argc, char** argv)
{
  double epsrel = 1e-3;
  double epsrel_min = 1.e-6;
  constexpr int ndim = 2;

  double ncall = 1.0e7;
  int titer = 20;
  int itmax = 20;
  int skip = 0;
  VegasParams params(ncall, titer, itmax, skip);

  double true_value = 120489.75982636053604;

  double lows[] = {0., 0.};
  double highs[] = {1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  GENZ_6_2D integrand;

  //print_mcubes_header();
    
  while (mcubes_time_and_call<GENZ_6_2D, ndim>(
           integrand, epsrel, true_value, "GENZ_6_2D", params, &volume) ==
           true &&
         epsrel >= epsrel_min) {
    break;
    epsrel /= 5.;
  }

  return 0;
}
