/*
  Compile with:
                nvcc Genz6_2D.cu -arch=sm_60
*/

//#include <CL/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <dpct/dpct.hpp>
#include "oneAPI/mcubes/demo_utils.dp.hpp"
#include "oneAPI/mcubes/vegasT.dp.hpp"


class GENZ_custom {
public:
  SYCL_EXTERNAL double operator()(double y, double z)
  {
    if (z > 0.5 || y > 0.5)
      return 0.;
    else
      return z*y*y;
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

  double true_value = 0.00520833;

  double lows[] = {0., 0.};
  double highs[] = {1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  GENZ_custom integrand;

  print_mcubes_header();
    
  while (mcubes_time_and_call<GENZ_custom, ndim>(
           integrand, epsrel, true_value, "GENZ_custom", params, &volume) ==
           true &&
         epsrel >= epsrel_min) {
    break;
    epsrel /= 5.;
  }

  return 0;
}
