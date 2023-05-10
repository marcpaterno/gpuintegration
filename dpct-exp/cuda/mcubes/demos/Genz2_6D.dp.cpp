#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "dpct-exp/cuda/mcubes/demos/demo_utils.dp.hpp"
#include "dpct-exp/cuda/mcubes/vegasT.dp.hpp"
#include "common/oneAPI/integrands.hpp"

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
  F_2_6D integrand;
  std::array<double, 6> required_ncall =
    //{1.e8, 1.e8, 1.e8, 1.e8, 1.e8, 1.e9};
    {1.e8, 1.e8, 1.e8, 1.e8, 1.e8, 1.e9};
  print_mcubes_header();
  bool success = false;
  size_t curr_epsrel = 0;
  do {
    params.ncall = required_ncall[curr_epsrel];
    for (int run = 0; run < 1; run++) {
      success = mcubes_time_and_call<F_2_6D, ndim, false, Custom_generator>(
        integrand, epsrel, true_value, "f2, 6", params, &volume);
      if (!success)
        break;
    }

    break;
    epsrel /= 5.;
    curr_epsrel++;
    if (curr_epsrel > required_ncall.size())
      break;
  } while (epsrel >= epsrel_min && success == true);

  return 0;
}
