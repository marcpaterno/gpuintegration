#include <chrono>
#include <cmath>
#include <iomanip>

#include "cudaPagani/quad/quad.h"
#include "cudaPagani/quad/util/cudaUtil.h"
#include "function.cuh"

#include "cudaPagani/quad/GPUquad/Interp2D.cuh"
#include "cudaPagani/quad/GPUquad/Pagani.cuh"
#include "cudaPagani/quad/util/Volume.cuh"

using namespace quad;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

constexpr double EPSABS = 1.0e-40;

int
main(int argc, char** argv)
{
  TYPE epsrel = 2.560e-09;
  constexpr int ndim = 8;

  Pagani<TYPE, ndim> pagani;
  BoxIntegral8_22 integrand;
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1

  double highs[ndim] = {1., 1., 1., 1., 1., 1., 1., 1.};
  double lows[ndim] = {0., 0., 0., 0., 0., 0., 0., 0.};
  Volume<double, ndim> vol(lows, highs);
  double true_value = 1495369.283757217694;

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  auto t0 = std::chrono::high_resolution_clock::now();
  cuhreResult result = pagani.integrate<BoxIntegral8_22>(
    integrand, epsrel, EPSABS, &vol, outfileVerbosity, _final, phase_I_type);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

  printf("%.20f +- %.20f epsrel:%e, nregions:%lu flag:%i time:%f error:%.17f, "
         "ratio:%.17f failed phase2 blocks:%i\n",
         result.estimate,
         result.errorest,
         epsrel,
         result.nregions,
         result.status,
         dt.count(),
         abs(true_value - result.estimate),
         result.errorest / MaxErr(result.estimate, epsrel, EPSABS),
         result.phase2_failedblocks);
  return 0;
}
