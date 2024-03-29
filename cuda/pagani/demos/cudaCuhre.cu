#include <chrono>
#include <cmath>
#include <iomanip>

#include "cuda/pagani/quad/quad.h"
#include "common/cuda/cudaUtil.h"
#include "cuda/pagani/demos/function.cuh"

#include "common/cuda/Interp2D.cuh"
#include "cuda/pagani/quad/GPUquad/Workspace.cuh"
#include "common/cuda/Volume.cuh"

#include "common/integration_result.hh"
#include "cuda/pagani/demos/new_time_and_call.cuh"

using namespace quad;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

constexpr double EPSABS = 1.0e-40;

int
main(int argc, char** argv)
{
  TYPE epsrel = 2.560e-09;
  constexpr int ndim = 8;

  Workspace<double, ndim> pagani;
  BoxIntegral8_22 integrand;
  Volume<double, ndim> vol;
  double true_value = 1495369.283757217694;

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  auto t0 = std::chrono::high_resolution_clock::now();
  numint::integration_result result = pagani.integrate<BoxIntegral8_22>(
    integrand, epsrel, EPSABS, vol);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

  printf("%.20f +- %.20f epsrel:%e, nregions:%lu flag:%i time:%f error:%.17f, "
         "ratio:%.17f\n",
         result.estimate,
         result.errorest,
         epsrel,
         result.nregions,
         result.status,
         dt.count(),
         abs(true_value - result.estimate),
         result.errorest / MaxErr(result.estimate, epsrel, EPSABS));
  return 0;
}
