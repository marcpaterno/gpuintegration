#include <chrono>
#include <cmath>
#include <iomanip>
#include <mpi.h>

#include "function.cuh"
#include "quad/quad.h"
#include "quad/util/cudaUtil.h"

#include "quad/GPUquad/Cuhre.cuh"
//#include "quad/util/Volume.cuh"
#include "demo_utils.h"

using namespace quad;
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

double
integrand2(double v, double w, double x, double y, double z)
{
  return fabs(cos(4. * v + 5. * w + 6. * x + 7. * y + 8. * z) / 0.6371054);
};

int
main()
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  constexpr int ndim = 5;
  Cuhre<double, ndim> cuhre(0, 0, 0, 0, 1);
  absCosSum5D integrand;
  double highs[ndim] = {1, 1, 1, 1, 1};
  double lows[ndim] = {0, 0, 0, 0, 0};
  Volume<double, ndim> vol(lows, highs);
  double epsrel = 4e-5;
  // double epsrel   = 8e-6;
  double epsabs = 1e-12;
  auto t0 = std::chrono::high_resolution_clock::now();
  cuhreResult result =
    cuhre.integrate<absCosSum5D>(integrand, epsrel, epsabs, &vol, 0);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  std::cout << result.estimate << "\t" << result.errorest << "\t"
            << result.nregions << std::endl;
  std::cout << "Time in ms:" << dt.count() << std::endl;
  return 0;
}
