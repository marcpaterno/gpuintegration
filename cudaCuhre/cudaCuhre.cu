#include <chrono>
#include <cmath>

#include "function.cuh"
#include "quad/quad.h"
#include "quad/util/cudaUtil.h"
#include <iomanip>

#include "quad/GPUquad/Cuhre.cuh"
#include "quad/util/Volume.cuh"

using namespace quad;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

constexpr double EPSABS = 1e-12;

int
main(int argc, char** argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);
  bool g_verbose = args.CheckCmdLineFlag("v");

  // Print usage
  if (args.CheckCmdLineFlag("help")) {
    printf("%s "
           "[--e=<relative-error>] "
           "[--verbose=<0/1>] "
           "\n",
           argv[0]);
    exit(0);
  }

  TYPE epsrel = 1.0e-3;
  if (args.CheckCmdLineFlag("e")) {
    args.GetCmdLineArgument("e", epsrel);
  }
  // Verbose output
  int verbose = 0;
  if (args.CheckCmdLineFlag("verbose")) {
    args.GetCmdLineArgument("verbose", verbose);
  }

  // Num Devices
  int numDevices = 1;
  if (args.CheckCmdLineFlag("N")) {
    args.GetCmdLineArgument("N", numDevices);
  }

  // Initialize device
  QuadDebugExit(args.DeviceInit());

  constexpr int ndim = 8;

  Cuhre<TYPE, ndim> cuhre(argc, argv, 0, verbose, numDevices);

  BoxIntegral8_22 integrand;
  int _final = 0;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
  // double true_value 	= .999926247661939;
  std::string id = "Genz_1abs_5d";

  double highs[ndim] = {1., 1., 1., 1., 1., 1., 1., 1.};
  double lows[ndim] = {0., 0., 0., 0., 0., 0., 0., 0.};
  Volume<double, ndim> vol(lows, highs);

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  auto t0 = std::chrono::high_resolution_clock::now();
  cuhreResult result = cuhre.integrate<BoxIntegral8_22>(
    integrand, epsrel, EPSABS, &vol, outfileVerbosity, _final, phase_I_type);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

  printf("%.15f +- %.15f epsrel:%f, nregions:%lu flag:%i time:%f\n",
         result.value,
         result.error,
         epsrel,
         result.nregions,
         result.status,
         dt.count());
  return 0;
}
