#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "function.cuh"
#include "quad/quad.h"
#include "quad/util/cudaUtil.h"
#include <iomanip>

#include "quad/GPUquad/GPUquad.cuh"

using namespace quad;

constexpr double EPSABS = 1e-12;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------


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

  TYPE epsrel = 1e-3;
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
  constexpr int ndim = 6;
  // for(int i = 0; i < 1; ++i)
  {
    TYPE integral = 0, error = 0;
    size_t nregions = 0, neval = 0;

    GPUcuhre<TYPE, ndim> cuhre(argc, argv, 0, verbose, numDevices);
    int errorFlag =
      cuhre.integrate(epsrel, EPSABS, integral, error, nregions, neval);
  }

  return 0;
}
