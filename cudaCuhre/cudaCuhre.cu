#include <chrono>
#include <cmath>

#include "function.cuh"
#include "quad/quad.h"
#include "quad/util/cudaUtil.h"
#include <iomanip>

#include "quad/GPUquad/Cuhre.cuh"
#include "quad/util/Volume.cuh"
#include "quad/GPUquad/Interp2D.cuh"

//#include "../y3_cluster_cpp/utils/datablock.hh"
//#include "../y3_cluster_cpp/utils/interp_2d.hh"
//#include "../y3_cluster_cpp/utils/primitives.hh"

using namespace quad;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

constexpr double EPSABS = 1.0e-40;

/*__global__ void
testKernel(Interp2D<double> tau){
	printf("tau:%f\n", tau(.13, 2.1));
}*/

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
	
  TYPE epsrel = 3.2e-7;
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
  GENZ_3_8D integrand;
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
	

  double highs[ndim] = {1., 1., 1., 1., 1., 1., 1., 1.};
  double lows[ndim]  = {0., 0., 0., 0., 0., 0., 0., 0.};
  Volume<double, ndim> vol(lows, highs);
  double true_value = 2.2751965817917756076e-10;

  //size_t rows = 22;
  //size_t cols = 5;
  //Interp2D<double> tau(cInterpC, cInterpR, tau_arr, cols, rows);
  //double gsl_val = 0.;
 // testKernel<<<1,1>>>(tau);
  
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  auto t0 = std::chrono::high_resolution_clock::now();
  cuhreResult result = cuhre.integrate<GENZ_3_8D>(
    integrand, epsrel, EPSABS, &vol, outfileVerbosity, _final, phase_I_type);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
			
  printf("%.20f +- %.20f epsrel:%e, nregions:%lu flag:%i time:%f error:%.17f, ratio:%.17f failed phase2 blocks:%i\n",
         result.estimate,
         result.errorest,
         epsrel,
         result.nregions,
         result.status,
         dt.count(),
		 abs(true_value-result.estimate),
		 result.errorest/MaxErr(result.estimate, epsrel, EPSABS),
		 result.phase2_failedblocks);
  return 0;
}
