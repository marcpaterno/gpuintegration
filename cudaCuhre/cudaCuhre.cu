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

constexpr double EPSABS = 1e-12;

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
	
  constexpr int ndim = 5;
  
  Cuhre<TYPE, ndim> cuhre(argc, argv, 0, verbose, numDevices);
  absCosSum5DWithoutK integrand;
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
	

  double highs[ndim] = {1., 1., 1., 1., 1.};
  double lows[ndim]  = {0., 0., 0., 0., 0.};
  Volume<double, ndim> vol(lows, highs);
  double true_value = 0.6371054;

  double tau_arr[] = {
      3.87497099e+00, 2.87383279e+00, 2.89974546e+00, 8.05299747e-01,
      5.82079679e-01, 4.25342329e-01, 3.16631643e-01, 2.31794166e-01,
      1.87431347e-01, 1.68846430e-01, 1.48897918e-01, 1.23155603e-01,
      1.17395703e-01, 9.65549883e-02, 8.12248716e-02, 6.77644921e-02,
      6.49962883e-02, 7.01815650e-02, 8.45228377e-02, 8.64134443e-02,
      2.47900879e-01, 1.50913981e+00, 3.99000196e+00, 2.99297068e+00,
      2.84870635e+00, 9.02123171e-01, 6.53503944e-01, 3.78202533e-01,
      2.93409763e-01, 2.04011587e-01, 1.70537624e-01, 1.59430876e-01,
      1.35222389e-01, 1.14403226e-01, 9.94072894e-02, 8.86215195e-02,
      7.85378863e-02, 5.96406985e-02, 5.21167369e-02, 4.53305139e-02,
      4.23910526e-02, 4.52286826e-02, 7.12116841e-02, 8.48740567e-02,
      3.99122151e+00, 2.99313873e+00, 1.24303891e+00, 7.65056083e-01,
      5.73145734e-01, 3.80157610e-01, 2.65550203e-01, 2.19892966e-01,
      1.75737759e-01, 1.51955192e-01, 1.29324115e-01, 1.16938673e-01,
      1.01643380e-01, 8.52507286e-02, 7.32086671e-02, 6.14961380e-02,
      5.45104285e-02, 4.75008980e-02, 4.52755127e-02, 4.21972860e-02,
      4.64843157e-02, 1.01169674e-01, 3.88854179e+00, 2.90137190e+00,
      1.09617428e+00, 7.18584529e-01, 5.28846654e-01, 3.61263803e-01,
      2.54901596e-01, 1.89673381e-01, 1.61594277e-01, 1.49463022e-01,
      1.30428339e-01, 1.10958353e-01, 9.64769254e-02, 8.81449968e-02,
      7.75405786e-02, 7.16844329e-02, 6.79409697e-02, 5.76577599e-02,
      4.92408225e-02, 3.67939006e-02, 2.71724239e-02, 2.13570930e-02,
      3.99031276e+00, 2.94701818e+00, 1.24592536e+00, 7.40921522e-01,
      5.08269967e-01, 3.63841999e-01, 2.67205308e-01, 1.92099780e-01,
      1.57416268e-01, 1.48313111e-01, 1.25128868e-01, 1.02699334e-01,
      9.52075683e-02, 8.44257008e-02, 7.10884458e-02, 6.04151374e-02,
      5.41311947e-02, 4.53025254e-02, 4.20917211e-02, 3.90545806e-02,
      5.22777255e-02, 6.61080543e-02};

  double cInterpC[] = {0.1, 0.15, 0.2, 0.25, 0.3};

  double cInterpR[] = {1.,         3.,          5.,          7.,
                        9.,         12.,         15.55555534, 20.,
                        24.,        26.11111069, 30.,         36.66666412,
                        40.,        47.22222137, 57.77777863, 68.33332825,
                        78.8888855, 89.44444275, 100.,        120.,
                        140.,       160.};
		
  //size_t rows = 22;
  //size_t cols = 5;
  //Interp2D<double> tau(cInterpC, cInterpR, tau_arr, cols, rows);
  //double gsl_val = 0.;
 // testKernel<<<1,1>>>(tau);
  
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  auto t0 = std::chrono::high_resolution_clock::now();
  cuhreResult result = cuhre.integrate<absCosSum5DWithoutK>(
    integrand, epsrel, EPSABS, &vol, outfileVerbosity, _final, phase_I_type);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
			
  printf("%.20f +- %.20f epsrel:%f, nregions:%lu flag:%i time:%f error:%.17f, ratio:%.17f failed phase2 blocks:%i\n",
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
