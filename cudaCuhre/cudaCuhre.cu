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

/*__global__ void
testKernel(double* darray, double* sum){
	//FUNC2 test_integrand;
	//printf("GPU RESULT %a\n",    test_integrand(0x1.f4b65783633c5p-1, 0x1.f4b65783633c5p-1, 0x1p-1, 0x1p-1, 0x1p-1, 0x1p-1, 0x1p-1, 0x1.69350f939876p-6));
	//printf("GPU RESULT %.17f\n", test_integrand(0x1.f4b65783633c5p-1, 0x1.f4b65783633c5p-1, 0x1p-1, 0x1p-1, 0x1p-1, 0x1p-1, 0x1p-1, 0x1.69350f939876p-6));
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	double x = 0;
	for(int i=0; i<2048; i++){
		x++;
	}
	
	darray[index] = x;
	
	for(int i=0; i<2048; i++){
		
	}
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
  BoxIntegral8_22 integrand;
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 1; // alternative phase 1
	
  double highs[ndim] = {1., 1., 1., 1., 1., 1., 1., 1.};
  double lows[ndim]  = {0., 0., 0., 0., 0., 0., 0., 0.};
  Volume<double, ndim> vol(lows, highs);
  double true_value = 1495369.283757217694;
  
  //double *darray = 0;
  //double *dsum = 0;
  //cudaMalloc((double**)&darray, 32678*sizeof(double);
  //cudaMalloc((double**)&dsum, sizeof(double);
  //testKernel<<<32678, 256>>>(darray, dsum);
  //cudaDeviceSynchronize();
  //cudaFree(darray);
  
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  auto t0 = std::chrono::high_resolution_clock::now();
  cuhreResult result = cuhre.integrate<BoxIntegral8_22>(
    integrand, epsrel, EPSABS, &vol, outfileVerbosity, _final, phase_I_type);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
	
  std::cout.precision(17);
  /*std::cout<<true_value<<",\t"
			<<epsrel<<",\t"
			<<EPSABS<<",\t"
			<<result.value<<",\t"
			<<result.error<<",\t"
			<<result.nregions<<",\t"
			<<result.status<<",\t"
			<<_final<<",\t"
			<<dt.count()<<std::endl;*/
			
  printf("%.20f +- %.20f epsrel:%f, nregions:%lu flag:%i time:%f error:%.17f, ratio:%.17f actual error:%f\n",
         result.value,
         result.error,
         epsrel,
         result.nregions,
         result.status,
         dt.count(),
		 abs(true_value-result.value),
		 result.error/MaxErr(result.value, epsrel, EPSABS),
		 abs(true_value - result.value));
  return 0;
}
