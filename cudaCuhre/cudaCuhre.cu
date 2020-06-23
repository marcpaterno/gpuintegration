#include <chrono>
#include <cmath>

#include "function.cuh"
#include "quad/quad.h"
#include "quad/util/cudaUtil.h"
#include <iomanip>

#include "quad/GPUquad/Cuhre.cuh"
#include "quad/util/Volume.cuh"

using namespace quad;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

constexpr double EPSABS = 1e-12;

class GENZ_1_8d{
	
	public:
	double normalization;
	double integral;
	__device__ __host__
	GENZ_1_8d(){
		integral = (1./315.) * sin(1.) * sin(3./2.) * sin(2.) * sin (5./2.) * sin(3.) *
                        sin(7./2.) * sin(4.) * (sin(37./2.) - sin(35./2.));
		normalization = 1./integral;
						
	}
	__device__ __host__ double
	operator()(double s, double t, double u, double v,
                 double w, double x, double y, double z){
		return normalization * cos(s + 2.*t + 3.*u + 4.*v + 5.*w + 6.*x + 7.*y + 8.*z);			 
	}
};



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
	
  TYPE epsrel = 4e-5;
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
		
	//Test integrand;
	absCosSum5D integrand;
	
	//double highs[ndim] = {1, 1, 1, 1, 1, 1, 1, 1};
    //double lows[ndim] =  {0, 0, 0, 0, 0, 0, 0, 0};
	double highs[ndim] = {1, 1, 1, 1, 1};
    double lows[ndim] =  {0, 0, 0, 0, 0};
    Volume<double, ndim> vol(lows, highs);
	using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
	
    /*cuhre.integrate<GENZ_1_8d>(&integrand, epsrel, EPSABS, integral, error, nregions, neval, &vol);*/
    auto t0 = std::chrono::high_resolution_clock::now();
	cuhreResult result = cuhre.integrate<absCosSum5D>(integrand, epsrel, EPSABS, &vol);
	MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
	std::cout<< result.value <<"\t"<< result.error <<"\t"<<result.nregions<<std::endl;
	std::cout<<"Time in ms:"<< dt.count()<<std::endl;
  return 0;
}
