#include <mpi.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iomanip>  
#include "function.cu"
#include "quad/quad.h"
#include "quad/util/cudaUtil.h"

#include "quad/GPUquad/GPUquad.cu"

using namespace quad;

#define EPSABS 1e-40

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose = false;

class TestFunc{
	__device__ 
	double
	operator()(double xx[], int ndim){
		double f = 0;
		for(int i=0; i<ndim; i++)
			f+=xx[i];
		return f;
	}
};

int main(int argc, char **argv){
  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");
  
  // Print usage
  if (args.CheckCmdLineFlag("help")){
    printf("%s "
            "[--e=<relative-error>] "
            "[--verbose=<0/1>] "
            "\n", argv[0]);
    exit(0);
  }
 
  TYPE epsrel = 1.28e-8; 
  if (args.CheckCmdLineFlag("e")){
    args.GetCmdLineArgument("e", epsrel);
  }
   // Verbose output
  int verbose = 0;
  if (args.CheckCmdLineFlag("verbose")){
    args.GetCmdLineArgument("verbose", verbose);
  }
 
  //Num Devices 
  int numDevices = 1;
  if (args.CheckCmdLineFlag("N")){
    args.GetCmdLineArgument("N", numDevices);
  }
  
  // Initialize device
  QuadDebugExit(args.DeviceInit());
 
  for(int i = 0; i < 1; ++i){
    TYPE integral = 0, error = 0;
    size_t nregions = 0, neval = 0;
    GPUcuhre<TYPE> *cuhre = new GPUcuhre<TYPE>(argc, argv, DIM, 0, verbose, numDevices);
    int errorFlag = cuhre->integrate(epsrel, EPSABS, integral, error, nregions, neval);
    //printf("%d\t%e\t%.10lf\t%.10f\t%ld\t%ld\t%d\n", DIM, epsrel, integral, error, nregions, neval, errorFlag);
    //std::cout << std::setprecision(9) << DIM << "\t" << epsrel << "\t" << std::setprecision(9)  << integral << "\t" << error << "\t" << nregions << "\t" << neval << "\t" << errorFlag  << std::endl;
   // MPI_Finalize(); 
    delete cuhre;
  }

  return 0;
}
