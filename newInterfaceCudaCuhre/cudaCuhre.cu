#include <mpi.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iomanip>  
#include "function.cu"
#include "quad/quad.h"
#include "quad/util/cudaUtil.h"

#include "quad/GPUquad/GPUquad.cu"
#include <chrono>

using namespace quad;

#define EPSABS 1e-40

namespace detail{
    class BoxIntegral8_22 {
        public:
          __device__ __host__ double
          operator()(double x,
                     double y,
                     double z,
                     double k,
                     double l,
                     double m,
                     double n,
                     double o)
          {
            double s = 22;
            double sum = 0;
            sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
                  pow(m, 2) + pow(n, 2) + pow(o, 2);
            return pow(sum, s / 2);
          }
    };
}

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

class Diagonal_ridge2D {
public:
  // correct answer: 1 on integration volume (-1,1)

  __device__ __host__ double
  operator()(double u, double v)
  {
    //if(u > 0.1 || v > 0.1)
   //     printf("%f, %f\n", u, v);
    double k = 0.01890022674239546529975841;
    return 4*k*u*u/(.01 + pow(u-v-(1./3.),2));
  }
};

int main(int argc, char **argv){
  CommandLineArgs args(argc, argv);
  double epsrel = 1.28e-8; 
  int verbose = 0;
  int numDevices = 1;
  constexpr int ndim = 2;
  double lows[] = {-1., -1.};	//original bounds
  double highs[] = {1., 1.};
  quad::Volume<double, ndim> vol(lows, highs);
  QuadDebugExit(args.DeviceInit());
  //detail::BoxIntegral8_22 integrand;
  Diagonal_ridge2D integrand;
  
  for(int i = 0; i < 1; ++i){
    double integral = 0, error = 0;
    size_t nregions = 0, neval = 0;
    GPUcuhre<double, ndim> cuhre(argc, argv, 0, verbose, numDevices);
    using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
    auto const t0 = std::chrono::high_resolution_clock::now();

    int errorFlag = cuhre.integrate<Diagonal_ridge2D>(integrand, epsrel, EPSABS, integral, error, nregions, neval, &vol);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
    printf("%.15e %.15e %lu %i\t", integral, error, nregions, errorFlag);
    std::cout << std::scientific << "Time:"<<dt.count() <<"\n";
  }

  return 0;
}
