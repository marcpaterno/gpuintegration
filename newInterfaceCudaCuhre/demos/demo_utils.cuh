#include <mpi.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "cudaCuhre/demos/function.cuh"
#include "newInterfaceCudaCuhre/quad/quad.h"
#include "newInterfaceCudaCuhre/quad/util/cudaUtil.h"
#include "quad/GPUquad/GPUquad.cu"
#include "newInterfaceCudaCuhre/quad/util/Volume.cuh"
#include "nvToolsExt.h" 

//#ifdef USE_NVTX

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

void PrintHeader(){
    std::cout << "id, value, epsrel, epsabs, estimate, errorest, regions, "
             "status, total_time\n";
}

template <typename F, int ndim>
bool
cu_time_and_call(std::string id,
                F integrand,
                double epsrel,
                double true_value,
                std::ostream& outfile,
                quad::Volume<double, ndim>* vol = nullptr)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-40;
  int key = 0;
  int numDevices = 1;
  int verbose = 0;
  int pargc = 0;
  char **pargv = nullptr;
  quad::GPUcuhre<double, ndim> alg(pargc, pargv, key, verbose, numDevices);
  
  double estimate = 0.;
  double errorest = 0.;
  size_t nregions = 0;
  size_t neval = 0;
  
  auto const t0 = std::chrono::high_resolution_clock::now();
  int errorFlag = 1;
  errorFlag = alg.integrate(integrand, epsrel, epsabs, estimate, errorest, nregions, neval, vol);                          
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  
 std::string hID;
 outfile.precision(17);
 outfile << std::fixed  << std::scientific 
          << id << ","
          << true_value << ","
          << epsrel << "," 
          << epsabs << "," 
          << estimate << ","
          << errorest << "," 
          << nregions << "," 
          << errorFlag << "," 
          << dt.count() 
          << std::endl;
  return !errorFlag;
}