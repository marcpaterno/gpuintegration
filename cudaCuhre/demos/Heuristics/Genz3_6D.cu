#include "cudaCuhre/demos/function.cuh"
#include "cudaCuhre/demos/demo_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

namespace detail{
    class GENZ_3_6D {
        public:
          __device__ __host__ double
          operator()(double x, double y, double z, double w, double v, double u)
          {
            return pow(1+6*u+5*v+4*w+3*x+2*y+z, -7);
          }
    };
}

int
main()
{
  double epsrel = 1.0e-3; // starting error tolerance.
  double const epsrel_min = 1.024e-10;
  double true_value = 7.1790160638199853886e-7;
  detail::GENZ_3_6D integrand;
 
  Config configuration;
  configuration.outfileVerbosity = 1;
  constexpr int ndim = 6;
  PrintHeader();
  
  int heuristics[3] = {0,4,2};
  for(int i=2; i>=0; i--)
  {
      epsrel = 1.e-3;
      configuration.heuristicID = heuristics[i];
      while (cu_time_and_call<detail::GENZ_3_6D, ndim>("GENZ_3_3D",
                           integrand,
                           epsrel,
                           true_value,
                           "gpucuhre",
                           std::cout,
                           configuration) == true &&
             epsrel > epsrel_min) {
        epsrel /= 5.0;
      }
  }
}
