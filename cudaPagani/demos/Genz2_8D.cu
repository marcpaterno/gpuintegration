#include "function.cuh"
#include "demo_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

namespace detail{
    class GENZ_3_8D {
        public:
          __device__ __host__ double
          operator()(double x, double y, double z, double w, double v, double u, double t, double s)
          {
            return pow(1+8*s+7*t+6*u+5*v+4*w+3*x+2*y+z, -9)/2.2751965817917756076e-10;
          }
    };
}

int
main()
{
  double epsrel = 2.56e-09;//1.e-3; // starting error tolerance.
  double const epsrel_min = 1.0e-13;
  double true_value = 3.015702399795044e+17;
  GENZ_2_8D integrand;
  constexpr int ndim = 8;
  Config configuration;
  configuration.outfileVerbosity = 0;
  //configuration.heuristicID = 0;
  PrintHeader();
  
  while (cu_time_and_call<GENZ_2_8D, ndim>("GENZ_2_8D",
                           integrand,
                           epsrel,
                           true_value,
                           "gpucuhre",
                           std::cout,
                           configuration) == true &&
             epsrel > epsrel_min) {
    epsrel /= 5.0;
    break;
   }
}
