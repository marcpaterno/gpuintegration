#include "cudaCuhre/demos/function.cuh"
#include "cudaCuhre/demos/demo_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

namespace detail{
    class GENZ_6_6D {
    public:
      __device__ __host__ double
      operator()(double u, double v, double w, double x, double y, double z)
      {
          if(z > .9 || y > .8 || x > .7 || w > .6 || v >.5 || u > .4)
              return 0.;
          else
              return exp(10*z + 9*y + 8*x + 7*w + 6*v + 5*u)/*/1.5477367885091207413e8*/;
      }
    };
}

int
main()
{
  double epsrel =  1e-3; // starting error tolerance.
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value =  1.5477367885091207413e8;
  detail::GENZ_6_6D integrand;

  constexpr int ndim = 6;
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 4;
  PrintHeader();
  int heuristics[3] = {0,4,2};
  
  for(int i=4; i>=0; i--){
      configuration.heuristicID = heuristics[i];
      epsrel = 1.0e-3;
      while (cu_time_and_call<detail::GENZ_6_6D, ndim>("GENZ_6_2D",
                           integrand,
                           epsrel,
                           true_value,
                           "gpucuhre",
                           std::cout,
                           configuration) == true &&
             epsrel >= epsrel_min) {
        epsrel /= 5.0;
      }
  }
}
