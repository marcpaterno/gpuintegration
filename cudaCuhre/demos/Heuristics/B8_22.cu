#include "demos/function.cuh"
#include "cudaCuhre/demos/demo_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

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
            return pow(sum, s / 2)/1495369.283757217694;
          }
    };
}

int
main()
{
  double epsrel = 3.2e-7;//1.0e-3; // starting error tolerance.
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value = 1.0;
  constexpr int ndim = 8;
  detail::BoxIntegral8_22 integrand;
    
  Config configuration;
  configuration.outfileVerbosity = 0;
  int heuristics[3] = {0, 2,4};
  
  PrintHeader();
  for(int i=2; i>=0; i--){
      epsrel = 1.0e-3;
      configuration.heuristicID = heuristics[i];
      while (cu_time_and_call<detail::BoxIntegral8_22, ndim>("B8_22",
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
