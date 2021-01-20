#include "demos/function.cuh"
#include "cudaCuhre/demos/demo_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;


namespace detail{
    class BoxIntegral8_15 {
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

        double s = 15;
        double sum = 0;
        sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
              pow(m, 2) + pow(n, 2) + pow(o, 2);
        return pow(sum, s / 2);
      }
    };    
}

int
main()
{
  double epsrel = 3.20000000000000060e-07;//1.0e-3; // starting error tolerance.
  double true_value = 8879.851175413485;
  double const epsrel_min = 1.0240000000000002e-10;
  detail::BoxIntegral8_15 integrand;
  constexpr int ndim = 8;
  
  constexpr int alternative_phase1 = 0;
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 4;
  
  PrintHeader();
  int heuristics[3] = {0, 2,4};
  
  PrintHeader();
  for(int i=2; i>=0; i--){
      epsrel = 1.0e-3;
      configuration.heuristicID = heuristics[i];
      while (cu_time_and_call<detail::BoxIntegral8_15, ndim>("B8_15",
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
