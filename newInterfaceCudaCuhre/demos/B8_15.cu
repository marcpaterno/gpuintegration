#include "demo_utils.cuh"
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
  double epsrel = 1.0e-3; // starting error tolerance.
  double true_value = 8879.851175413485;
  double const epsrel_min = 1.0240000000000002e-10;
  detail::BoxIntegral8_15 integrand;
  constexpr int ndim = 8;
  PrintHeader();
  
  while (cu_time_and_call<detail::BoxIntegral8_15, ndim>("B8_15",
                       integrand,
                       epsrel,
                       true_value,
                       std::cout) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
  }
}
