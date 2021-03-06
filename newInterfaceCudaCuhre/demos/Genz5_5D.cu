#include "demo_utils.cuh"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

namespace detail{
    class GENZ_5_5D {
    public:
      __device__ __host__ double
      operator()(double x, double y, double z, double k, double m)
      {
        double beta = .5;
        double t1 = -10.*fabs(x - beta) - 10.* fabs(y - beta) - 10.* fabs(z - beta) - 10.* fabs(k - beta) - 10.* fabs(m - beta);
        return exp(t1);
      }
    };
}

int
main()
{
  double epsrel = 1.0e-3; // starting error tolerance.
  double const epsrel_min = 1.024e-10;
  double true_value = 0.0003093636;
  detail::GENZ_5_5D integrand;
  
  constexpr int ndim = 5;
  PrintHeader();
  while (cu_time_and_call<detail::GENZ_5_5D, ndim>("GENZ_5_D",
                       integrand,
                       epsrel,
                       true_value,
                       std::cout) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
  }
}
