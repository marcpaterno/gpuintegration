#include "demo_utils.cuh"
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
            return pow(1+8*s+7*t+6*u+5*v+4*w+3*x+2*y+z, -9)/*/2.2751965817917756076e-10*/;
          }
    };
}

int
main()
{
  double epsrel = 1e-3;
  double const epsrel_min = 1.024e-10;
  double true_value = 2.2751965817917756076e-10;
  detail::GENZ_3_8D integrand;
  PrintHeader();
  
  constexpr int ndim = 8;
  
  while (cu_time_and_call<detail::GENZ_3_8D, ndim>("Genz3_8D",
                           integrand,
                           epsrel,
                           true_value,
                           std::cout) == true &&
             epsrel > epsrel_min) {
    epsrel /= 5.0;
   }

  
}
