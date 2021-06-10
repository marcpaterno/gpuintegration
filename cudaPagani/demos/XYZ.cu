#include "demo_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

class XYZ {
    public:
    __device__ __host__ double
    operator()(double x, double y, double z)
    {
        return x*y*z;
    }
};   

int
main()
{
  double epsrel = 1.0e-3; // starting error tolerance.
  double true_value = .125;
  double const epsrel_min = 1.0240000000000002e-10;
  XYZ integrand;
  constexpr int ndim = 3;
  
  Config configuration;
  configuration.outfileVerbosity = 0;

  PrintHeader();
  while (cu_time_and_call<XYZ, ndim>("XYZ",
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