#include "demo_utils.cuh"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

int
main()
{
  double epsrel = 1.0e-3; // starting error tolerance.
  double const epsrel_min = 1.0e-13;
  double true_value = 23434.04;
  GENZ_2_2D integrand;
  constexpr int ndim = 2;
  
  PrintHeader();
  while (cu_time_and_call<GENZ_2_2D, ndim>("GENZ_2_2D",
                       integrand,
                       epsrel,
                       true_value,
                       std::cout) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }
}
