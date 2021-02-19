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
  double const epsrel_min = 1.024e-10;
  double true_value = 1.286889807581113e+13;
  GENZ_2_6D integrand;
  constexpr int ndim = 6;
  
  PrintHeader();
  while (cu_time_and_call<GENZ_2_6D, ndim>("GENZ_2_6D",
                       integrand,
                       epsrel,
                       true_value,
                       std::cout) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
  }
}
