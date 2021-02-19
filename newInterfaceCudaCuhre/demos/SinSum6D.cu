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
  double const epsrel_min = 1.0e-12;
  double true_value = -49.165073;
  SinSum6D integrand;
  constexpr int ndim = 6;
  
  double lows[]  = {0.,  0.,  0.,  0., 0., 0.};	//original bounds
  double highs[] = {10., 10., 10., 10.,10.,10.};
  quad::Volume<double, ndim> vol(lows, highs);
  PrintHeader();
  
  while (cu_time_and_call<SinSum6D, ndim>("SinSum6D",
                       integrand,
                       epsrel,
                       true_value,
                       std::cout,
                       &vol) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }
}
