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
  double const epsrel_min = 1.0e-9;
  double true_value = 1.;
  Gauss9D integrand;
  
  double lows[]  = {-1., -1., -1., -1., -1., -1., -1., -1., -1.};
  double highs[] = { 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.};
  
  constexpr int ndim = 9;
  quad::Volume<double, ndim> vol(lows, highs);
  
  PrintHeader();
  while (cu_time_and_call<Gauss9D, ndim>("pdc_f1_depth",
                       integrand,
                       epsrel,
                       true_value,
                       std::cout,
                       &vol) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }
}
