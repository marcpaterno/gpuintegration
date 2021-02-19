#include "demo_utils.cuh"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

int
main()
{
  double epsrel =  1e-3;//3.20000000000000060e-07;//1e-3;//2.56000000000000067e-09;;//8.00000000000000133e-06;////
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value = 1.;
  Diagonal_ridge2D integrand;
  
  PrintHeader();
  
  constexpr int ndim = 2;
  double lows[] = {-1., -1.};	//original bounds
  double highs[] = {1., 1.};
  quad::Volume<double, ndim> vol(lows, highs);
  
  while (cu_time_and_call<Diagonal_ridge2D, ndim>("Diagonal_Ridge2D",
                                            integrand,
                                            epsrel,
                                            true_value,
                                            std::cout,
                                            &vol) == true &&
                                            epsrel >= epsrel_min) {
    epsrel /= 5.0;
    //break;
  }
}
