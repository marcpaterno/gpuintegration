#include "cudaCuhre/integrands/sig_miscent.cuh"
#include "demo_utils.cuh"
#include <chrono>
#include <iostream>

int
main()
{
  double const radius_ = 0x1p+0;
  double const zo_low_ = 0x1.999999999999ap-3;
  double const zo_high_ = 0x1.6666666666666p-2;
  double epsrel = 1.0e-3;
  integral<GPU> d_integrand;
  d_integrand.set_grid_point({zo_low_, zo_high_, radius_});
  constexpr int ndim = 7;
  double true_value = 0.;
  double lows[] = {20., 5., 5., .15, 29., 0., 0.};	//original bounds
  double highs[] = {30., 50., 50., .75, 38., 1., 6.28318530718};
  quad::Volume<double, ndim> vol(lows, highs);
  
  Config configuration;
  configuration.outfileVerbosity = 1;
  int heuristics[3] = {0, 2, 4};
  PrintHeader();
  
  for(int i=2; i>=0; i--){
      epsrel = 1e-3;
      configuration.heuristicID = heuristics[i];
      while (cu_time_and_call<integral<GPU>>("pdc_f1_latest",
                                          d_integrand,
                                          epsrel,
                                          true_value,
                                          "gpucuhre",
                                          std::cout,
                                          configuration,
                                          &vol)) {
        break;
      }
  }

  return 0;
}