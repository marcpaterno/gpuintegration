#include "cudaCuhre/integrands/scalarIntegrandSigma.cuh"
#include "demo_utils.cuh"
#include <chrono>
#include <iostream>

int
main()
{
  double const radius_ = 0.45;
  double const zt = .5;
  
  double lows[]  = {1.0, 32.64165641};
  double highs[] = {2.0, 33.33480359};
  constexpr int ndim = 2;
  quad::Volume<double, ndim> vol(lows, highs);  
  double epsrel = 1.33657182142986e-07;
  
  quad::Snapshotsim_ScalarIntegrand_Sigma<GPU> d_integrand;
  d_integrand.set_grid_point({zt, radius_});
  double true_value = 0.;
  double epsrel_min = 1.0e-10;
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 4;
  
  PrintHeader();
  while (cu_time_and_call<quad::Snapshotsim_ScalarIntegrand_Sigma<GPU>, ndim>("pdc_f1_latest",
                                      d_integrand,
                                      epsrel,
                                      true_value,
                                      "gpucuhre",
                                      std::cout,
                                      configuration,
                                      &vol)&&
         epsrel >= epsrel_min) {
    epsrel = epsrel / 1.5;
  }
  
  return 0;
}