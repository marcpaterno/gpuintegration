#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include "function.cuh"
#include "quad/quad.h"
#include "quad/util/cudaUtil.h"
#include <iomanip>
#include "quad/GPUquad/Cuhre.cuh"
#include "quad/util/Volume.cuh"

using namespace quad;

template <typename F>
bool
time_and_call(F integrand, double epsrel, double true_value, char const* algname, std::stringstream& outfile,int _final= 0)
{
  using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-40;

  double lows[] =  {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol(lows, highs);
  quad::Cuhre<double, ndim> alg(0, nullptr, 0, 0, 1);
  
  std::string id 			= "BoxIntegral8_22";
  int outfileVerbosity  	= 0;
  int phase_I_type 			= 0; // alternative phase 1
  int appendMode			= 1;
  
  auto const t0 = std::chrono::high_resolution_clock::now();
  auto const result = alg.integrate<BoxIntegral8_22>(integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(result.value - true_value);
  bool const good = (result.status == 0 || result.status == 2);
  printf("good:%i result.status:%i\n", good, result.status);
  printf("%.15f +- %.15f epsrel:%e final:%i nregions:%lu flag:%i time:%f\n", result.value, result.error, epsrel, _final, result.nregions, result.status, dt.count());
  
  FinalDataPrint(outfile, id, true_value, epsrel, epsabs, result.value, result.error,
					result.nregions, result.status, _final, dt.count(), id + ".csv", appendMode);
	
  return good;
}

int main(){
	double epsrel  = 1.0e-3;  // starting error tolerance.	
	double lows[]  =  {0., 0., 0., 0., 0., 0., 0., 0.};
    double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
	
	constexpr int ndim 	= 8;
	int _final 			= 0;
	double true_value 	= 1495369.283757217694;
	quad::Volume<double, ndim> vol(lows, highs);
    quad::Cuhre<double, ndim> cuhre(0, nullptr, 0, 0, 1);
	
	std::stringstream outfile;
	BoxIntegral8_22 integrand;
	
	
	printf("Testing final = 1\n");
	_final = 1;

	while (time_and_call(integrand, epsrel, true_value, "gpucuhre", outfile,  _final) == true && epsrel>=1e-8) {
      epsrel /= 5.0;
	}
	
	_final = 0;
	epsrel = 1.0e-3;
	
	outfile<<"id, value, epsrel, epsabs, estimate, errorest, regions, converge, final, total_time"<<std::endl; 
	printf("Testing final = 1\n");
	while (time_and_call(integrand, epsrel, true_value, "gpucuhre", outfile, _final) == true && epsrel>=1e-8) {
      epsrel /= 5.0;
	}
	
	printf("Here\n");
}