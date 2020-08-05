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
time_and_call(F integrand, double epsrel, double true_value, std::string id , std::stringstream& outfile,	int _final= 0)
{
  using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-40;
	
  double lows[] =  {0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1.};
  
  constexpr int ndim = 5;
  quad::Volume<double, ndim> vol(lows, highs);
  quad::Cuhre<double, ndim> alg(0, nullptr, 0, 0, 1);
	
  int outfileVerbosity  	= 0;
  int phase_I_type 			= 0; // alternative phase 1
  int appendMode			= 1;
  
  auto const t0 = std::chrono::high_resolution_clock::now();
  cuhreResult const result = alg.integrate<absCosSum5DWithoutKPlus1>(integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  bool good = false;
  
  if(result.status == 0 || result.status == 2){
	  good = true;
  }
  
  outfile.precision(20);
  std::cout.precision(17);
  FinalDataPrint(outfile, id, true_value, epsrel, epsabs, result.value, result.error,
					result.nregions, result.status, _final, dt.count(), "absCosSumPlus1.csv", appendMode);
					
  outfile.str(""); //clear string stream
  std::cout<<id<<",\t"
		   <<true_value<<",\t"
			<<epsrel<<",\t\t\t"
			<<epsabs<<",\t"
			<<result.value<<",\t"
			<<result.error<<",\t"
			<<result.nregions<<",\t"
			<<result.status<<",\t"
			<<_final<<",\t"
			<<dt.count()<<std::endl;
 
  return good;
}

int main(){
	double epsrel  = 1.0e-3;  // starting error tolerance.	
	double true_value 	= 0.9999262476619335  ;
	std::stringstream outfile;
	absCosSum5DWithoutKPlus1 integrand;
	//std::cout<<"id,\t value,\t epsrel,\t epsabs,\t estimate,\t errorest,\t regions,\t converge,\t final,\t total_time"<<std::endl; 
	outfile<<"id,\t\t\t\t\t value,\t epsrel,\t epsabs,\t estimate,\t errorest,\t regions,\t converge,\t final,\t total_time"<<std::endl; 
	int _final = 1;
	while (time_and_call(integrand, epsrel, true_value, "pdcuhre_f1", outfile, _final) == true && epsrel>=1e-8) {
		epsrel /= 5.0;
	}
	
	_final = 0;
	epsrel = 1.0e-3;
	
	
	while (time_and_call(integrand, epsrel, true_value, "pdcuhre_f0", outfile, _final) == true && epsrel >= 2.56e-09) {
      epsrel = epsrel>=1e-6 ? epsrel / 5.0 : epsrel / 2.0;
	}
	
	printf("Here\n");
}