#include "cudaCuhre/integrands/sig_miscent.cuh" 
#include "quad/GPUquad/Cuhre.cuh"
#include <iostream>
#include <chrono>	

template <typename F>
bool
time_and_call(std::string id,
              F integrand,
              double epsrel,
              double true_value,
              char const* algname,
              std::ostream& outfile,
              int _final = 0)
{
	//printf("time_and_call d_integrand Mor des cols:%lu\n", integrand.mor.sig_interp->_cols);
	//printf("inside time and call\n");
	//printf("time_and_call d_integrand Mor des cols:%lu\n", integrand.mor.sig_interp->_cols);
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-12;

  double lows[] =  {20., 5.,  5., .15,  29., 0., 0.};
  double highs[] = {30., 50., 50.,.75,  38., 1., 6.28318530718};

  constexpr int ndim = 7;
  quad::Volume<double, ndim> vol(lows, highs);
  int const key = 0;
  int const verbose = 0;
  int const numdevices = 1;
  quad::Cuhre<double, ndim> alg(0, nullptr, key, verbose, numdevices);
	
  int outfileVerbosity = 1;
  constexpr int phase_I_type = 0; // alternative phase 1

  auto const t0 = std::chrono::high_resolution_clock::now();
  
  cuhreResult const result = alg.integrate<F>(integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);
  
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(result.estimate - true_value);
  bool good = false;

  if (result.status == 0 || result.status == 2) {
    good = true;
  }
  outfile.precision(15);
  outfile << std::fixed << id << ",\t" << std::scientific << true_value << ",\t"
          << std::scientific << epsrel << ",\t\t\t" << std::scientific
          << epsabs << ",\t" << std::scientific << result.estimate << ",\t"
          << std::scientific << result.errorest << ",\t" << std::fixed
          << result.nregions << ",\t" << std::fixed << result.status << ",\t"
          << _final << ",\t" << dt.count() << std::endl;

  return good;
}

int main(){
	double const radius_ = 0x1p+0;
	double const zo_low_ = 0x1.999999999999ap-3;
	double const zo_high_ = 0x1.6666666666666p-2;
	
	int _final  = 1;
	double epsrel = 1.0e-3;
	integral<GPU> d_integrand;
	d_integrand.set_grid_point({zo_low_, zo_high_, radius_});
	double true_value = 0.;
	
	 while(time_and_call<integral<GPU>>("pdc_f1_latest",
			d_integrand,
   			epsrel,
   			true_value,
			"gpucuhre",
   			std::cout,
   			_final)){
		epsrel = epsrel/1.5;	
   }
	
	return 0;
}