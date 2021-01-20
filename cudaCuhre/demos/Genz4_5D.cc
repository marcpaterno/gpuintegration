#include "cuba.h"
#include "cubacpp/cuhre.hh"
#include "demo_utils.h"

//#include "../cudaCuhre/quad/quad.h"

#include <cmath>
#include <iostream>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

class GENZ_4_5D {
public:
	double
	operator()(double x, double y, double z, double w, double v){
		double alpha = 25.;
		double beta = .5;
		return exp(-1.0*(pow(alpha,2)*pow(x-beta, 2) + 
				         pow(alpha,2)*pow(y-beta, 2) +
				         pow(alpha,2)*pow(z-beta, 2) +
				         pow(alpha,2)*pow(w-beta, 2) +
				         pow(alpha,2)*pow(v-beta, 2))
				  );
	}
};

template <typename ALG, typename F>
bool
time_and_call_alt(ALG const& a, F f, double epsrel, double correct_answer, std::string algname)
{
  using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-20;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto res = a.integrate(f, epsrel, epsabs);
  
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  int _final = 1;
  std::cout.precision(17); 
  std::cout<<algname<<","
		   <<correct_answer<<"," << std::scientific 
			<<epsrel<<","
			<<epsabs<<","
			<<res.value<<","
			<<res.error<<","
			<<res.nregions<<","
			<<res.status<<","
			<<_final<<","
			<<dt.count()<<std::endl;
  if(res.status == 0)
	return true;
  else
	return false;
}

int main()
{
  cubacores(0, 0); // turn off the forking use in CUBA's CUHRE.
  unsigned long long constexpr maxeval = 1000 * 1000 * 1000;
  double const epsrel_min = 1.024e-10;
  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;
  
  cout<<"id, value, epsrel, epsabs, estimate, errorest, regions, converge, final, total_time\n";
  GENZ_4_5D integrand;
  double epsrel = 1.0e-3;
  double true_value = 1.79132603674879e-06;
  /*while(epsrel >= epsrel_min && time_and_call_alt<cubacpp::Cuhre, GENZ_4_5D>(cuhre, integrand, epsrel, true_value, "dc_f0") == true )
  {
    epsrel /= 5.0;
  }*/
  
  //int verbose = 0;
  //int _final = 4;
  cuhre.flags = 4;
  while(epsrel >= epsrel_min && time_and_call_alt<cubacpp::Cuhre , GENZ_4_5D>(cuhre, integrand, epsrel, true_value, "Genz4_5D") == true)
  {
      epsrel /= 5.0;
  }
  return 0;
}
