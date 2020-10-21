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

template <typename ALG, typename F>
bool
time_and_call_alt(ALG const& a, F f, double epsrel, double correct_answer, std::string algname)
{
  using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-16;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto res = a.integrate(f, epsrel, epsabs);
  
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  int _final = 0;
  std::cout.precision(15); 
  std::cout<<algname<<","
		   <<std::to_string(correct_answer)<<","
			<<epsrel<<","
			<<epsabs<<","
			<<std::to_string(res.value)<<","
			<<std::to_string(res.error)<<","
			<<res.nregions<<","
			<<res.status<<","
			<<_final<<","
			<<dt.count()<<std::endl;
  if(res.status == 0)
	return true;
  else
	return false;
}

double 
GENZ_2_6D(double x, double y/*, double z, double k, double l, double m*/)
  {
	double a = 50.;
    double b = .5;
	
    double term_1 = 1./((1./pow(a,2)) + pow(x- b, 2));
    double term_2 = 1./((1./pow(a,2)) + pow(y- b, 2));
	//double term_3 = 1./((1./pow(a,2)) + pow(z- b, 2));
	//double term_4 = 1./((1./pow(a,2)) + pow(k- b, 2));
	//double term_5 = 1./((1./pow(a,2)) + pow(l- b, 2));
	//double term_6 = 1./((1./pow(a,2)) + pow(m- b, 2));
	
    double val  = term_1 * term_2 /** term_3 * term_4 * term_5 * term_6*/;
	return val;
  }



int main()
{
  cubacores(0, 0); // turn off the forking use in CUBA's CUHRE.
  unsigned long long constexpr maxeval = 1000 * 1000 * 1000;
  double const epsrel_min = 1.024e-10;
  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;

  cout<<"id, value, epsrel, epsabs, estimate, errorest, regions, converge, final, total_time\n";
  
  double epsrel = 1.0e-3;
  double true_value = pow(153.0818, 6);
  while(epsrel >= epsrel_min && time_and_call_alt(cuhre, GENZ_2_6D, epsrel, true_value, "dc_f0") == true)
  {
     epsrel /= 5.0;
  }
  
  int verbose = 0;
  int _final = 4;
  cuhre.flags = verbose | _final;
  epsrel = 1.0e-3;
  while(epsrel >= epsrel_min && time_and_call_alt(cuhre, GENZ_2_6D, epsrel, true_value, "dc_f1") == true)
  {
      epsrel /= 5.0;
  }
  return 0;
}
