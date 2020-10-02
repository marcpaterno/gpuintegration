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
  
  cubacpp::array<9> lows  = {-1., -1., -1., -1., -1., -1., -1., -1., -1.};
  cubacpp::array<9> highs = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
  cubacpp::integration_volume_for_t<F> vol(lows, highs);
  auto res = a.integrate(f, epsrel, epsabs, vol);
  
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double absolute_error = std::abs(res.value - correct_answer);
  bool const good = (res.status == 0);
  int converge = !good;
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

#define PI 3.14159265358979323844

double Gauss9D(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o,
			 double p)
{
	double sum = pow(x,2) + pow(y,2) + pow(z, 2) + pow(k, 2) + pow(l,2) + pow(m, 2) + pow(n, 2) + pow(o, 2) + pow(p,2);
	return exp(-1*sum/(2*pow(0.01,2)))*(1/pow(sqrt(2*PI)*0.01, 9));
}


int main()
{
  cubacores(0, 0); // turn off the forking use in CUBA's CUHRE.
  unsigned long long constexpr maxeval = 1000 * 1000 * 1000;
  double const epsrel_min = 1.0e-9;
  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;

  cout<<"id, value, epsrel, epsabs, estimate, errorest, regions, converge, final, total_time\n";
  
  double epsrel = 1.0e-3;
  double true_value = 1.0;
   while(time_and_call_alt(cuhre, Gauss9D, epsrel, true_value, "dc_f0") == true &&  epsrel >= epsrel_min)
  {
     epsrel = epsrel >= 1e-6 ? epsrel / 5.0 : epsrel / 2.0;
  }
  
  cuhre.flags = 4;
  epsrel = 1.0e-3;
  while(time_and_call_alt(cuhre, Gauss9D, epsrel, true_value, "dc_f1") == true && epsrel >= epsrel_min)
  {
      epsrel /= 5.0;
  }
  return 0;
}
