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
  double constexpr epsabs = 1.0e-40;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto res = a.integrate(f, epsrel, epsabs);
  
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  int _final = 0;
  
  std::cout.precision(20);
  
  std::cout<<algname<<","
		    <<std::scientific<<correct_answer<<","
			<<epsrel<<","
			<<epsabs<<","
			<<std::scientific<<res.value<<","
			<<std::scientific<<res.error<<","
			<<res.nregions<<","
			<<res.status<<","
			<<_final<<","
			<<dt.count()<<std::endl;
			
  if(res.status == 0)
	return true;
  else
	return false;
}

double absCosSum5DWithoutKPlus1(double v, double w, double x, double y, double z)
{
    return fabs(cos(4. * v + 5. * w + 6. * x + 7. * y + 8. * z) + 1.0);
}

int main()
{
  cubacores(0, 0); // turn off the forking use in CUBA's CUHRE.
  unsigned long long constexpr maxeval = 1000 * 1000 * 1000;

  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;
  double const epsrel_min = 1.0e-10;
  cout<<"id,\t value,\t epsrel,\t epsabs,\t estimate,\t errorest,\t regions,\t converge,\t final,\t total_time\n";
	
  double epsrel = 1.0e-3;
  double true_value = 0.9999262476619335;
 /* while(time_and_call_alt(cuhre, absCosSum5DWithoutKPlus1, epsrel, true_value, "dc_f0") == true && epsrel >= epsrel_min)
  {
    epsrel = epsrel >= 1e-6 ? epsrel / 5.0 : epsrel / 2.0;
  }
  **/
  cuhre.flags = 4;
  epsrel = 1.0e-3;
  while(time_and_call_alt(cuhre, absCosSum5DWithoutKPlus1, epsrel, true_value, "dc_f1") == true && epsrel >= epsrel_min)
  {
     epsrel /= 5.0;
  }
  return 0;
}
