#include "cuba.h"
#include "cubacpp/cuhre.hh"

#include "demo_utils.h"

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
time_and_call_alt(ALG const& a, F f, double epsrel, std::string algname)
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
			<<epsrel<<","
			<<epsabs<<","
			<<std::scientific<<res.value<<","
			<<std::scientific<<res.error<<","
			<<res.nregions<<","
			<<res.status<<","
			<<_final<<","
			<<dt.count()<<std::endl;
			
   /*printf("%s, %.20f, %e, %e, %.20f, %.20f, %i, %i, %i, %.15f\n", id, 
																 true_value,
																 epsrel,
																 epsabs, 
																 result.estimate,
																 result.errorest,
																 result.nregions, 
																 result.status,
																 _final,
																 dt.count());*/
  if(res.status == 0)
	return true;
  else
	return false;
}

double absCosSum5DWithoutK(double v, double w, double x, double y, double z)
{
    return fabs(cos(4. * v + 5. * w + 6. * x + 7. * y + 8. * z));
}

int main()
{
  cubacores(0, 0); // turn off the forking use in CUBA's CUHRE.
  unsigned long long constexpr maxeval = 1000 * 1000 * 1000;

  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;
  double const epsrel_min = 1.0e-10;
  cout<<"id,\t value,\t epsrel,\t epsabs,\t estimate,\t errorest,\t regions,\t converge,\t final,\t total_time\n";
	
  //double epsrel = 1.0e-3;
  //while(time_and_call_alt(cuhre, absCosSum5DWithoutK, epsrel, "dc_f0") == true && epsrel >= epsrel_min)
  {
  //  epsrel = epsrel >= 1e-6 ? epsrel / 5.0 : epsrel / 2.0;
  }
  
  cuhre.flags = 4;
  double epsrel = 1.0e-3;
  while(time_and_call_alt(cuhre, absCosSum5DWithoutK, epsrel, "dc_f1") == true && epsrel >= epsrel_min)
  {
     epsrel /= 5.0;
  }
  return 0;
}
