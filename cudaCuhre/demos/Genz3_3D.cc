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
  std::cout.precision(15); 
  std::cout<<algname<<","
		   <<correct_answer<<","
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

double GENZ_3_3D(double x, double y, double z)
{
    return pow(1+3*x+2*y+z, -4);
}


int main()
{
  cubacores(0, 0); // turn off the forking use in CUBA's CUHRE.
  unsigned long long constexpr maxeval = 1000 * 1000 * 1000;
  double const epsrel_min = 1.0e-12;
  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;

  cout<<"id, value, epsrel, epsabs, estimate, errorest, regions, converge, final, total_time\n";
  
  double epsrel = 1.0e-3;
  double true_value = 0.010846560846560846561;
   //while(time_and_call_alt(cuhre, B8_22, epsrel, true_value, "dc_f0") == true &&  epsrel >= epsrel_min)
  //{
   //  epsrel = epsrel >= 1e-6 ? epsrel / 5.0 : epsrel / 2.0;
  //}
  
  cuhre.flags = 4;
  epsrel = 1.0e-3;
  while(time_and_call_alt(cuhre, GENZ_3_3D, epsrel, true_value, "dc_f1") == true && epsrel >= epsrel_min)
  {
      epsrel /= 5.0;
  }
  return 0;
}
