#include "cubacpp/cuhre.hh"
#include "cuba.h"
#include "demo_utils.h"
#include "fun6.cuh"

#include <iostream>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main()
{
  unsigned long long constexpr maxeval = 1000 * 1000 * 1000;

  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;

  std::cout << "alg\tepsrel\tvalue\terrorest\terror\tneval\tnregions\ttime\n";
	
  double epsrel = 1.0e-3;
  for (int i = 0; i <= 6; ++i, epsrel /= 10.0)
  {
    time_and_call(cuhre, fun6, epsrel, 1.0, "cuhre");
    
    // add call to GPU integration here.
    // Can this be done while keeping this a C++ (not CUDA) file?
    // If not, what should we be doing instead?
  }
}
