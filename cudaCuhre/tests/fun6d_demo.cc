#include "cubacpp/cuhre.hh"
#include "cuba.h"
#include "demo_utils.h"

//#include "../cudaCuhre/quad/quad.h"

#include <cmath>
#include <iostream>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

// From Mathematica 12.1 Integrate, symbolic integration over unit hypercube.
// This is the multilplier that gives fun6 an integrated value of 1 over the
// unit hypercube.
double const fun6_normalization = 12.0/(7.0 - 6 * std::log(2.0) * std::log(2.0) + std::log(64.0));

double fun6(double u, double v, double w, double x, double y, double z)
{
  return fun6_normalization * (u * v + (std::pow(w, y) * x * y)/(1+u) + z*z);
}


int main()
{
  cubacores(0, 0); // turn off the forking use in CUBA's CUHRE.
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
