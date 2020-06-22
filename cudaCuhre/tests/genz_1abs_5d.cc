#include "cubacpp/cuhre.hh"
#include "cuba.h"
#include "demo_utils.h"

#include <cmath>
#include <iostream>

// From Mathematica 12.1 Integration, symbolic integration over unit hypercube.
// This is the multiplier that gives genz_1_8d an integrated value of 1 over the
// unit hypercube.
using std::sin;
using std::cos;
using std::abs;

double const integral = 6.371054e-01; // Value is approximate
double const normalization = 1./integral;

double genz_1abs_5d(double v,
                 double w, double x, double y, double z)
{
  return normalization * abs(cos(4.*v + 5.*w + 6.*x + 7.*y + 8.*z));
}

int main()
{
  cubacores(0, 0); // turn off the forking use in CUBA's CUHRE
  unsigned long long constexpr maxeval = 1 * 1000 * 1000 * 1000ull;

  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;

  std::cout << "alg\tepsrel\tvalue\terrorest\terror\tneval\tnregions\ttime\n";

  double epsrel = 1.0e-3;
  for (int i = 0; i < 3; ++i) {
    time_and_call(cuhre, genz_1abs_5d, epsrel, 1.0, "cuhre");
    epsrel = epsrel / 5.0;
  }
//   while(time_and_call(cuhre, genz_1abs_5d, epsrel, 1.0, "cuhre"))
//   {
//     epsrel = epsrel / 5.0;
//   }
}

