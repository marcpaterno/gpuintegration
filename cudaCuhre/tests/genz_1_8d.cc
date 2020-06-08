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

double const integral = (1./315.) * sin(1.) * sin(3./2.) * sin(2.) * sin (5./2.) * sin(3.) *
                        sin(7./2.) * sin(4.) * (sin(37./2.) - sin(35./2.));
double const normalization = 1./integral;

double genz_1_8d(double s, double t, double u, double v,
                 double w, double x, double y, double z)
{
  return normalization * cos(s + 2.*t + 3.*u + 4.*v + 5.*w + 6.*x + 7.*y + 8.*z);
}

int main()
{
  cubacores(0, 0); // turn off the forking use in CUBA's CUHRE
  unsigned long long constexpr maxeval = 10 * 1000 * 1000 * 1000ull;

  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;

  std::cout << "alg\tepsrel\tvalue\terrorest\terror\tneval\tnregions\ttime\n";

  double epsrel = 1.0e-3;
  for (int i = 0; i <=6; ++i, epsrel /= 10.0)
  {
    // Quit the first time the integration does not converge.
    bool const rc = time_and_call(cuhre, genz_1_8d, epsrel, 1.0, "cuhre");
    if (!rc) break;
  }
}

