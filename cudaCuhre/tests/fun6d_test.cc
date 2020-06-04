#include "cubacpp/cuhre.hh"
#include "cuba.h"

//#include "../cudaCuhre/quad/quad.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

// From Mathematica 12.1 Integrate, symbolic integration over unit hypercube.
// This is the multilplier that gives fun6 an integrated value of 1 over the
// unit hypercube/
double const fun6_normalization = 12.0/(7.0 - 6 * std::log(2.0) * std::log(2.0) + std::log(64.0));

double fun6(double u, double v, double w, double x, double y, double z)
{
  return fun6_normalization * (u * v + (std::pow(w, y) * x * y)/(1+u) + z*z);
}

template <typename ALG, typename F>
void
time_and_call(ALG const& a, F f, double epsrel, double correct_answer, char const* algname)
{
  using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-16;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto res = a.integrate(f, epsrel, epsabs);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double absolute_error = std::abs(res.value - correct_answer);
  bool const good = (res.status == 0);
  std::cout << std::scientific
           << algname << '\t'
            << epsrel << '\t';
  if (good) {
    std::cout << res.value << '\t'
              << res.error << '\t'
              << absolute_error << '\t';
  } else {
    std:: cout << "NA\tNA\tNA\t";
  }
  std::cout << res.neval << '\t'
            << res.nregions << '\t'
            << dt.count()
            << std::endl;
}


int main()
{
  cubacores(0, 0); // turn off the forking use in CUBA's CUHRE.
  unsigned long long constexpr maxeval = 1000 * 1000 * 1000;

  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;
  
  double epsrel = 1.0e-3;
  for (int i = 1; i <= 3; ++i)
  {
    epsrel /= 10;
    time_and_call(cuhre, fun6, epsrel, 1.0, "cuhre");
  }
}
