#include "cubacpp/cuhre.hh"
#include "cubacpp/vegas.hh"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

// From Mathematica 12.1 Integrate, symbolic integration over unit hypercube.
// This is the multilplier that gives fun6 an integrated value of 1 over the
// unit hypercube/
double const fun6_normalization =
  12.0 / (7.0 - 6 * std::log(2.0) * std::log(2.0) + std::log(64.0));

double
fun6(double u, double v, double w, double x, double y, double z)
{
  return fun6_normalization *
         (u * v + (std::pow(w, y) * x * y) / (1 + u) + z * z);
}

template <typename ALG, typename F>
void
time_and_call(ALG const& a,
              F f,
              double epsrel,
              double correct_answer,
              char const* algname)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-100;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto res = a.integrate(f, epsrel, epsabs);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double absolute_error = std::abs(res.value - correct_answer);
  bool const good = (res.status == 0);
  std::cout << std::scientific << algname << '\t' << epsrel << '\t';
  if (good) {
    std::cout << res.value << '\t' << res.error << '\t' << absolute_error
              << '\t';
  } else {
    std::cout << "NA\tNA\tNA\t";
  }
  std::cout << res.neval << '\t' << res.nregions << '\t' << dt.count()
            << std::endl;
}

int
main()
{
  cubacores(0, 0); // turn off the forking use.
  unsigned long long constexpr maxeval = 100 * 1000 * 1000 * 1000UL;
  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;
  cubacpp::Vegas vegas;
  vegas.maxeval = maxeval;
  std::cout << "alg\tepsrel\tvalue\terrorest\terror\tneval\tnregions\ttime\n";

  double epsrel = 1.0e-3;
  // warm up the CPU
  auto r = cuhre.integrate(fun6, epsrel, 1.0e-12);
  if (r.status != 0) return 1;
  for (int i = 1; i <= 20; ++i) {
    cuhre.flags = 0;
    time_and_call(cuhre, fun6, epsrel, 1.0, "cuhre_0");
    cuhre.flags = 4;
    time_and_call(cuhre, fun6, epsrel, 1.0, "cuhre_1");
    // time_and_call(vegas, fun6, epsrel, 1.0, "vegas");
    epsrel /= 5.0;
  }
}
