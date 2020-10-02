
//#include "../cudaCuhre/quad/quad.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

template <typename ALG, typename F>
bool
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
  return res.status == 0;
}

