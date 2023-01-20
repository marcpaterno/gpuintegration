#include <iostream>
#include <math.h>
#include <chrono>
#include "cuda/pagani/quad/GPUquad/Workspace.cuh"

struct Fcn {
  double const normalization =
    6.0 / (-sin(1.0) - sin(2.0) + sin(4.0) + sin(5.0) - sin(6.0));

  __device__ double
  operator()(double x, double y, double z) const
  {
    return normalization * cos(x + 2. * y + 3 * z);
  }
};

int
main()
{
  constexpr int ndim = 3;
  Workspace<double, ndim> pagani;
  double const epsrel = 1.0e-3;
  double const epsabs = 1.0e-15;
  Fcn fcn;
  quad::Volume<double, ndim> vol;
  for (int i = 1; i <= 10000; ++i) {
    auto res = pagani.integrate(fcn, epsrel / i, epsabs, vol);
    if (i % 100 == 0) {
      std::cout << i << ' ' << res << '\n';
    }
  }
}
