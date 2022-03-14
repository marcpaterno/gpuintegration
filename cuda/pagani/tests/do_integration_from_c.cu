extern "C" {
#include "cuda/pagani/tests/do_integration_from_c.h"
}

#include "cuda/pagani/quad/GPUquad/Pagani.cuh"

extern "C" {

struct Integrand {
  __device__ double
  operator()(double x, double y) const
  {
    return x * y;
  }
};

using namespace quad;

int
do_integration_from_c(double* res)
{
  if (!res)
    return 1;
  double const epsrel = 1.0e-3;
  double const epsabs = 1.0e-12;
  Integrand integrand;
  constexpr int ndim = 2;

  quad::Pagani<double, ndim> pagani;
  auto const result = pagani.integrate<Integrand>(integrand, epsrel, epsabs);
  int rc = result.status;
  if (rc == 0)
    *res = result.estimate;
  return rc;
}
}
