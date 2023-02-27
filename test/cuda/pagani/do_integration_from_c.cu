extern "C" {
#include "test/cuda/pagani/do_integration_from_c.h"
}

#include <chrono>
#include "cuda/pagani/quad/GPUquad/Workspace.cuh"

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
  double const epsrel = 1.0e-6;
  double const epsabs = 1.0e-12;
  Integrand integrand;
  constexpr int ndim = 2;
  quad::Volume<double, ndim> vol;

  Workspace<double, ndim> pagani;
  auto const result =
    pagani.integrate<Integrand>(integrand, epsrel, epsabs, vol);
  int rc = result.status;
  if (rc == 0)
    *res = result.estimate;
  return rc;
}
}
