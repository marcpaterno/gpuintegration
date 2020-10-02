#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "function.cuh"
#include "quad/quad.h"
#include "quad/util/cudaUtil.h"
#include <iomanip>

#include "demo_utils.h"
#include "quad/GPUquad/Cuhre.cuh"
#include "quad/util/Volume.cuh"
using namespace quad;

class GENZ_1_8d {

public:
  double normalization;
  double integral;
  __device__ __host__
  GENZ_1_8d()
  {
    integral = (1. / 315.) * sin(1.) * sin(3. / 2.) * sin(2.) * sin(5. / 2.) *
               sin(3.) * sin(7. / 2.) * sin(4.) *
               (sin(37. / 2.) - sin(35. / 2.));
    normalization = 1. / integral;
  }

  __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z)
  {
    return normalization * cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x +
                               7. * y + 8. * z);
  }
};

__host__ __device__ double
genz_1_8d(double s,
          double t,
          double u,
          double v,
          double w,
          double x,
          double y,
          double z)
{
  double integral = (1. / 315.) * sin(1.) * sin(3. / 2.) * sin(2.) *
                    sin(5. / 2.) * sin(3.) * sin(7. / 2.) * sin(4.) *
                    (sin(37. / 2.) - sin(35. / 2.));
  double normalization = 1. / integral;
  return normalization *
         cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x + 7. * y + 8. * z);
}

int
main()
{
  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  constexpr int ndim = 8;
  Cuhre<double, ndim> GPUcuhre(0, 0, 0, 0, 1);
  GENZ_1_8d integrand;
  double highs[ndim] = {1, 1, 1, 1, 1, 1, 1, 1};
  double lows[ndim] = {0, 0, 0, 0, 0, 0, 0, 0};
  Volume<double, ndim> vol(lows, highs);
  // GPUcuhre.integrate<GENZ_1_8d>(integrand, epsrel, epsabs, &vol);
  GPUcuhre.integrate(genz_1_8d, epsrel, epsabs, &vol);

  return 0;
}
