#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class Pow_of_sum {
public:
  __device__ __host__ double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
  {
    return pow(1. + 8. * s + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, 9.);
  }
};

int
main()
{
  constexpr int ndim = 8;
  Pow_of_sum integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<Pow_of_sum, ndim>(integrand, vol);
  return 0;
}
