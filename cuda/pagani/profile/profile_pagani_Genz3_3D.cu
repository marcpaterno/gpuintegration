#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_3_3D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z)
  {
    return pow(1 + 3 * x + 2 * y + z, -4);
  }
};


int
main()
{
  constexpr int ndim = 3;
  GENZ_3_3D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<GENZ_3_3D, ndim>(integrand, vol);
  return 0;
}
