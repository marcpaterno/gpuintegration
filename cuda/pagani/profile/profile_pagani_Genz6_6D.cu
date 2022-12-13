#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class F_6_6D {
public:
  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y, double z)
  {
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u);
  }
};

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 100;
  constexpr int ndim = 6;
  F_6_6D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<F_6_6D, ndim>(integrand, vol, num_repeats);
  return 0;
}
