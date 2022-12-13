#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class SinSum6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    return sin(x + y + z + k + l + m);
  }
};

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 6;
  SinSum6D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<SinSum6D, ndim>(integrand, vol, num_repeats);
  return 0;
}
