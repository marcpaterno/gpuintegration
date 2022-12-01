#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_4_5D {
  public:
    __device__ __host__ double
    operator()(double x, double y, double z, double w, double v)
    {
		double sum = 0.;
	for(int i=0; i < 1000; ++i)
		sum += (x*y*z*w*v)/(x/y/z/w/v);
	return sum;
    }
};

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 5;
  GENZ_4_5D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<GENZ_4_5D, ndim>(integrand, vol, num_repeats);
  return 0;
}
