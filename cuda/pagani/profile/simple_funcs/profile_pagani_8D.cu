#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_5_8D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q)
  {
	  double sum = 0.;
	for(int i=0; i < 1000; ++i)
		sum += (x*y*z*k*m*n*p*q)/(x/y/z/k/m/n/p/q);
	return sum;
  }
};

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 8;
  GENZ_5_8D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<GENZ_5_8D, ndim>(integrand, vol, num_repeats);
  return 0;
}
