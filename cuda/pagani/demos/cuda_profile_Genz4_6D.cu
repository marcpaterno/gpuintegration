#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_4_6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double w, double v, double b)
  {
	return 1.;
    double beta = .5;
    return exp(-1.0 * pow(25, 2) *
               (pow(x - beta, 2) + pow(y - beta, 2) + pow(z - beta, 2) +
                pow(w - beta, 2) + pow(v - beta, 2) + pow(b - beta, 2)));
  }
};

int
main()
{
  constexpr int ndim = 6;
  GENZ_4_6D integrand;
  quad::Volume<double, ndim> vol;

  //for(int i=0; i < 10; ++i)
	call_cubature_rules<GENZ_4_6D, ndim>(integrand, vol);
	
  return 0;
}
