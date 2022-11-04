#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_2_6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double l, double m)
  {
	  
    const double a = 50.;
    const double b = .5;

    const double term_1 = 1. / ((1. / powf(a, 2.)) + powf(x - b, 2.));
    const double term_2 = 1. / ((1. / powf(a, 2.)) + powf(y - b, 2.));
    const double term_3 = 1. / ((1. / powf(a, 2.)) + powf(z - b, 2.));
    const double term_4 = 1. / ((1. / powf(a, 2.)) + powf(k - b, 2.));
    const double term_5 = 1. / ((1. / powf(a, 2.)) + powf(l - b, 2.));
    const double term_6 = 1. / ((1. / powf(a, 2.)) + powf(m - b, 2.));
	
    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
    return val;
  }
};

int
main()
{
  constexpr int ndim = 6;
  GENZ_2_6D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<GENZ_2_6D, ndim>(integrand, vol);
  return 0;
}
