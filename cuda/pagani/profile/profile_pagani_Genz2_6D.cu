#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class F_2_6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double l, double m)
  {
	
	
	const double a = 50.;
    const double b = .5;
    const double term_1 = 1. / ((1. / pow(a, 2.)) + pow(x - b, 2.));
    const double term_2 = 1. / ((1. / pow(a, 2.)) + pow(y - b, 2.));
    const double term_3 = 1. / ((1. / pow(a, 2.)) + pow(z - b, 2.));
    const double term_4 = 1. / ((1. / pow(a, 2.)) + pow(k - b, 2.));
    const double term_5 = 1. / ((1. / pow(a, 2.)) + pow(l - b, 2.));
    const double term_6 = 1. / ((1. / pow(a, 2.)) + pow(m - b, 2.));
	
    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
    return val;
  }
};

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 6;
  F_2_6D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<F_2_6D, ndim>(integrand, vol, num_repeats);
  return 0;
}
