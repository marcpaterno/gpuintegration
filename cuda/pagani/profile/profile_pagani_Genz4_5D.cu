#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class F_4_5D {
  public:
    __device__ __host__ double
    operator()(double x, double y, double z, double w, double v)
    {
	  double beta = .5;
      return exp(
        -1.0 * (pow(25., 2.) * pow(x - beta, 2.) + 
				pow(25., 2.) * pow(y - beta, 2.) +
                pow(25., 2.) * pow(z - beta, 2.) + 
				pow(25., 2.) * pow(w - beta, 2.) +
                pow(25., 2.) * pow(v - beta, 2.)));
    }
};

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 5;
  F_4_5D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<F_4_5D, ndim>(integrand, vol, num_repeats);
  return 0;
}
