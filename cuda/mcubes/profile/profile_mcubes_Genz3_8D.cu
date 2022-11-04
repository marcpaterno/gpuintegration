#include "cuda/mcubes/demos/demo_utils.cuh"
#include "cuda/mcubes/vegasT.cuh"

class GENZ_3_8D {
public:
  __host__ __device__ double operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
  {
    return pow(1 + 8 * s + 7 * t + 6 * u + 5 * v + 4 * w + 3 * x + 2 * y + z, -9);
  }
};

int
main(int argc, char** argv)
{
  double epsrel = 1e-3;
  constexpr int ndim = 8;

  double ncall = 1.0e8;
  int titer = 1;
  int itmax = 1;
  int skip = 0;
  VegasParams params(ncall, titer, itmax, skip);
  double true_value = 0.010846560846560846561;
  quad::Volume<double, ndim> volume;
  
  GENZ_3_8D integrand;
  std::array<double, 6> required_ncall = {1.e5, 1.e6, 1.e7, 1.e8, 1.e9, 2.e9};
   
  size_t run = 0;
  
  for(auto num_samples : required_ncall){
    params.ncall = num_samples;
    
	signle_invocation_time_and_call<GENZ_3_8D, ndim>(
        integrand, epsrel, true_value, "f3, 8", params, &volume);
	run++;
	if(run > required_ncall.size())
		break;
  }

  return 0;
}