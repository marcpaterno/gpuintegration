#include "cuda/mcubes/demos/demo_utils.cuh"
#include "cuda/mcubes/vegasT.cuh"

class F_3_3D {
public:
  __host__ __device__ double operator()(double x, double y, double z)
  {
  return 0.1;
    return pow(1 + 3 * x + 2 * y + z, -4);
  }
};

int
main(int argc, char** argv)
{
  double epsrel = 1e-3;
  constexpr int ndim = 3;

  double ncall = 1.0e8;
  int titer = 1;
  int itmax = 1;
  int skip = 0;
  VegasParams params(ncall, titer, itmax, skip);
  double true_value = 0.010846560846560846561;
  
  double lows[] = {0., 0., 0.};
  double highs[] = {1., 1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  
  F_3_3D integrand;
  std::array<double, 6> required_ncall = {1.e5, 1.e6, 1.e7, 1.e8, 1.e9, 2.e9};
   
  size_t run = 0;
  
  for(auto num_samples : required_ncall){
    params.ncall = num_samples;
    
	signle_invocation_time_and_call<F_3_3D, ndim>(
        integrand, epsrel, true_value, "f3, 3", params, &volume);
	run++;
	if(run > required_ncall.size())
		break;
  }

  return 0;
}