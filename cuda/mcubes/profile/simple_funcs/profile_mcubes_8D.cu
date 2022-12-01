#include "cuda/mcubes/demos/demo_utils.cuh"
#include "cuda/mcubes/vegasT.cuh"

class Integrand_8D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q) const
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
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 100;
  double epsrel = 1e-3;
  constexpr int ndim = 8;

  double ncall = 1.0e8;
  int titer = 1;
  int itmax = 1;
  int skip = 0;
  VegasParams params(ncall, titer, itmax, skip);
  double true_value = 2.425217625641885e-06;
  
  double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  
  Integrand_8D integrand;
  std::array<double, 4> required_ncall = {1.e8, 1.e9, 2.e9, 3.e9};
  size_t run = 0;
  
  for(auto num_samples : required_ncall){
    params.ncall = num_samples;
    
	signle_invocation_time_and_call<Integrand_8D, ndim>(
        integrand, epsrel, true_value, "f5, 8", params, &volume, num_repeats);
	run++;
	if(run > required_ncall.size())
		break;
  }

  return 0;
}