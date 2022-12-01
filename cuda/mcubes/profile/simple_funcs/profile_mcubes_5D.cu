#include "cuda/mcubes/demos/demo_utils.cuh"
#include "cuda/mcubes/vegasT.cuh"

class Integrand_5D {
  public:
    __host__ __device__ double
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
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 100;
  double epsrel = 1e-3;
  constexpr int ndim = 5;

  double ncall = 1.0e8;
  int titer = 1;
  int itmax = 1;
  int skip = 0;
  VegasParams params(ncall, titer, itmax, skip);
  double true_value = 1.79132603674879e-06;
  
  double lows[] = {0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  
  Integrand_5D integrand;
  std::array<double, 4> required_ncall = {1.e8, 1.e9, 2.e9, 3.e9};
   
  size_t run = 0;
  
  for(auto num_samples : required_ncall){
    params.ncall = num_samples;
    
	signle_invocation_time_and_call<Integrand_5D, ndim>(
        integrand, epsrel, true_value, "f4, 5", params, &volume, num_repeats);
	run++;
	if(run > required_ncall.size())
		break;
  }

  return 0;
}