#include "cuda/mcubes/demos/demo_utils.cuh"
#include "cuda/mcubes/vegasT.cuh"

class GENZ_2_6D {
  public:
    __host__ __device__ double
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
  double epsrel = 1e-3;
  constexpr int ndim = 6;

  double ncall = 1.0e8;
  int titer = 1;
  int itmax = 1;
  int skip = 0;
  VegasParams params(ncall, titer, itmax, skip);
  double true_value = 1.286889807581113e+13;
  
  double lows[] = {0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  
  GENZ_2_6D integrand;
  std::array<double, 6> required_ncall = {1.e5, 1.e6, 1.e7, 1.e8, 1.e9, 2.e9};
   
  size_t run = 0;
  
  for(auto num_samples : required_ncall){
    params.ncall = num_samples;
    
	signle_invocation_time_and_call<GENZ_2_6D, ndim>(
        integrand, epsrel, true_value, "f2, 6", params, &volume);
	run++;
	if(run > required_ncall.size())
		break;
  }

  return 0;
}
