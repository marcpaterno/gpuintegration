#include "vegas/vegasT.cuh"
#include "vegas/util/Volume.cuh"

class SinSum6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    return sin(x + y + z + k + l + m);
  }
};

int main(int argc, char **argv)
{
    double epsrel = 1e-3;
    double epsabs = 1e-20;
		
	constexpr int ndim = 6;
	double ncall = 4.0e9;
	int titer = 20;
	int itmax = 0;
	int skip = 0;
    verbosity = 0;
    
//double avgi, chi2a, sd;
    std::cout <<"id, estimate, std, chi, iters, adj_iters, skip_iters, ncall, time, abserr, relerr\n";
    
	double lows[] = {0., 0., 0., 0., 0., 0.};
	double highs[] = {10., 10., 10., 10., 10., 10.};
	quad::Volume<double, ndim> volume(lows, highs);
    SinSum6D integrand;    
	
    auto res = integrate<SinSum6D, ndim>(integrand, ndim, epsrel, epsabs, ncall, titer, itmax, skip, &volume);
        
    std::cout.precision(15); 
    std::cout << std::scientific << res.estimate << "," 
        << std::scientific << res.errorest << "," 
        << res.chi_sq << ","
        << res.status << "\n";
		
    res = simple_integrate<SinSum6D, ndim>(integrand, ndim, epsrel, epsabs, ncall, titer, itmax, skip, &volume);
        
    std::cout.precision(15); 
    std::cout << std::scientific << res.estimate << "," 
        << std::scientific << res.errorest << "," 
        << res.chi_sq << ","
        << res.status << "\n";
	return 0;

}