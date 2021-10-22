#include "vegas/vegasT.cuh"
#include "vegas/demos/demo_utils.cuh"
#include "vegas/vegasT1D.cuh"
#include "math.h"

class Gauss9D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o,
             double p)
  {
    double sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
                 pow(m, 2) + pow(n, 2) + pow(o, 2) + pow(p, 2);
    return exp(-1 * sum / (2 * pow(0.01, 2))) *
           (1 / pow(sqrt(2 * M_PI) * 0.01, 9));
  }
};

int 
main(int argc, char** argv)
{
  double epsrel = 1.e-3;
  double epsrel_min = 1e-9;
  constexpr int ndim = 9;
  double epsabs = 1.e-20;
  double ncall = 1.0e7;
  int titer = 15;
  int itmax = 10;
  int skip = 10;
  double true_value = 1.;
  VegasParams params(ncall, titer, itmax, skip);

  std::cout << "id, estimate, std, chi, iters, adj_iters, skip_iters, ncall, "
               "time, abserr, relerr\n";

  double lows[] = {-1., -1., -1., -1., -1., -1., -1., -1., -1.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  
  Gauss9D integrand;
      
  /*while(mcubes_time_and_call<Gauss9D, ndim>
    (integrand, epsrel, true_value, "Gauss9D", params, &volume) == true && epsrel >= epsrel_min){
        epsrel /= 5.;
    }*/
using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;  
  constexpr bool MCUBES_DEBUG = false;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto res = cuda_mcubes::simple_integrate<Gauss9D, ndim, MCUBES_DEBUG>(integrand, ndim, epsrel, epsabs, params.ncall, &volume, params.t_iter, params.num_adjust_iters, params.num_skip_iters);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  std::cout << "Gauss9D" << "," 
            << std::scientific << true_value << "," 
            << std::scientific << res.estimate << "," 
            << std::scientific << res.errorest << "," 
            << dt.count() << "\n";  
  
  
  res = mcubes1D::simple_integrate1D<double, Gauss9D, ndim>(integrand, 
        epsrel, 
        epsabs, 
        params.ncall, 
        &volume, 
        titer, 
        itmax, 
        skip);  
    

  std::cout.precision(15);
  printf("-----------------------\n");
  std::cout << "Gauss9DD" << "," 
            << std::scientific << true_value << "," 
            << std::scientific << res.estimate << "," 
            << std::scientific << res.errorest << "," 
            << dt.count() << "\n";
  return 0;
}
