#include "vegas/util/Volume.cuh"
#include "vegas/vegasT.cuh"

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
           (1 / pow(sqrt(2 * PI) * 0.01, 9));
  }
};

int
main(int argc, char** argv)
{
  double epsrel = 1e-3;
  double epsabs = 1e-20;

  // double regn[2 * MXDIM + 1];

  // int fcode = 0;
  constexpr int ndim = 9;
  // float LL = 0.;
  // float UL = 10.;
  double ncall = 1.0e8;
  int titer = 20;
  int itmax = 10;
  int skip = 0;
  verbosity = 0;

  // double avgi, chi2a, sd;
  std::cout << "id, estimate, std, chi, iters, adj_iters, skip_iters, ncall, "
               "time, abserr, relerr\n";

  double lows[] = {-1., -1., -1., -1., -1., -1., -1., -1., -1.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  Gauss9D integrand;

  auto res = integrate<Gauss9D, ndim>(
    integrand, ndim, epsrel, epsabs, ncall, titer, itmax, skip, &volume);

  std::cout.precision(15);
  std::cout << std::scientific << res.estimate << "," << std::scientific
            << res.errorest << "," << res.chi_sq << "," << res.status << "\n";

  res = simple_integrate<Gauss9D, ndim>(
    integrand, ndim, epsrel, epsabs, ncall, titer, itmax, skip, &volume);

  std::cout.precision(15);
  std::cout << std::scientific << res.estimate << "," << std::scientific
            << res.errorest << "," << res.chi_sq << "," << res.status << "\n";
  return 0;
}