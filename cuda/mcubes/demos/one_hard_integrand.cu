#include "cuda/mcubes/demos/demo_utils.cuh"
#include "cuda/mcubes/vegasT.cuh"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

class GENZ_4_8D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double k,
             double m,
             double n)
  {
    // double alpha = 25.;
    double beta = .5;
    return exp(-1.0 *
               (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2) +
                pow(25, 2) * pow(z - beta, 2) + pow(25, 2) * pow(w - beta, 2) +
                pow(25, 2) * pow(v - beta, 2) + pow(25, 2) * pow(k - beta, 2) +
                pow(25, 2) * pow(m - beta, 2) + pow(25, 2) * pow(n - beta, 2)));
  }
};


int
main(int argc, char** argv)
{
  std::vector<std::string> args(argv+1, argv+argc);
  if (args.empty())
  {
    std::cerr << "You must specify a value for ncall\n";
    return 1;
  }
  double const epsabs = 1.e-100;
  double const epsrel = 1.e-7;
  double const ncall = std::stod(args[0]);

  constexpr int ndim = 8;
  int const titer = 100;
  int const itmax = 15;
  int const skip = 5;

  GENZ_4_8D integrand;
  quad::Volume<double, ndim> volume;

  auto result = cuda_mcubes::integrate<GENZ_4_8D, 8>(integrand, epsrel, epsabs, ncall, &volume, titer, itmax, skip);
  std::cout << result << '\n';
}
