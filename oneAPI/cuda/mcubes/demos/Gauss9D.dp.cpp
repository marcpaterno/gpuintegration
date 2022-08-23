#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "math.h"
#include "./demo_utils.dp.hpp"
#include "./vegasT.dp.hpp"
#include "./vegasT1D.dp.hpp"

class Gauss9D {
public:
  SYCL_EXTERNAL double operator()(double x, double y, double z, double k,
                                  double l, double m, double n, double o,
                                  double p)
  {
    double sum =
        x * x + y * y + z * z + k * k + l * l + m * m + n * n + o * o + p * p;
    return exp(-1 * sum / (2 * 0.01 * 0.01)) *
           (1 / sycl::pown(sycl::sqrt(2 * M_PI) * 0.01, 9));
  }
};

int
main(int argc, char** argv)
{
  double epsrel = 1.e-3;
  constexpr int ndim = 9;
  double epsabs = 1.e-20;
  double ncall = 1.0e8;
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

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  constexpr bool MCUBES_DEBUG = false;
  for (int run = 0; run < 100; run++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto res = cuda_mcubes::integrate<Gauss9D, ndim, MCUBES_DEBUG>(
      integrand,
      epsrel,
      epsabs,
      params.ncall,
      &volume,
      params.t_iter,
      params.num_adjust_iters,
      params.num_skip_iters);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

    std::cout << "Gauss9D"
              << "," << epsrel << "," << std::scientific << true_value << ","
              << std::scientific << res.estimate << "," << std::scientific
              << res.errorest << "," << res.chi_sq << "," << params.t_iter
              << "," << params.num_adjust_iters << "," << params.num_skip_iters
              << "," << res.iters << "," << params.ncall << "," << res.neval
              << "," << dt.count() << "," << res.status << "\n";
    break;
  }

  return 0;
}
