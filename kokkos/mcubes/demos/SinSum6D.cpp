#include "time_and_call.h"
#include "common/kokkos/Volume.cuh"
class SinSum6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    return sin(x + y + z + k + l + m);
  }
};

int
main()
{
  Kokkos::initialize();
  {
    double epsrel = 1.e-3;
    constexpr int ndim = 6;
    double ncall = 2.0e9;
    int titer = 10;
    int itmax = 0;
    int skip = 0;
    VegasParams params(ncall, titer, itmax, skip);
  
    double true_value = -49.165073;
    std::cout << "id, estimate, std, chi, iters, adj_iters, skip_iters, ncall, "
                 "time, abserr, relerr\n";

    double lows[] = {0., 0., 0., 0., 0., 0.};
    double highs[] = {10., 10., 10., 10., 10., 10.};
    quad::Volume<double, ndim> volume(lows, highs);
    SinSum6D integrand;

    PrintHeader();
    bool success = false;
    success = mcubes_time_and_call<SinSum6D, ndim>(
      integrand, epsrel, true_value, "f4 5D", params, &volume);
  }
  Kokkos::finalize();
  return 0;
}
