#ifndef GPUINTEGRATION_VEGAS_MCUBES_CUH
#define GPUINTEGRATION_VEGAS_MCUBES_CUH

#include "vegas/util/util.cuh"

namespace quad {
  struct mcubes {
    long long int maxcalls = 10 * 1000 * 1000ULL;
    int total_iters = 15;
    int adjust_iters = 15;
    int skip_iters = 5;

    template <typename F>
    IntegrationResult integrate(F&& f, double epsabs, double epsrel);
  };
}


template <typename F>
IntegrationResult
quad::mcubes::integrate(
  F const& f,
  double epsabs,
  double epsrel,
  quad::Volume<double, y3_cluster::arity<F>::value> const* pvol)
{
  constexpr std::size_t ndim = y3_cluster::arity<F>::value;
  return simple_integrate<F, ndim>(f,
                                   ndim,
                                   epsrel,
                                   epsabs,
                                   static_cast<double>(maxcalls),
                                   total_iters,
                                   adjust_iters,
                                   skip_iters,
                                   pvol);
}

#endif
