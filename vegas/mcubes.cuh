#ifndef GPUINTEGRATION_VEGAS_MCUBES_CUH
#define GPUINTEGRATION_VEGAS_MCUBES_CUH

#include "cubacpp/arity.hh"
#include "vegas/util/util.cuh"
#include "vegas/vegasT.cuh"
#include "cudaPagani/quad/quad.h"

namespace quad {
  struct mcubes {
    long long int maxcalls = 10 * 1000 * 1000ULL;
    int total_iters = 15;
    int adjust_iters = 15;
    int skip_iters = 5;

    template <typename F>
    cuhreResult<double> integrate(F const& f, double epsabs, double epsrel, quad::Volume<double, cubacpp::arity<F>()> const* pvol);
  };
}


template <typename F>
cuhreResult<double>
quad::mcubes::integrate(
  F const& f,
  double epsabs,
  double epsrel,
  quad::Volume<double, cubacpp::arity<F>()> const* pvol)
{
  constexpr std::size_t ndim = cubacpp::arity<F>();
  return simple_integrate<F, ndim>(f,
                                   ndim,
                                   epsrel,
                                   epsabs,
                                   static_cast<double>(maxcalls),
				   pvol,
                                   total_iters,
                                   adjust_iters,
                                   skip_iters);
}

#endif
