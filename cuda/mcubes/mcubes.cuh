#ifndef GPUINTEGRATION_VEGAS_MCUBES_CUH
#define GPUINTEGRATION_VEGAS_MCUBES_CUH

#include "cubacpp/arity.hh"
#include "cuda/pagani/quad/quad.h"
#include "cuda/mcubes/vegasT.cuh"
#include "common/integration_result.hh"

namespace quad {
  struct mcubes {
    long long int maxcalls = 1000 * 1000ULL;
    int total_iters = 70;
    int adjust_iters = 20;
    int skip_iters = 0;

    template <typename F>
    numint::integration_result integrate(
      F const& f,
      double epsabs,
      double epsrel,
      quad::Volume<double, cubacpp::arity<F>()> const* pvol);
  };
}

template <typename F>
numint::integration_result
quad::mcubes::integrate(F const& f,
                        double epsabs,
                        double epsrel,
                        quad::Volume<double, cubacpp::arity<F>()> const* pvol)
{
  constexpr std::size_t ndim = cubacpp::arity<F>();
  return cuda_mcubes::simple_integrate<F, ndim>(f,
                                                epsrel,
                                                epsabs,
                                                static_cast<double>(maxcalls),
                                                pvol,
                                                total_iters,
                                                adjust_iters,
                                                skip_iters);
}

#endif
