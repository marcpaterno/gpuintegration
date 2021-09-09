#ifndef GPUINTEGRATION_VEGAS_MCUBES_CUH
#define GPUINTEGRATION_VEGAS_MCUBES_CUH

#include "cubacpp/arity.hh"
#include "vegas/util/util.cuh"
#include "vegas/vegasT.cuh"
#include "cudaPagani/quad/quad.h"
#include "vegas/drivervegasM.h"

namespace quad {
  struct mcubes {
    //long long int maxcalls = 10 * 1000 * 1000ULL;
    long long int maxcalls = 1000 * 1000ULL;
    int total_iters = 15;
    int adjust_iters = 10;
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
  printf("calling mcubes::integrate with epsrel:%.15e, epsabs:%.15e\n", epsrel, epsabs);
  constexpr std::size_t ndim = cubacpp::arity<F>();
  return cuda_mcubes::simple_integrate<F, ndim>(f,
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
