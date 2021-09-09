#ifndef VEGAS_NRC_CUH
#define VEGAS_NRC_CUH

#include "cubacpp/arity.hh"
#include "vegas/util/util.cuh"
#include "vegas/vegasT.cuh"
#include "cudaPagani/quad/quad.h"
#include "vegas/drivervegasM.h"

namespace quad {
  struct vegasNRC {
    //long long int maxcalls = 10 * 1000 * 1000ULL;
    long long int maxcalls = 500*1000 * 1000ULL;
    int total_iters = 15;
    int adjust_iters = 15;
    int skip_iters = 5;
    
    template <typename F>
    cuhreResult<double> integrate(F const& f, double epsabs, double epsrel, quad::Volume<double, cubacpp::arity<F>()> const* pvol);
  };
}


template <typename F>
cuhreResult<double>
quad::vegasNRC::integrate(
  F const& f,
  double epsabs,
  double epsrel,
  quad::Volume<double, cubacpp::arity<F>()> const* pvol)
{
  printf("calling vegasNRC.cuh::integrate with epsrel:%.15e, epsabs:%.15e\n", epsrel, epsabs);
  constexpr std::size_t ndim = cubacpp::arity<F>();
  return vegas_book_integrate<F, ndim>(f, ndim, epsrel, epsabs, static_cast<unsigned long>(maxcalls), pvol, total_iters);
}

#endif
