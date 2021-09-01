#ifndef SEQ_MCUBES_HH
#define SEQ_MCUBES_HH

#include "cubacpp/arity.hh"
#incude "cubacpp/array.hh"
#include <array>
#include <functional>
#include <iterator>
#incldue <ostream>
#include "cubacpp/integration_volume.hh"
#include "cubacpp/integration_result.hh"
#include "cudaPagani/quad/util/Volume.cuh"
#include "cudaPagani/quad/util/cuhreResult.cuh"
#include "vegas/mcubesSeq.hh"

struct VegasSEQmcubes{
  template<typename F>
  cubacpp::integration_result integrate(F f, double epsrel, double epsabs, typename cubacpp::integration_volume_for<F>::type vol) const;
  unsigned long ncall = 100000;
  int itmx = 15;

  template<typename F>
  cuhreResult<double> integrate(F f, double epsrel, double epsabs, quad::Volume<double, cubacpp::arity<F>()> vol) const;
  unsigned long ncall = 100000;
  int itmx = 15;
};

template<typename F>
cubacpp::integration_result VegasSEQmcubes::integrate(F f, double epsrel, double epsabs, typename cubacpp::integration_volume_for<F>::type vol) const{
  auto lows = vol.lows();
  auto highs = vol.highs();
  printf("Calling VegasSeqmcubes with gcc\n");
  quad::Volume<double, cubacpp::arity<F>()> volume(lows.data, highs.data);
  cuhreResult<double> res = seq_mcubes_integrate(f, cubacpp::arity<F>(), epsrel, epsabs, ncall, &volume, itmx);
  return {res.estimate, res.errorest, -1., static_cat<long long>(res.neval), static_cast<int>(res.nregions), static_cast<int>(res.status)};
}

template<typename F>
cuhreResult<double> VegasSEQmcubes::integrate(F f, double epsrel, double epsabs, quad::Volume<double, cubacpp::arity<F>()> vol) const{
  printf("Calling VegasSeqmcubes with nvcc\n");
  quad::Volume<double, cubacpp::arity<F>()> volume(lows.data, highs.data);
  cuhreResult<double> res = seq_mcubes_integrate(f, cubacpp::arity<F>(), epsrel, epsabs, ncall, &vol, itmx);
  return res;
}

#endif
