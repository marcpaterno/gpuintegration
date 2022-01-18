#ifndef SEQ_MCUBES_HH
#define SEQ_MCUBES_HH

#include "cubacpp/arity.hh"
#include "cubacpp/array.hh"
#include "cubacpp/integration_result.hh"
#include "cubacpp/integration_volume.hh"
#include "cuda/cudaPagani/quad/util/Volume.cuh"
#include "cuda/cudaPagani/quad/util/cuhreResult.cuh"
#include "cuda/mcubes/mcubesSeq.hh"
#include <array>
#include <functional>
#include <iostream>
#include <iterator>
#include <ostream>

struct VegasSEQmcubes {
  template <typename F>
  cubacpp::integration_result integrate(
    F f,
    double epsrel,
    double epsabs,
    typename cubacpp::integration_volume_for<F>::type vol) const;

  template <typename F>
  cubacpp::integration_result integrate(F f,
                                        double epsrel,
                                        double epsabs) const;

  // unsigned long ncall = 500000000;
  // unsigned long maxcalls = 1000000;
  long long int maxcalls = 1000000;
  int itmx = 6;

  template <typename F>
  cuhreResult<double> integrate(
    F f,
    double epsrel,
    double epsabs,
    quad::Volume<double, cubacpp::arity<F>()> const* vol) const;
};

template <typename F>
cubacpp::integration_result
VegasSEQmcubes::integrate(
  F f,
  double epsrel,
  double epsabs,
  typename cubacpp::integration_volume_for<F>::type vol) const
{
  auto lows = vol.lows();
  auto highs = vol.highs();

  quad::Volume<double, cubacpp::arity<F>()> volume(lows.data, highs.data);
  cuhreResult<double> res =
    seq_mcubes_integrate(f,
                         cubacpp::arity<F>(),
                         epsrel,
                         epsabs,
                         static_cast<long long int>(maxcalls),
                         &volume,
                         itmx);
  return {res.estimate,
          res.errorest,
          -1.,
          static_cast<long long>(res.neval),
          static_cast<int>(res.nregions),
          static_cast<int>(res.status)};
}

template <typename F>
cubacpp::integration_result
VegasSEQmcubes::integrate(F f, double epsrel, double epsabs) const
{
  printf("Invoking from input of cubacpp::Volume\n");
  std::cout << "ncall:" << maxcalls << "\n";
  quad::Volume<double, cubacpp::arity<F>()> volume;
  cuhreResult<double> res =
    seq_mcubes_integrate(f,
                         cubacpp::arity<F>(),
                         epsrel,
                         epsabs,
                         static_cast<long long int>(maxcalls),
                         &volume,
                         itmx);
  return {res.estimate,
          res.errorest,
          -1.,
          static_cast<long long>(res.neval),
          static_cast<int>(res.nregions),
          static_cast<int>(res.status)};
}

template <typename F>
cuhreResult<double>
VegasSEQmcubes::integrate(
  F f,
  double epsrel,
  double epsabs,
  quad::Volume<double, cubacpp::arity<F>()> const* vol) const
{
  printf("Invoking from input of quad::Volume\n");
  std::cout << "ncall:" << maxcalls << "\n";
  // std::cout<<"itmx:"<<itmx<<"\n";
  cuhreResult<double> res =
    seq_mcubes_integrate(f,
                         cubacpp::arity<F>(),
                         epsrel,
                         epsabs,
                         static_cast<long long int>(maxcalls),
                         vol,
                         itmx);
  return res;
}

#endif
