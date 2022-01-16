#ifndef VEGAS_NRC_HH
#define VEGAS_NRC_HH

#include "cubacpp/arity.hh"
#include "cubacpp/array.hh"
#include <array>
#include <functional>
#include <iterator>
#include <ostream>
//#define class struct
//#define private public
#include "cubacpp/integration_volume.hh"
//#define class class
#include "cubacpp/integration_result.hh"
//#define private private
#define __device__
#define __host__
#include "cudaPagani/quad/util/Volume.cuh"
#include "cudaPagani/quad/util/cuhreResult.cuh"
#include "vegas/drivervegasM.h"

struct VegasNRC {
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

  template <typename F>
  cuhreResult<double> integrate(
    F f,
    double epsrel,
    double epsabs,
    quad::Volume<double, cubacpp::arity<F>()> const* vol) const;

  // unsigned long ncall = 100000;
  unsigned long ncall = 500 * 1000 * 1000;
  int itmx = 15;
};

template <typename F>
cubacpp::integration_result
VegasNRC::integrate(F f,
                    double epsrel,
                    double epsabs,
                    typename cubacpp::integration_volume_for<F>::type vol) const
{
  auto lows = vol.lows();
  auto highs = vol.highs();

  quad::Volume<double, cubacpp::arity<F>()> volume(lows.data, highs.data);
  cuhreResult<double> res = vegas_book_integrate(
    f, cubacpp::arity<F>(), epsrel, epsabs, ncall, &volume, itmx);

  return {res.estimate,
          res.errorest,
          -1.,
          static_cast<long long>(res.neval),
          static_cast<int>(res.nregions),
          static_cast<int>(res.status)};
}

template <typename F>
cubacpp::integration_result
VegasNRC::integrate(F f, double epsrel, double epsabs) const
{

  quad::Volume<double, cubacpp::arity<F>()> volume;
  cuhreResult<double> res = vegas_book_integrate(
    f, cubacpp::arity<F>(), epsrel, epsabs, ncall, &volume, itmx);

  return {res.estimate,
          res.errorest,
          -1.,
          static_cast<long long>(res.neval),
          static_cast<int>(res.nregions),
          static_cast<int>(res.status)};
}

template <typename F>
cuhreResult<double>
VegasNRC::integrate(F f,
                    double epsrel,
                    double epsabs,
                    quad::Volume<double, cubacpp::arity<F>()> const* vol) const
{
  printf("invoking from vegasNRC.hh and using quad::Volume\n");
  // quad::Volume<double, cubacpp::arity<F>()> volume(vol.lows.data,
  // vol->highs.data);
  cuhreResult<double> res = vegas_book_integrate(
    f, cubacpp::arity<F>(), epsrel, epsabs, ncall, vol, itmx);

  return res;
}

#endif
