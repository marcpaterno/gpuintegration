#ifndef CUBACPP_VEGAS_HH
#define CUBACPP_VEGAS_HH

#include "cuba.h"
#include "cubacpp/arity.hh"
#include "cubacpp/cuba_wrapped_integrand.hh"
#include "cubacpp/integrand_traits.hh"
#include "cubacpp/integration_result.hh"
#include "cubacpp/integration_volume.hh"

namespace cubacpp {

  template <typename F>
  auto
  VegasIntegrate(F const& user_function,
                 double epsrel,
                 double epsabs,
                 integration_volume_for_t<F> vol = {},
                 int flags = 0,
                 long long mineval = 0,
                 long long maxeval = 50000,
                 long long nstart = 1000,
                 long long nincrease = 500,
                 long long nbatch = 1000)
  {
    using results_t = typename integrand_traits<F>::integration_results_type;

    constexpr int N = integrand_traits<F>::ndim;

    integrand_t rescaled_function = detail::cuba_wrapped_integrand<F>;
    detail::definite_integral adapt(&user_function, &vol);
    constexpr int nvec = 1;
    results_t res = detail::default_integration_results(user_function);
    auto [nc, pvals, perrs, pprobs] =
      detail::make_cuba_args(user_function, res);
    res.nregions = -1; // Vegas does not tell us how many regions it used.

    llVegas(N,
            nc,
            rescaled_function,
            (void*)&adapt,
            nvec,
            epsrel,
            epsabs,
            flags,
            0, // seed
            mineval,
            maxeval,
            nstart,
            nincrease,
            nbatch,
            0,
            nullptr,
            nullptr,
            &res.neval,
            &res.status,
            pvals,
            perrs,
            pprobs);
    return res;
  }

  struct Vegas {
    int flags = 0;
    long long int mineval = 0;
    long long int maxeval = 50000;
    long long nstart = 1000;
    long long nincrease = 500;
    long long nbatch = 1000;

    template <typename F>
    auto
    integrate(F const& user_function, double epsrel, double epsabs) const
    {
      integration_volume_for_t<F> unitvolume;
      return VegasIntegrate(user_function,
                            epsrel,
                            epsabs,
                            unitvolume,
                            flags,
                            mineval,
                            maxeval,
                            nstart,
                            nincrease,
                            nbatch);
    }

    template <typename F>
    auto
    integrate(F const& user_function,
              double epsrel,
              double epsabs,
              integration_volume_for_t<F> volume) const
    {
      return VegasIntegrate(user_function,
                            epsrel,
                            epsabs,
                            volume,
                            flags,
                            mineval,
                            maxeval,
                            nstart,
                            nincrease,
                            nbatch);
    }
  };
}

#endif
