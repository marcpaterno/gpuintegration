#ifndef CUBACPP_SUAVE_HH
#define CUBACPP_SUAVE_HH

#include "cuba.h"
#include "cubacpp/arity.hh"
#include "cubacpp/cuba_wrapped_integrand.hh"
#include "cubacpp/integrand_traits.hh"
#include "cubacpp/integration_result.hh"

namespace cubacpp {

  template <typename F>
  auto
  SuaveIntegrate(F const& user_function,
                 double epsrel,
                 double epsabs,
                 integration_volume_for_t<F> vol = {},
                 int flags = 0,
                 long long mineval = 0,
                 long long maxeval = 50000,
                 long long nnew = 1000,
                 long long nmin = 2,
                 double flatness = 25)
  {
    using results_t = typename integrand_traits<F>::integration_results_type;

    constexpr int N = integrand_traits<F>::ndim;

    integrand_t rescaled_function = detail::cuba_wrapped_integrand<F>;
    detail::definite_integral adapt(&user_function, &vol);
    constexpr int nvec = 1;
    results_t res = detail::default_integration_results(user_function);
    auto [nc, pvals, perrs, pprobs] =
      detail::make_cuba_args(user_function, res);

    llSuave(N,
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
            nnew,
            nmin,
            flatness,
            nullptr,
            nullptr,
            &res.nregions,
            &res.neval,
            &res.status,
            pvals,
            perrs,
            pprobs);
    return res;
  }

  struct Suave {
    int flags = 0;
    long long int mineval = 0;
    long long int maxeval = 50000;
    long long int nnew = 1000;
    long long int nmin = 2;
    double flatness = 25.0;

    template <typename F>
    auto
    integrate(F const& user_function, double epsrel, double epsabs) const
    {
      integration_volume_for_t<F> unitvolume;
      return SuaveIntegrate(user_function,
                            epsrel,
                            epsabs,
                            unitvolume,
                            flags,
                            mineval,
                            maxeval,
                            nnew,
                            nmin,
                            flatness);
    }

    template <typename F>
    auto
    integrate(F const& user_function,
              double epsrel,
              double epsabs,
              integration_volume_for_t<F> volume) const
    {
      return SuaveIntegrate(user_function,
                            epsrel,
                            epsabs,
                            volume,
                            flags,
                            mineval,
                            maxeval,
                            nnew,
                            nmin,
                            flatness);
    }
  };
}

#endif
