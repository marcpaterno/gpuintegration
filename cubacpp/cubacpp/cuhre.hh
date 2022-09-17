#ifndef CUBACPP_CUHRE_HH
#define CUBACPP_CUHRE_HH

#include "cuba.h"
#include "cubacpp/arity.hh"
#include "cubacpp/cuba_wrapped_integrand.hh"
#include "cubacpp/integrand_traits.hh"
#include "cubacpp/integration_result.hh"
#include "cubacpp/integration_volume.hh"
#include <optional>

namespace cubacpp {

  namespace cuhre_private {
    // Adjust the user-supplied 'key' to one of the values allowed by CUHRE.
    inline std::optional<int>
    adjust_key(int key, int N)
    {
      if (key < 0) {
        switch (N) {
          case 1:
          case 2:
            key = 13;
            break;
          case 3:
            key = 11;
            break;
          default:
            key = 9;
        }
      }
      if (key == 7 || key == 9 || key == 11 || key == 13)
        return {key};
      return {};
    }
  } // cuhre_private

  template <typename F>
  auto
  // CuhreIntegrate(F const& user_function,
  CuhreIntegrate(F user_function,
                 double epsrel,
                 double epsabs,
                 integration_volume_for_t<F> vol = {},
                 int flags = 0,
                 long long mineval = 0,
                 long long maxeval = 50000,
                 int key = -1)
  {
    using results_t = typename integrand_traits<F>::integration_results_type;
    using integral_adapter_t = typename cubacpp::detail::definite_integral<F>;

    constexpr int N = integrand_traits<F>::ndim;
    // If key was not specified, deduce the highest order we can use based on
    // the dimensionality of the integrand.
    static_assert(N >= 2, "Integrand dimension must be two or more");
    auto maybe_key = cuhre_private::adjust_key(key, N);
    if (!maybe_key)
      return results_t{};

    // rescaled_function is the function that will be called by CUHRE.
    // It will always be given argument values within the unit hypercube.
    integrand_t rescaled_function = detail::cuba_wrapped_integrand<F>;
    // detail::definite_integral adapt(&user_function, &vol)
    integral_adapter_t adapt(&user_function, &vol);
    constexpr int nvec = 1;
    results_t res = detail::default_integration_results(user_function);
    auto [nc, pvals, perrs, pprobs] =
      detail::make_cuba_args(user_function, res);

    llCuhre(N,
            nc,
            rescaled_function,
            (void*)&adapt,
            nvec,
            epsrel,
            epsabs,
            flags,
            mineval,
            maxeval,
            *maybe_key,
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

  struct Cuhre {
    int flags = 0;
    long long int mineval = 0;
    long long int maxeval = 50000;
    int key = -1;

    template <typename F>
    auto
    integrate(F const& user_function, double epsrel, double epsabs) const
    {
      integration_volume_for_t<F> unitvolume;
      return CuhreIntegrate(user_function,
                            epsrel,
                            epsabs,
                            unitvolume,
                            flags,
                            mineval,
                            maxeval,
                            key);
    }

    template <typename F>
    auto
    integrate(F const& user_function,
              double epsrel,
              double epsabs,
              integration_volume_for_t<F> volume) const
    {
      return CuhreIntegrate(
        user_function, epsrel, epsabs, volume, flags, mineval, maxeval, key);
    }
  };
}

#endif
