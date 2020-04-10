#ifndef Y3_CLUSTER_LC_LT_T_HH
#define Y3_CLUSTER_LC_LT_T_HH

#include "utils/datablock.hh"
#include "utils/interp_2d.hh"
#include "utils/primitives.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace y3_cluster {
  struct LC_LT_t {

    static Interp2D const tau_interp;
    static Interp2D const mu_interp;
    static Interp2D const sigma_interp;
    static Interp2D const fmsk_interp;
    static Interp2D const fprj_interp;

    explicit LC_LT_t(const cosmosis::DataBlock&) {}
    LC_LT_t() {}

    double
    operator()(double lc, double lt, double zt) const
    {
      const auto tau = tau_interp(lt, zt);
      const auto mu = mu_interp(lt, zt);
      const auto sigma = sigma_interp(lt, zt);
      const auto fmsk = fmsk_interp(lt, zt);
      const auto fprj = fprj_interp(lt, zt);

      const auto exptau =
        std::exp(tau * (2.0 * mu + tau * sigma * sigma - 2.0 * lc) / 2.0);
      const auto root_two_sigma = std::sqrt(2.0) * sigma;
      const auto mu_tau_sig_sqr = mu + tau * sigma * sigma;

      // Helper function for common pattern
      const auto erfc_scaled = [root_two_sigma](double a, double b) {
        return std::erfc((a - b) / root_two_sigma);
      };

      // eq. (33)
      return (1.0 - fmsk) * (1.0 - fprj) * y3_cluster::gaussian(lc, mu, sigma) +
             0.5 * ((1.0 - fmsk) * fprj * tau + fmsk * fprj / lt) * exptau *
               erfc_scaled(mu_tau_sig_sqr, lc) +
             0.5 * fmsk / lt *
               (erfc_scaled(lc, mu) - erfc_scaled(lc + lt, mu)) -
             0.5 * fmsk * fprj / lt *
               (std::exp(-tau * lt) * exptau *
                erfc_scaled(mu_tau_sig_sqr, lc + lt));
    }
  };
}

#endif
