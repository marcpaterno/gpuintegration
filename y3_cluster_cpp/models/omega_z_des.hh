#ifndef Y3_CLUSTER_OMEGA_Z_DES_HH
#define Y3_CLUSTER_OMEGA_Z_DES_HH

#include "utils/datablock.hh"
#include "utils/polynomial.hh"

#include <array>
#include <cmath>

namespace y3_cluster {
  struct OMEGA_Z_DES {
  public:
    OMEGA_Z_DES() = default;
    OMEGA_Z_DES(cosmosis::DataBlock&) {}

    double
    operator()(double zt) const
    {
      static const y3_cluster::polynomial<6> SDSS_fit{
        {0.0, 0.0, 0.0, -0.00262353, 0.01940118, 0.45133063}};
      static const y3_cluster::polynomial<6> SDSS_fit2{{1.33647377e+4,
                                                        1.35291046e+3,
                                                        -1.26204891e+2,
                                                        -2.83454918e+1,
                                                        -2.26465905,
                                                        3.84958753e-1}};
      static const y3_cluster::polynomial<6> SDSS_fit3{
        {0, 0, -1.88101967, 4.8071839, -4.11424324, 1.18196785}};
      // Returns effective survey area in rad^2
      if (zt < 0.504) {
        return SDSS_fit(zt);
      } else if (zt < 0.7) {
        return SDSS_fit2(zt - 0.6);
      } else {
        return SDSS_fit3(zt);
      }
    }
  };
}

#endif
