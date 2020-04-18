#ifndef Y3_CLUSTER_OMEGA_Z_SDSS_HH
#define Y3_CLUSTER_OMEGA_Z_SDSS_HH

#include "utils/datablock.hh"
#include "utils/polynomial.hh"

#include <array>
#include <cmath>

namespace y3_cluster {
  struct OMEGA_Z_SDSS {
  public:
    OMEGA_Z_SDSS() = default;
    OMEGA_Z_SDSS(cosmosis::DataBlock&) {}

    double
    operator()(double zt) const
    {
      static const y3_cluster::polynomial<12> SDSS_fit{{-1.14293122E05,
                                                        5.96846869E04,
                                                        9.24239180E03,
                                                        -2.23118813E03,
                                                        -4.52580713E03,
                                                        1.18404878E03,
                                                        1.27951911E02,
                                                        -5.05716847E01,
                                                        1.01744577E00,
                                                        -3.11253383E-01,
                                                        5.48481084E-03,
                                                        3.12629987E00}};
      // Returns effective survey area in rad^2
      return SDSS_fit(zt - 0.2);
    }
  };
}

#endif
