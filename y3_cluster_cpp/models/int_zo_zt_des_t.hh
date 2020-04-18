#ifndef Y3_CLUSTER_INT_ZO_ZT_HH
#define Y3_CLUSTER_INT_ZO_ZT_HH

#include "models/sigma_photoz_des.hh"
#include "utils/datablock_reader.hh"
#include <cmath>

namespace y3_cluster {

  class INT_ZO_ZT_DES_t {
  public:
    INT_ZO_ZT_DES_t() {}

    double
    operator()(double zomin, double zomax, double zt) const
    {
      double _sigma = _sigma_photoz_des(zt);
      double base = std::sqrt(2) * _sigma;
      return (std::erf((zomax - zt) / base) - std::erf((zomin - zt) / base)) /
             2.0;
    }

  private:
    y3_cluster::SIGMA_PHOTOZ_DES_t _sigma_photoz_des;
  };
}

#endif
