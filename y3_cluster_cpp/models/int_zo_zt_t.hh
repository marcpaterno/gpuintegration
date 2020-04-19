#ifndef Y3_CLUSTER_INT_ZO_ZT_HH
#define Y3_CLUSTER_INT_ZO_ZT_HH

#include "utils/datablock_reader.hh"
#include <cmath>

namespace y3_cluster {

  class INT_ZO_ZT_t {
  public:
    INT_ZO_ZT_t () = default;
    explicit INT_ZO_ZT_t(double sigma) : _sigma(sigma) {}

    explicit INT_ZO_ZT_t(cosmosis::DataBlock& sample)
      : _sigma(
          get_datablock<double>(sample, "cluster_abundance", "zo_zt_sigma"))
    {}

    double
    operator()(double zomin, double zomax, double zt) const
    {
      // P(zo | zt) := (1.0 / sqrt(2pi) / sigma) * exp(- (zo - zt) * (zo - zt) /
      // (2 * sigma * sigma))
      //    (i.e., a standard gaussian)
      // So,
      //    \int P(zo|zt) d(zo), zo in [zomin, zomax]
      //     == (erf((zomax - zt) / base) - erf((zomin - zt) / base)) / 2
      using std::erf;
      double _sigma2;
      _sigma2 = 0.02129638 - 0.25085154 * zt + 1.11756659 * zt * zt -
                1.22925508 * zt * zt * zt;
      //_sigma2 = 0.02009036-0.24311016*zt+1.0778892*zt*zt-1.20483017*zt*zt*zt;
      double base = std::sqrt(2) * _sigma2;
      return (erf((zomax - zt) / base) - erf((zomin - zt) / base)) / 2.0;
    }

  private:
    double _sigma = 0.0;
  };
}

#endif
