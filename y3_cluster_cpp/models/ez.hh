#ifndef Y3_CLUSTER_CPP_EZ_HH
#define Y3_CLUSTER_CPP_EZ_HH

#include "ez_sq.hh"
#include "utils/datablock_reader.hh"

#include <cmath>
#include <iomanip>
#include <istream>
#include <ostream>

namespace y3_cluster {
  class EZ {
  public:
    EZ() = default;

    EZ(double omega_m, double omega_l, double omega_k)
      : _ezsq(omega_m, omega_l, omega_k)
    {}

    explicit EZ(cosmosis::DataBlock& sample)
      : EZ(get_datablock<double>(sample, "cosmological_parameters", "omega_m"),
           get_datablock<double>(sample,
                                 "cosmological_parameters",
                                 "omega_lambda"),
           get_datablock<double>(sample, "cosmological_parameters", "omega_k"))
    {}

    double
    operator()(double z) const
    {
      auto const sqr = _ezsq(z);
      return std::sqrt(sqr);
    }

    friend std::ostream&
    operator<<(std::ostream& os, EZ const& m)
    {
      os << std::hexfloat;
      os << m._ezsq;
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, EZ& m)
    {
      assert(is.good());
      is >> m._ezsq;
      return is;
    }

  private:
    EZ_sq _ezsq;
  };
}

#endif
