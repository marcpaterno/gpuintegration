#ifndef Y3_CLUSTER_CPP_ROFFSET_T_HH
#define Y3_CLUSTER_CPP_ROFFSET_T_HH

#include "utils/datablock_reader.hh"
#include <cmath>
#include <iomanip>
#include <istream>
#include <ostream>

namespace y3_cluster {

  class ROFFSET_t {
  public:
    ROFFSET_t() = default;

    explicit ROFFSET_t(double tau) : _tau(tau) {}

    explicit ROFFSET_t(cosmosis::DataBlock& sample)
      : _tau(get_datablock<double>(sample, "cluster_abundance", "roffset_tau"))
    {}

    double
    operator()(double x) const
    {
      // eq. 36
      return x / _tau / _tau * std::exp(-x / _tau);
    }

    friend std::ostream&
    operator<<(std::ostream& os, ROFFSET_t const& m)
    {
      os << std::hexfloat << m._tau;
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, ROFFSET_t& m)
    {
      is >> m._tau;
      return is;
    }

  private:
    double _tau;
  };
}

#endif
