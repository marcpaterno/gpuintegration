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
      auto const old_flags = os.flags();
      os << std::hexfloat << m._tau;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, ROFFSET_t& m)
    {
      std::string buffer;
      std::getline(is, buffer);
      if (!is) return is;
      m._tau = std::stod(buffer);
      return is;
    }

  private:
    double _tau = 0.0;
  };
}

#endif
