#ifndef Y3_CLUSTER_CPP_EZ_SQ_HH
#define Y3_CLUSTER_CPP_EZ_SQ_HH

#include <iomanip>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "utils/str_to_doubles.hh"

namespace y3_cluster {
  class EZ_sq {
  public:
    EZ_sq() = default;

    EZ_sq(double omega_m, double omega_l, double omega_k)
      : _omega_m(omega_m), _omega_l(omega_l), _omega_k(omega_k)
    {}
    double
    operator()(double z) const
    {
      // NOTE: this is valid only for \Lambda CDM cosmology, not wCDM
      double const zplus1 = 1.0 + z;
      return (_omega_m * zplus1 * zplus1 * zplus1 + _omega_k * zplus1 * zplus1 +
              _omega_l);
    }

    friend std::ostream&
    operator<<(std::ostream& os, EZ_sq const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat << m._omega_m << ' ' << m._omega_l << ' '
         << m._omega_k;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, EZ_sq& m)
    {
      assert(is.good());
      std::string buffer;
      std::getline(is, buffer);
      std::vector<double> const vals_read = cosmosis::str_to_doubles(buffer);
      if (vals_read.size() == 3)
      {
        m._omega_m = vals_read[0];
        m._omega_l = vals_read[1];
        m._omega_k = vals_read[2];
      }
      else
      {
        is.setstate(std::ios_base::failbit);
      };
      return is;
    }

  private:
    double _omega_m = 0.0;
    double _omega_l = 0.0;
    double _omega_k = 0.0;
  };
}

#endif
