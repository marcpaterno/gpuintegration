#ifndef Y3_CLUSTER_CPP_EZ_SQ_HH
#define Y3_CLUSTER_CPP_EZ_SQ_HH

#include <iomanip>
#include <istream>
#include <ostream>
#include <stdexcept>

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
      os << std::hexfloat << m._omega_m << ' ' << m._omega_l << ' '
         << m._omega_k;
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, EZ_sq& m)
    {
      assert(is.good());
      is >> m._omega_m >> m._omega_l >> m._omega_k;
      if (!is)
        throw std::runtime_error("Failed to read a EZ_sq object");
      return is;
    }

  private:
    double _omega_m;
    double _omega_l;
    double _omega_k;
  };
}

#endif
