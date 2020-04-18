#ifndef Y3_CLUSTER_CPP_LO_LC_T_HH
#define Y3_CLUSTER_CPP_LO_LC_T_HH

#include "utils/datablock_reader.hh"
#include "utils/primitives.hh"

#include <cmath>
#include <iomanip>
#include <istream>
#include <ostream>

namespace y3_cluster {

  class LO_LC_t {
  public:
    LO_LC_t() = default;

    LO_LC_t(double alpha, double a, double b, double R_lambda)
      : _alpha(alpha), _a(a), _b(b), _R_lambda(R_lambda)
    {}

    explicit LO_LC_t(cosmosis::DataBlock& sample)
      : _alpha(
          get_datablock<double>(sample, "cluster_abundance", "LO_LC_alpha"))
      , _a(get_datablock<double>(sample, "cluster_abundance", "LO_LC_a"))
      , _b(get_datablock<double>(sample, "cluster_abundance", "LO_LC_b"))
      , _R_lambda(
          get_datablock<double>(sample, "cluster_abundance", "LO_LC_R_lambda"))
    {}

    double
    operator()(double lo, double lc, double R_mis) const
    {
      /* eq. (35) */
      double x = R_mis / _R_lambda;
      double y = lo / lc;
      double mu_y = std::exp(-x * x / _alpha / _alpha);
      double sigma_y = _a * std::atan(_b * x);
      // Need 1/lc scaling for total probability = 1
      return y3_cluster::gaussian(y, mu_y, sigma_y) / lc;
    }

    friend std::ostream&
    operator<<(std::ostream& os, LO_LC_t const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat << m._alpha << ' ' << m._a << ' ' << m._b << ' '
         << m._R_lambda;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, LO_LC_t& m)
    {
      std::string buffer;
      std::getline(is, buffer);
      std::vector<double> const vals_read = cosmosis::str_to_doubles(buffer);
      if (vals_read.size() == 4)
      {
        m._alpha = vals_read[0];
        m._a = vals_read[1];
        m._b = vals_read[2];
        m._R_lambda = vals_read[3];
      }
      else {
        is.setstate(std::ios_base::failbit);
      }
      return is;
    }

  private:
    double _alpha = 0.0;
    double _a = 0.0;
    double _b = 0.0;
    double _R_lambda = 0.0;
  };
}

#endif
