#ifndef Y3_CLUSTER_CPP_HMF_T_HH
#define Y3_CLUSTER_CPP_HMF_T_HH

#include "utils/datablock.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_2d.hh"
#include "utils/ndarray.hh"
#include "utils/read_vector.hh"

#include <iomanip>
#include <istream>
#include <memory>
#include <ostream>

namespace y3_cluster {

  namespace {
    inline std::vector<double>
    _adjust_to_log(cosmosis::DataBlock& db, const std::vector<double>& masses)
    {
      std::vector<double> output = masses;
      double omega_m =
        get_datablock<double>(db, "cosmological_parameters", "omega_M");
      double omega_mu =
        get_datablock<double>(db, "cosmological_parameters", "omega_nu");
      for (auto& x : output)
        x = std::log(x * (omega_m - omega_mu));
      return output;
    }

  }

  class HMF_t {
  public:
    HMF_t() = default;

    HMF_t(std::shared_ptr<Interp2D const> nmz, double s, double q)
      : _nmz(nmz), _s(s), _q(q)
    {}

    using doubles = std::vector<double>;

    explicit HMF_t(cosmosis::DataBlock& sample)
      : _nmz(std::make_shared<Interp2D const>(
          _adjust_to_log(
            sample,
            get_datablock<doubles>(sample, "mass_function", "m_h")),
          get_datablock<doubles>(sample, "mass_function", "z"),
          get_datablock<cosmosis::ndarray<double>>(sample,
                                                   "mass_function",
                                                   "dndlnmh")))
      , _s(get_datablock<double>(sample, "cluster_abundance", "hmf_s"))
      , _q(get_datablock<double>(sample, "cluster_abundance", "hmf_q"))
    {}

    double
    operator()(double lnM, double zt) const
    {
      return _nmz->clamp(lnM, zt) *
             (_s * (lnM * 0.4342944819 - 13.8124426028) + _q);
      // 0.4342944819 is log(e)
    }

    friend std::ostream&
    operator<<(std::ostream& os, HMF_t const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat;
      os << *(m._nmz) << '\n' << m._s << ' ' << m._q;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, HMF_t& m)
    {
      assert(is.good());
      auto table = std::make_shared<Interp2D>();
      is >> *table;
      std::string buffer;
      std::getline(is, buffer);
      std::vector<double> const vals_read = cosmosis::str_to_doubles(buffer);
      if (vals_read.size() == 2)
      {
        m = HMF_t(table, vals_read[0], vals_read[1]);
      }
      else
      {
        is.setstate(std::ios_base::failbit);
      };
      return is;
    }

  private:
    std::shared_ptr<Interp2D const> _nmz;
    double _s = 0.0;
    double _q = 0.0;
  };
}

#endif
