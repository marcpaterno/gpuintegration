#ifndef Y3_CLUSTER_DV_DO_DZ_T_HH
#define Y3_CLUSTER_DV_DO_DZ_T_HH

#include "ez.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_1d.hh"
#include "utils/read_vector.hh"

#include <istream>
#include <memory>
#include <ostream>
#include <vector>

namespace y3_cluster {
  class DV_DO_DZ_t {
  public:
    DV_DO_DZ_t() = default;

    DV_DO_DZ_t(std::shared_ptr<Interp1D const> da, y3_cluster::EZ ezt, double h)
      : _da(da), _ezt(ezt), _h(h)
    {}

    using doubles = std::vector<double>;

    explicit DV_DO_DZ_t(cosmosis::DataBlock& sample)
      : _da(std::make_shared<y3_cluster::Interp1D const>(
          get_datablock<doubles>(sample, "distances", "z"),
          get_datablock<doubles>(sample, "distances", "d_a")))
      , _ezt(y3_cluster::EZ(sample))
      , _h(get_datablock<double>(sample, "cosmological_parameters", "h0"))
    {}

    double
    operator()(double zt) const
    {
      double const da_z = _da->eval(zt); // da_z needs to be in Mpc
      // Units: (Mpc/h)^3
      // 2997.92 is Hubble distance, c/H_0
      return 2997.92 * (1.0 + zt) * (1.0 + zt) * da_z * _h * da_z * _h /
             _ezt(zt);
    }

    friend std::ostream&
    operator<<(std::ostream& os, DV_DO_DZ_t const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat << *m._da << '\n' << m._ezt << '\n' << m._h;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, DV_DO_DZ_t& m)
    {
      assert(is.good());
      auto da = std::make_shared<Interp1D>();
      is >> *da;
      if (!is) return is;
      EZ ez;
      is >> ez;
      if (!is) return is;
      std::string buffer;
      std::getline(is, buffer);
      if (!is) return is;
      double const h = std::stod(buffer);
      m = DV_DO_DZ_t(da, ez, h);
      return is;
    }

  private:
    std::shared_ptr<Interp1D const> _da;
    y3_cluster::EZ _ezt;
    double _h;
  };
}

#endif
