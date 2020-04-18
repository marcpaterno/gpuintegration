#ifndef Y3_CLUSTER_DEL_SIG_TOM_HH
#define Y3_CLUSTER_DEL_SIG_TOM_HH

#include "models/ez.hh"
#include "utils/datablock.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_2d.hh"
#include "utils/ndarray.hh"
#include "utils/primitives.hh"

#include <cmath>
#include <memory>

namespace y3_cluster {
  class SIG_SUM {
  private:
    std::shared_ptr<Interp2D const> _sigma1;
    std::shared_ptr<Interp2D const> _sigma2;
    std::shared_ptr<Interp2D const> _bias;

  public:
    using doubles = std::vector<double>;

    SIG_SUM() = default;

    SIG_SUM(std::shared_ptr<Interp2D const> sigma1,
            std::shared_ptr<Interp2D const> sigma2,
            std::shared_ptr<Interp2D const> bias)
      : _sigma1(sigma1), _sigma2(sigma2), _bias(bias)
    {}

    explicit SIG_SUM(cosmosis::DataBlock& sample)
      : _sigma1(std::make_shared<Interp2D const>(
          get_datablock<doubles>(sample, "deltasigma", "r_sigma_deltasigma"),
          get_datablock<doubles>(sample, "deltasigma", "lnM"),
          get_datablock<cosmosis::ndarray<double>>(sample,
                                                   "deltasigma",
                                                   "sigma_1")))
      , _sigma2(std::make_shared<Interp2D const>(
          get_datablock<doubles>(sample, "deltasigma", "r_sigma_deltasigma"),
          get_datablock<doubles>(sample, "matter_power_lin", "z"),
          get_datablock<cosmosis::ndarray<double>>(sample,
                                                   "deltasigma",
                                                   "sigma_2")))
      , _bias(std::make_shared<Interp2D const>(
          get_datablock<doubles>(sample, "matter_power_lin", "z"),
          get_datablock<doubles>(sample, "deltasigma", "lnM"),
          get_datablock<cosmosis::ndarray<double>>(sample,
                                                   "deltasigma",
                                                   "bias")))
    {}

    double
    operator()(double r, double lnM, double zt) const
    /*r in h^-1 Mpc */ /* M in h^-1 M_solar, represents M_{200} */
    {
      double _sig_1 = _sigma1->clamp(r, lnM);
      double _sig_2 = _bias->clamp(zt, lnM) * _sigma2->clamp(r, zt);
      // TODO: h factor?
      // if (del_sig_1 >= del_sig_2) {
      // return (1.+zt)*(1.+zt)*(1.+zt)*(_sig_1+_sig_2);
      return (_sig_1 + _sig_2);
      /*} else {
        return 1e12*(1.+zt)*(1.+zt)*(1.+zt)*del_sig_2;
      } */
    }

    friend std::ostream&
    operator<<(std::ostream& os, SIG_SUM const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat << *m._sigma1 << '\n' << *m._sigma2 << '\n' << *m._bias;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, SIG_SUM& m)
    {
      auto sigma1 = std::make_shared<Interp2D>();
      is >> *sigma1;
      if (!is) return is;
      auto sigma2 = std::make_shared<Interp2D>();
      is >> *sigma2;
      if (!is) return is;
      auto bias = std::make_shared<Interp2D>();
      is >> *bias;
      if (!is) return is;
      m = SIG_SUM(sigma1, sigma2, bias);
      return is;
    }
  };
}

#endif
