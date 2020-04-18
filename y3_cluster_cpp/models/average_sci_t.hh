#ifndef Y3_CLUSTER_AVERAGE_SCI_HH
#define Y3_CLUSTER_AVERAGE_SCI_HH

#include "models/ez.hh"
#include "utils/datablock.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_1d.hh"
#include "utils/ndarray.hh"
#include "utils/primitives.hh"

#include <cmath>
#include <iomanip>
#include <istream>
#include <memory>
#include <ostream>

namespace y3_cluster {
  class AVERAGE_SCI_t {
  private:
    std::shared_ptr<Interp1D const> _sci;

  public:
    AVERAGE_SCI_t() = default;

    using doubles = std::vector<double>;

    explicit AVERAGE_SCI_t(std::shared_ptr<Interp1D> table) : _sci(table) {}

    explicit AVERAGE_SCI_t(cosmosis::DataBlock& sample)
      : _sci(std::make_shared<Interp1D const>(
          get_datablock<doubles>(sample, "average_sigma_crit_inv", "zlense"),
          get_datablock<doubles>(sample,
                                 "average_sigma_crit_inv",
                                 "sci_average")))
    {}

    double
    operator()(double zt) const
    {
      return _sci->eval(zt);
    }

    friend std::ostream&
    operator<<(std::ostream& os, AVERAGE_SCI_t const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat << *(m._sci);
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, AVERAGE_SCI_t& m)
    {
      assert(is.good());
      auto table = std::make_shared<Interp1D>();
      is >> *table;
      if (!is) return is;
      m = AVERAGE_SCI_t(table);
      return is;
    }
  };
}

#endif
