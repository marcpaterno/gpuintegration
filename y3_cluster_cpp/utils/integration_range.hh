#ifndef Y3_CLUSTER_INTEGRATION_RANGE_HH
#define Y3_CLUSTER_INTEGRATION_RANGE_HH

#include "utils/datablock.hh"
#include <ostream>
#include <stdexcept>
#include <string>

// Use of this header should be removed when the deprecated constructors
// are removed.
#include "datablock_reader.hh"
namespace y3_cluster {
  class IntegrationRange {
  public:
    // This constructor should be removed.
    // It produces an IntegrationRange of zero length, which can not
    // be used for integration.
    IntegrationRange() = default;

    // Create an IntegrationRange that goes from 'a' to 'b'.
    IntegrationRange(double a, double b) : _a(a), _range(b - a)
    {
      if (_range == 0.0)
        throw std::logic_error("zero-length IntegrationRange");
    }

    // This constructor should be removed.
    // * it can only be used from within a module labeled "cluster_abundance".
    // * it reproduced behavior implemented in the constructor from 2 doubles.
    IntegrationRange(cosmosis::DataBlock& sample, std::string const& var)
    {
      double b;
      std::string min = var + "_min";
      std::string max = var + "_max";
      _a = get_datablock<double>(sample, "cluster_abundance", min.c_str());
      b = get_datablock<double>(sample, "cluster_abundance", max.c_str());
      _range = b - _a;
      if (_range == 0.0)
        throw std::logic_error("zero-length IntegrationRange");
    }

    [[nodiscard]] double
    jacobian() const
    {
      return _range;
    }

    [[nodiscard]] double
    transform(double x) const
    {
      return _range * x + _a;
    }

    friend std::ostream& operator<<(std::ostream& os, const IntegrationRange&);

  private:
    double _a{0};
    double _range{0};
  };

  inline std::ostream&
  operator<<(std::ostream& os, const y3_cluster::IntegrationRange& ir)
  {
    return os << "[" << ir._a << ", " << ir._a + ir._range << "]";
  }

  // Create an integration range from the named parameter in the
  // DataBlock.
  //
  // NOTE: The DataBlock is taken by non-const reference because
  // the value accessors are non-const.
  inline IntegrationRange
  make_integration_range(cosmosis::DataBlock& configuration,
                         std::string const& section,
                         std::string const& var)
  {
    std::string min(var);
    min += "_min";
    std::string max(var);
    max += "_max";

    double const a = configuration.view<double>(section, min);
    double const b = configuration.view<double>(section, max);

    return IntegrationRange(a, b);
  }

} // namespace y3_cluster

#endif
