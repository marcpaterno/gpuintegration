#ifndef CUBACPP_INTEGRATION_RANGE_HH
#define CUBACPP_INTEGRATION_RANGE_HH

#include <ostream>
#include <stdexcept>
#include <string>

namespace cubacpp {
  class IntegrationRange {

  public:
    // Create an IntegrationRange that goes from 'a' to 'b'.
    IntegrationRange(double a, double b) : _a(a), _range(b - a)
    {
      if (b <= a)
        throw std::logic_error(
          "Upper limit of integration must be larger than lower limit.");
    }

    double
    jacobian() const
    {
      return _range;
    }

    double
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
}

#endif
