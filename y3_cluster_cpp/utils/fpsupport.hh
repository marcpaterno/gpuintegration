#ifndef Y3_CLUSTER_FPSUPPORT_HH
#define Y3_CLUSTER_FPSUPPORT_HH

#include <algorithm>
#include <cmath>

namespace fpsupport {

  inline bool
  is_equivalent(double x, double y, double relTol, double absTol)
  {
    return (std::abs(x - y) <=
            std::max(absTol, relTol * std::max(std::abs(x), std::abs(y))));
  }

  // Detect "icky" values: NaN and infinities
  inline bool
  icky(double x)
  {
    auto const code = std::fpclassify(x);
    return (code == FP_NAN) || (code == FP_INFINITE);
  }
} // namespace fpsupport

#endif
