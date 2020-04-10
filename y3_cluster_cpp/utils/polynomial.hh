#ifndef _Y3_POLYNOMIAL_HH
#define _Y3_POLYNOMIAL_HH

#include <array>
#include <cstddef>

namespace y3_cluster {

  /* Represents and calculates a polynomial.
   * A polynomial has a fixed order, or degree, and takes the highest-power
   * coefficient first. e.g.,
   *
   *   f(x) = 3*x^2 - x + 2.5
   *    -> polynomial<3> f{3, -1, 2.5};
   */
  template <std::size_t Order>
  class polynomial {
    const std::array<double, Order> coeffs;

  public:
    constexpr polynomial(std::array<double, Order> coeffs) : coeffs(coeffs) {}

    constexpr double
    operator()(const double x) const
    {
      double out = 0.0;
      for (auto i = 0u; i < Order; i++)
        out = coeffs[i] + x * out;
      return out;
    }
  };

} // namespace y3_cluster

#endif
