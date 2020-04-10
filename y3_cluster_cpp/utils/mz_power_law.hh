#ifndef Y3_CLUSTER_MZ_POWER_LAW_HH
#define Y3_CLUSTER_MZ_POWER_LAW_HH

#include <cmath>

namespace y3_cluster {
  // mz_power_law represents a commonly-used power law relationship, with the
  // form
  //       A * m**B * (1+z)**C
  // with A, B and C being constants set in the construction of the power law
  // object.

  class mz_power_law {
  public:
    mz_power_law(double A, double B, double C) noexcept;

    // The function call operator evaluates the power law at the given
    // values of lnM and z. Note that the first parameter is not mass,
    // but ln(mass).
    double operator()(double lnM, double z) const noexcept;

  private:
    double const log_A_;
    double const B_;
    double const C_;
  };
} // namespace y3_cluster

inline y3_cluster::mz_power_law::mz_power_law(double A,
                                              double B,
                                              double C) noexcept
  : log_A_(std::log(A)), B_(B), C_(C)
{}

inline double
y3_cluster::mz_power_law::operator()(double lnM, double z) const noexcept
{
  double const log_res = B_ * lnM + C_ * std::log1p(z) + log_A_;
  return std::exp(log_res);
}
#endif
