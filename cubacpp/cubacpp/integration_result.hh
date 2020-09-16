#ifndef CUBACPP_INTEGRATION_RESULT_HH
#define CUBACPP_INTEGRATION_RESULT_HH

#include "cubacpp/common_results.hh"
#include <array>
#include <ostream>
#include <vector>

namespace cubacpp {

  // All integration functions in cubacpp return an object of one of two types.
  // When integrating a function that has a fixed-size return value (e.g.
  // 'double' or 'std::arary<double, 3>'), the type integration_results<N> is
  // used. When integrating a function that has a return type of std::vector<T>,
  // the type integration_results_v is used.
  // integration_results<1> is specialed to contain the value, error estimate,
  // and "probability" for the result as 'double'; for N != 1,
  // integration_results<N> contains an std::array of values for these
  // quantities. The specialization is provided not to save space (it
  // doesn't) but to provide more convenient notation for the common case.

  struct integration_results_v : public common_results {
    std::vector<double> value;
    std::vector<double> error;
    std::vector<double> prob;

    integration_results_v(std::vector<double> v,
                          std::vector<double> e,
                          std::vector<double> p,
                          long long neval,
                          int nreg,
                          int stat);

    integration_results_v() = default;
    explicit integration_results_v(std::size_t n);
  };

  inline integration_results_v::integration_results_v(std::vector<double> v,
                                                      std::vector<double> e,
                                                      std::vector<double> p,
                                                      long long neval,
                                                      int nreg,
                                                      int stat)
    : common_results(neval, nreg, stat)
    , value(std::move(v))
    , error(std::move(e))
    , prob(std::move(p))
  {}

  inline integration_results_v::integration_results_v(std::size_t n)
    : common_results(), value(n), error(n), prob(n)
  {}

  inline std::ostream&
  operator<<(std::ostream& os, integration_results_v const& r)
  {
    os << "neval: " << r.neval << " nregions:" << r.nregions
       << " status:" << r.status << '\n';
    for (std::size_t i = 0; i != r.value.size(); ++i) {
      os << "Value:" << r.value[i] << " +/- " << r.error[i]
         << " prob: " << r.prob[i] << '\n';
    }
    return os;
  }

  template <std::size_t N>
  struct integration_results : public common_results {
    std::array<double, N> value;
    std::array<double, N> error;
    std::array<double, N> prob;

    integration_results(std::array<double, N> const& v,
                        std::array<double, N> const& e,
                        std::array<double, N> const& p,
                        long long neval,
                        int nreg,
                        int stat)
      : common_results(neval, nreg, stat), value(v), error(e), prob(p)
    {}

    integration_results() = default;
  };

  template <>
  struct integration_results<1> : public common_results {
    double value; // the best estimate of the integral
    double error; // the estimated uncertainty of the integral
    double prob;  // the chisquared probability; see CUBA docs
    integration_results<1>(double v,
                           double e,
                           double p,
                           long long neval,
                           int nreg,
                           int stat)
      : common_results(neval, nreg, stat), value(v), error(e), prob(p)
    {}

    integration_results<1>() = default;
  };

  using integration_result = integration_results<1>;

  template <std::size_t N>
  std::ostream&
  operator<<(std::ostream& os, integration_results<N> const& r)
  {
    os << "neval: " << r.neval << " nregions: " << r.nregions
       << " status: " << r.status << '\n';
    for (std::size_t i = 0; i != N; ++i) {
      os << "Value: " << r.value[i] << " +/- " << r.error[i]
         << " prob: " << r.prob[i] << '\n';
    }
    return os;
  }

  inline std::ostream&
  operator<<(std::ostream& os, integration_result const& r)
  {
    os << "neval: " << r.neval << " nregions: " << r.nregions
       << " status: " << r.status << '\n'
       << "Value: " << r.value << " +/- " << r.error << " prob: " << r.prob;

    return os;
  }
}

#endif
