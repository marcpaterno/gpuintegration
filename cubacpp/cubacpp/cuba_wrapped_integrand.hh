#ifndef CUBACPP_CUBA_WRAPPED_INTEGRAND_HH
#define CUBACPP_CUBA_WRAPPED_INTEGRAND_HH

#include "cubacpp/integrand_traits.hh"
#include "cubacpp/integration_volume.hh"
#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple> // for std::apply
#include <vector>

namespace cubacpp::detail {

  template <std::size_t N>
  std::array<double, N>
  to_array(double const* x)
  {
    std::array<double, N> res;
    for (std::size_t i = 0; i < N; ++i)
      res[i] = x[i];
    return res;
  }

  // definite_integral<F> is the "userdata" object whose address is passed back
  // to the function the integration routines call at each invocation. cubacpp
  // uses this object to peform the scaling of arguments from the range
  //     0 <= u <= 1
  // which CUBA uses, to
  //     a <= x <= b
  // which the user-supplied callable object expects. The values of 'a' and 'b'
  // are determined by the volume of integration supplied by the user in his
  // call to the integration routine.
  template <typename F>
  struct definite_integral {
    F* uf = nullptr; // not F const*, to support C-style functions.
    integration_volume_for_t<F> const* pvol = nullptr;

    // We define this constructor so that definite_integral will have the
    // correct implicit deduction guide.
    definite_integral(F* pfun, integration_volume_for_t<F> const* pv);
  };

  template <typename F>
  definite_integral<F>::definite_integral(
    F* pfun,
    const cubacpp::integration_volume_for_t<F>* pv)
    : uf(pfun), pvol(pv)
  {}

  // cuba_wrapped_integrand<F> is a free function with a signature that is
  // expected by the CUBA integration routines. It uses an IntegrationVolume
  // to map the argument values supplied by the CUBA integration routines (which
  // are always within the unit hypercube) to values in the volume specified.
  template <typename F>
  int
  cuba_wrapped_integrand(int const* reported_dimensions,
                         const double* x,
                         [[maybe_unused]] int const* ncomp,
                         double* f,
                         void* obj)
  {
    // N is the number of arguments the function F expects.
    constexpr int N = integrand_traits<F>::ndim;
    if (*reported_dimensions != N)
      return -999;

    auto pAdapterThing = reinterpret_cast<definite_integral<F>*>(obj);

    auto scaled_args = pAdapterThing->pvol->transform(to_array<N>(x));
    // The type of 'res' will be:
    //    std::array<double, n> or
    //    std::vector<double> or
    //    double
    // The call to std::apply is what actually invokes the user's integrand.
    auto const res = std::apply(*(pAdapterThing->uf), scaled_args);

    if constexpr (std::is_same_v<decltype(res), double const>) {
      *f = res * pAdapterThing->pvol->jacobian();
    } else {
      if (static_cast<std::size_t>(*ncomp) != res.size())
        return -999;
      for (auto const val : res) {
        *f = val * pAdapterThing->pvol->jacobian();
        ++f;
      }
    };
    return 0;
  }
}

#endif
