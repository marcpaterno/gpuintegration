#ifndef CUBACPP_INTEGRAND_TRAITS_HH
#define CUBACPP_INTEGRAND_TRAITS_HH

#include "cubacpp/arity.hh"
#include "cubacpp/integration_result.hh"
#include <array>
#include <tuple> // for std::apply
#include <vector>

namespace cubacpp {

  // Forward declarations.
  template <typename F>
  struct integrand_traits;

  namespace detail {

    template <typename F, typename RES>
    std::tuple<std::size_t, double*, double*, double*> make_cuba_args(
      F f,
      RES& result);

    // The following names should not be used outside this header.
    namespace integration_traits_private {

      // integration_results_type_for<T>::type is the return type for
      // integration routines when integrating a function with return type T. No
      // general case is supplied; there are only specializations.
      template <typename>
      struct integration_results_type_for;

      template <>
      struct integration_results_type_for<std::vector<double>> {
        using type = integration_results_v;
      };

      template <>
      struct integration_results_type_for<double> {
        using type = integration_result;
      };

      template <std::size_t N>
      struct integration_results_type_for<std::array<double, N>> {
        using type = integration_results<N>;
      };

      template <typename F>
      std::size_t
      runtime_ncomp(F f)
      {
        typename integrand_traits<F>::arg_type args{};
        return std::apply(f, args).size();
      }

      template <typename F>
      constexpr std::size_t
      ncomp()
      {
        using rt = typename integrand_traits<F>::function_return_type;
        return sizeof(rt) / sizeof(double);
      }

    }
  }

  // integrand_traits<F> provides information about the callable type
  // F.
  // The static data member 'ndim' reports the number of arguments
  // of the callable object (the dimensionality of the integral to be
  // calculated).
  //
  // The nested type 'function_return_type' reports the return
  // type of the call. Expected types are 'double', or 'std::array<double,
  // N>' for any positive integral N, or std::vector<double>.
  //
  // The nested type 'integration_results_type' reports the type returned
  // from integrating this integrand.
  //
  // The nested type 'arg_type' reports the array typed used to pass
  // arguments to the wrapped function.
  template <typename F>
  struct integrand_traits {
    static constexpr int ndim = arity<F>();

    using function_return_type =
      decltype(std::apply(std::declval<F>(), std::array<double, ndim>()));

    using integration_results_type =
      typename detail::integration_traits_private::integration_results_type_for<
        function_return_type>::type;

    using arg_type = std::array<double, ndim>;
  };

  namespace detail {

    template <typename F>
    static typename integrand_traits<F>::integration_results_type
    default_integration_results([[maybe_unused]] F f)
    {
      using irtype = typename integrand_traits<F>::integration_results_type;
      using frtype = typename integrand_traits<F>::function_return_type;
      irtype result;
      if constexpr (std::is_same_v<frtype, std::vector<double>>) {
        result = irtype(integration_traits_private::runtime_ncomp(f));
      };
      return result;
    }

    // make_cuba_args<F,R> returns a tuple of values that are to passed to the
    // CUBA integration routines. The pointers are CUBA's output arguments;
    // the addresses returned are the addresses in the object 'result', which is
    // to be returned by the cubacpp integration function.
    template <typename F, typename RES>
    std::tuple<std::size_t, double*, double*, double*>
    make_cuba_args(F f, RES& result)
    {
      // Result is here to silence a compilation warning from nvcc 11.0.
      // nvcc 11.0 also has trouble with make_tuple.
      std::tuple<std::size_t, double*, double*, double*> tup;
      if constexpr (std::is_same_v<typename integrand_traits<decltype(f)>::function_return_type,
                                   std::vector<double>>) {
        tup = std::make_tuple(detail::integration_traits_private::runtime_ncomp(f),
                              result.value.data(),
                              result.error.data(),
                              result.prob.data());
      } else {
        tup = std::make_tuple(detail::integration_traits_private::ncomp<F>(),
                              (double*)&result.value,
                              (double*)&result.error,
                              (double*)&result.prob);
      }
      return tup;
    }

  }
}
#endif
