#ifndef Y3_CLUSTER_CPP_TRANSFORM_HH
#define Y3_CLUSTER_CPP_TRANSFORM_HH

#include <array>
#include <cstddef> // for std::size_t
#include <functional>
#include <type_traits>
#include <utility>

namespace y3_cluster {

  // Anonymous namespace; implementation details in here.
  namespace {

    // Helper function for y3_cluster::transform
    template <class U, class T, std::size_t N, class F, std::size_t... Is>
    constexpr std::array<U, N> transform_impl(std::array<T, N> const& xs,
                                              F f,
                                              std::index_sequence<Is...>);
  } // namespace

  // transform() takes an std::array and a function that maps the type T held in
  // that array to a possibly different type U, and returns an array of U,
  // containg the result of applying the function to each element of the
  // original array.
  //
  // Note: If the input array is constexpr, the result array is also.
  //
  // USAGE:
  //   std::array<double, 2> const xs { 2.5, 3.5 };
  //   auto ys = y3_cluster::transform(xs, [](double x) {return 2.*x;});
  //   // ys is std::array<double, 2> with values {5.0, 7.0};
  template <class T,
            std::size_t N,
            class F,
            class U = std::invoke_result_t<F, T>>
  constexpr std::array<U, N>
  transform(std::array<T, N> const& xs, F f)
  {
    (void)f; // to silence unused variable warning from compilers that don't
             // know maybe_unused.
    return transform_impl<U>(
      xs, std::forward<F>(f), std::make_index_sequence<N>());
  }

  // Implementation of helper template.
  namespace {
    template <class U, class T, std::size_t N, class F, std::size_t... Is>
    constexpr std::array<U, N>
    transform_impl(std::array<T, N> const& xs, F f, std::index_sequence<Is...>)
    {
      (void)f; // to silence unused variable warning from compilers that don't
               // know maybe_unused.
      return {{f(xs[Is])...}};
    }
  } // namespace

  template <typename R, typename T>
  std::vector<R>
  transform(std::vector<T> const& in, std::function<R(T const&)> f)
  {
    auto ret = std::vector<R>(in.size());
    std::transform(begin(in), end(in), begin(ret), f);
    return ret;
  }

} // namespace y3_cluster

#endif
