#ifndef CUBACPP_ARITY_HH
#define CUBACPP_ARITY_HH

#include <type_traits>

// clang++, as of v 5.0.1, does not yet have std::invocable. However, it does
// have std::__invokable, that does the job. Apple's clang has different version
// numbering, and also needs this fix, until Xcode 10.0.

#ifdef __clang__
#if (__clang_major__ < 6) || (__apple_build_version__ < 10001145)
namespace std {
  template <class F, class... ARGS>
  using is_invocable = __invokable<F, ARGS...>;
}
#endif
#endif

namespace cubacpp {
  namespace detail {

    // Forward declaration for helper functions.
    template <typename F, typename ArgType, int Limit, typename... Args>
    constexpr int nargs_with_type_impl();

    template <typename F, typename T, int Limit = 30>
    constexpr int nargs_with_type();

  } // detail

  // Return the number of arguments of type T, up to limit Limit, with which the
  // callable type F can be invoked. The limit is provided in order to avoid
  // excessive compilation times in the case that F can *not* be invoked with
  // any number of arguments of type T (because, for example, it also takes some
  // argument of a different type).
  template <typename F>
  constexpr int
  arity()
  {
    return detail::nargs_with_type<F, double>();
  }
}

// Helper function implementations below

namespace cubacpp::detail {
  template <typename F, typename T, int Limit>
  constexpr int
  nargs_with_type()
  {
    return nargs_with_type_impl<F, T, Limit + 1>();
  }

  template <typename F, typename ArgType, int Limit, typename... Args>
  constexpr int
  nargs_with_type_impl()
  {
    int result = -1;
    if constexpr (sizeof...(Args) == Limit) {
      // result is already -1
    } else if (std::is_invocable<F, Args...>::value) {
      result = sizeof...(Args);
    } else {
      return nargs_with_type_impl<F,
                                  ArgType,
                                  Limit,
                                  ArgType,
                                  Args...>(); // Add another argument
    }
    return result;
  }

} // cubacpp::detail

#endif
