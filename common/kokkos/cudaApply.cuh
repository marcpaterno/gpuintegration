#ifndef KOKKOS_QUAD_UTIL_CUDAAPPLY_CUH
#define KOKKOS_QUAD_UTIL_CUDAAPPLY_CUH

#include "common/kokkos/cudaMemoryUtil.h"

namespace gpu {
  template <typename T, std::size_t s>
  class cudaArray {
  public:
    KOKKOS_INLINE_FUNCTION const T*
    begin() const
    {
      return &data[0];
    }

    KOKKOS_INLINE_FUNCTION const T*
    end() const
    {
      return (&data[0] + s);
    }

    KOKKOS_INLINE_FUNCTION constexpr std::size_t
    size() const
    {
      return s;
    }

    KOKKOS_INLINE_FUNCTION T&
    operator[](std::size_t i)
    {
      return data[i];
    }

    KOKKOS_INLINE_FUNCTION T const&
    operator[](std::size_t i) const
    {
      return data[i];
    }

    T data[s];
  };

  namespace detail {
    template <class F, size_t N, std::size_t... I>
    KOKKOS_INLINE_FUNCTION double
    apply_impl(F&& f,
               gpu::cudaArray<double, N> const& data,
               std::index_sequence<I...>)
    {
      return f(data[I]...);
    };
  }

  template <class F, size_t N>
  KOKKOS_INLINE_FUNCTION double
  // Unsure if we need to pass 'f' by value, for GPU execution
  apply(F&& f, gpu::cudaArray<double, N> const& data)
  {
    return detail::apply_impl(
      std::forward<F>(f), data, std::make_index_sequence<N>());
  }
}

#endif