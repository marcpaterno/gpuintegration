#ifndef CUDACUHRE_QUAD_UTIL_CUDAAPPLY_CUH
#define CUDACUHRE_QUAD_UTIL_CUDAAPPLY_CUH

#include "cudaArray.cuh"
namespace gpu {

  namespace detail {
    template <class F, size_t N, std::size_t... I>
    __device__ double
    apply_impl(F&& f,
               gpu::cudaArray<double, N> const& data,
               std::index_sequence<I...>)
    {
      return f(data[I]...);
    };
  }

  template <class F, size_t N>
  __device__ double
  // Unsure if we need to pass 'f' by value, for GPU execution
  apply(F&& f, gpu::cudaArray<double, N> const& data)
  {
    return detail::apply_impl(
      std::forward<F>(f), data, std::make_index_sequence<N>());
  }
}

#endif