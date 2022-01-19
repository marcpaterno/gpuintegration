#ifndef CUDACUHRE_QUAD_UTIL_CUDAAPPLY_CUH
#define CUDACUHRE_QUAD_UTIL_CUDAAPPLY_CUH

namespace gpu {
  template <typename T, std::size_t s>
  class cudaArray {
  public:
    __host__ __device__ const T*
    begin() const
    {
      return &data[0];
    }

    __host__ __device__ const T*
    end() const
    {
      return (&data[0] + s);
    }

    __host__ __device__ constexpr std::size_t
    size() const
    {
      return s;
    }

    __host__ __device__ T&
    operator[](std::size_t i)
    {
      return data[i];
    }

    __host__ __device__ T const&
    operator[](std::size_t i) const
    {
      return data[i];
    }

    T data[s];
  };

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