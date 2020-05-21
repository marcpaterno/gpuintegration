#ifndef CUDACUHRE_QUAD_UTIL_CUDAARRAY_CUH
#define CUDACUHRE_QUAD_UTIL_CUDAARRAY_CUH

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

    __host__ __device__ T& operator[](std::size_t i) { return data[i]; }

    __host__ __device__ T const& operator[](std::size_t i) const
    {
      return data[i];
    }

    T data[s];
  };
};

#endif