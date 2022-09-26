#ifndef ONE_API_QUAD_UTIL_CUDAARRAY_CUH
#define ONE_API_QUAD_UTIL_CUDAARRAY_CUH

#include <cstring>
//#include "cuda/cudaPagani/quad/quad.h"

namespace gpu {
  template <typename T, std::size_t s>
  class cudaArray {
  public:
    void
    Initialize(T const* initData)
    {
      std::memcpy(data, initData, sizeof(T) * s);
    }

    const T*
    begin() const
    {
      return &data[0];
    }

    const T*
    end() const
    {
      return (&data[0] + s);
    }

    constexpr std::size_t
    size() const
    {
      return s;
    }

    T&
    operator[](std::size_t i)
    {
      return data[i];
    }

    T const&
    operator[](std::size_t i) const
    {
      return data[i];
    }

    T data[s];
  };
}

#endif
