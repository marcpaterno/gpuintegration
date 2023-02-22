#ifndef CUDACUHRE_QUAD_UTIL_CUDAARRAY_CUH
#define CUDACUHRE_QUAD_UTIL_CUDAARRAY_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstring>
#include <array>
#include "dpct-exp/cuda/pagani/quad/quad.h"
#include "dpct-exp/common/cuda/cudaMemoryUtil.h"

// cudaArray is meant to allow in-kernel use of functions that expect std::array interface, e.g. std::forward

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

  template <typename T>
  class cudaDynamicArray {
  public:
    
    cudaDynamicArray(const cudaDynamicArray& a)
    {
#ifndef DPCT_COMPATIBILITY_TEMP
      N = a.N;
	  _data = quad::cuda_malloc_managed<T>(N);
      memcpy(_data, a._data, sizeof(T) * a.N);
#else
      // can't instantiate on device and then access on host
      N = a.N;
      _data = new T[a.N];
      memcpy(_data, a._data, sizeof(T) * a.N);
#endif
    }

    
    cudaDynamicArray()
    {
      _data = nullptr;
      N = 0;
    }

    // make everything host device

    cudaDynamicArray(T const* initData, size_t s)
    {
      N = s;
	  _data = quad::cuda_malloc_managed<T>(s);
	  quad::cuda_memcpy_to_device<T>(_data, initData, s);
    }

	explicit cudaDynamicArray(size_t s)
    {
      N = s;
	  _data = quad::cuda_malloc_managed<T>(s);
    }

    void
    Reserve(size_t s)
    {
      N = s;
	  _data = quad::cuda_malloc_managed<T>(s);
    }

    ~cudaDynamicArray()
    {
#ifndef SYCL_LANGUAGE_VERSION
      cudaFree(_data);
#endif
    }

    const T*
    begin() const
    {
      return &_data[0];
    }

    const T*
    end() const
    {
      return (&_data[0] + N);
    }

    constexpr std::size_t
    size() const
    {
      return N;
    }

	
	T*
	data(){
		return _data;
	}
	
    T&
    operator[](std::size_t i)
    {
      return _data[i];
    }

    T
    operator[](std::size_t i) const
    {
      return _data[i];
    }
	
	
	private: 
	
		T* _data;
		size_t N;
  }; // cudaDynamicArray

};

#endif
