#ifndef CUDACUHRE_QUAD_UTIL_CUDAARRAY_CUH
#define CUDACUHRE_QUAD_UTIL_CUDAARRAY_CUH

#include <CL/sycl.hpp>
#include <cstring>
#include "oneAPI/pagani/quad/quad.h"
#include "common/oneAPI/cudaMemoryUtil.h"

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
  class cudaDynamicArray{
  public:
    
    cudaDynamicArray(const cudaDynamicArray& a){
        N = a.N;
        _data = quad::cuda_malloc_managed<T>(a.N);
		quad::cuda_memcpy_device_to_device<T>(_data, a._data, a.size());
    }
    
    cudaDynamicArray()
    {
      _data = nullptr;
      N = 0;
    }
    
    //make everything host device
    
    cudaDynamicArray(T const* initData, size_t s) { 
        auto q_ct1 =  sycl::queue(sycl::gpu_selector());
		N = s;
		_data = quad::cuda_malloc_managed<T>(s);
		quad::cuda_memcpy_to_device<T>(_data, initData, s);
    }
    
    void
    Reserve(size_t s)
    {
      N = s;
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
	  _data = quad::cuda_malloc_managed<T>(s);
	  
    }
    
    explicit
    cudaDynamicArray(size_t s)
    {
      N = s;
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
	  _data = quad::cuda_malloc_managed<T>(s);
    }
	
    ~cudaDynamicArray()
    {
		auto q_ct1 =  sycl::queue(sycl::gpu_selector());
		sycl::free(_data, q_ct1);
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

    SYCL_EXTERNAL constexpr std::size_t
    size() const
    {
      return N;
    }
	
    SYCL_EXTERNAL T&
    operator[](std::size_t i)
    {
      return _data[i];
    }

    SYCL_EXTERNAL T
    operator[](std::size_t i) const
    {
      return _data[i];
    }
	
	T*
	data(){
		return _data;
	}

	private:
	
		T* _data;
		size_t N;
  }; // cudaDynamicArray

};

#endif
