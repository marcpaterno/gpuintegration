#ifndef CUDACUHRE_QUAD_UTIL_CUDAARRAY_CUH
#define CUDACUHRE_QUAD_UTIL_CUDAARRAY_CUH

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <cstring>
#include "oneAPI/pagani/quad/quad.h"
#include "oneAPI/pagani/quad/util/cudaMemoryUtil.h"

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
	//#ifndef DPCT_COMPATIBILITY_TEMP
            N = a.N;
            //cudaMallocManaged((void**)&data, sizeof(T) * a.N);
            &data = quad::cuda_malloc_managed<T>(a.N);
			memcpy(data, a.data, sizeof(T) * a.N);
       /* #else
            //can't instantiate on device and then access on host
            N = a.N;
            data = new T[a.N];
            memcpy(data, a.data, sizeof(T) * a.N);*/
        //#endif
    }
    
    
    cudaDynamicArray()
    {
      data = nullptr;
      N = 0;
    }
    
    //make everything host device
    
    cudaDynamicArray(T const* initData, size_t s) { 
        Initialize(initData, s); 
    }
    
    void    
    Initialize(T const* initData, size_t s)
    {
      //dpct::device_ext& dev_ct1 = dpct::get_current_device();
      //sycl::queue& q_ct1 = dev_ct1.default_queue();
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      N = s;
      data = (T*)sycl::malloc_shared(sizeof(T) * s, q_ct1);
      q_ct1.memcpy(data, initData, sizeof(T) * s).wait();
    }

    void
    Reserve(size_t s)
    {
      N = s;
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      data = (T*)sycl::malloc_shared(sizeof(T) * s, q_ct1);
    }
    
    explicit
    cudaDynamicArray(size_t s)
    {
      N = s;
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      data = (T*)sycl::malloc_shared(sizeof(T) * s, q_ct1);
    }
    ~cudaDynamicArray()
    {
#ifndef SYCL_LANGUAGE_VERSION
      cudaFree(data);
#endif
    }

    const T*
    begin() const
    {
      return &data[0];
    }

    const T*
    end() const
    {
      return (&data[0] + N);
    }

    constexpr std::size_t
    size() const
    {
      return N;
    }

    /*cudaDynamicArray&
    operator=(const cudaDynamicArray& source)
    {
      cudaMallocManaged((void**)&data, sizeof(T) * source.size());
      cudaMemcpy(data, source.data, sizeof(T) * N, cudaMemcpyHostToDevice);
      return *this;
    }*/

    T&
    operator[](std::size_t i)
    {
      return data[i];
    }

    T
    operator[](std::size_t i) const
    {
      return data[i];
    }

    T* data;
    size_t N;
  }; // cudaDynamicArray

};

#endif
