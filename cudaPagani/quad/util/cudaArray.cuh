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

  template <typename T>
  class cudaDynamicArray {
  public:
  
    cudaDynamicArray(){
       data = nullptr;
       N = 0;
    }
    
    // host-only function
    void
    Initialize(T const* initData, size_t s)
    {
      N = s;
      cudaMallocManaged((void**)&data, sizeof(T) * s);
      cudaMemcpy(data, initData, sizeof(T) * s, cudaMemcpyHostToDevice);
    }
    
    __host__ __device__
    ~cudaDynamicArray() {
      #ifndef __CUDACC__
      cudaFree(data);
      #endif
   }

    __host__ __device__ const T*
    begin() const
    {
      return &data[0];
    }

    __host__ __device__ const T*
    end() const
    {
      return (&data[0] + N);
    }

    __host__ __device__  constexpr std::size_t
    size() const
    {
      return N;
    }

    cudaDynamicArray&
    operator=(const cudaDynamicArray& source)
    {
      cudaMallocManaged((void**)&data, sizeof(T) * source.size());
      cudaMemcpy(data, source.data, sizeof(T) * N, cudaMemcpyHostToDevice);
      return *this;
    }

    __host__ __device__ T&
    operator[](std::size_t i)
    {
      return data[i];
    }

    __host__ __device__ T
    operator[](std::size_t i) const
    {
      return data[i];
    }

    T* data;
    size_t N;
  }; // cudaDynamicArray

};

#endif