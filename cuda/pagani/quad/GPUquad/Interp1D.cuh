#ifndef GPUQUADINTERP1D_H
#define GPUQUADINTERP1D_H

#include "cuda/pagani/quad/quad.h"
#include "cuda/pagani/quad/util/cudaArray.cuh"
#include "cuda/pagani/quad/util/cudaMemoryUtil.h"
#include "cuda/pagani/quad/util/cudaTimerUtil.h"
#include "cuda/pagani/quad/util/str_to_doubles.hh"
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace quad {

  // Helper struct to describe an index range.
  template<typename T>
  struct IndexRange {
    size_t left = 0;
    size_t right = 0;

    __device__ __host__ bool is_valid() const;
    __device__ __host__ IndexRange<T> middle() const;
    __device__ __host__ void adjust_edges(T const* xs,
                                          T val,
                                          IndexRange<T> middle);
  };

  template<typename T>
  class Interp1D {

    size_t _cols = 0;
    T* _xs = nullptr;
    T* _zs = nullptr;

    // Copy the pointed-to arrays into managed memory.
    // The class interface guarantees that the array lengths
    // match the memory allocation done.
    void _initialize(T const* x, T const* z);

    __device__ __host__ bool _in_range(T val, IndexRange<T> range) const;
    __device__ __host__ IndexRange<T>
    _find_smallest__index_range(T val) const;

  public:
    size_t
    get_device_mem_footprint()
    {
      return 8 * _cols * _cols;
    }

    Interp1D();
    Interp1D(const Interp1D& source);
    Interp1D& operator=(Interp1D const& rhs);
    Interp1D(Interp1D&&) = delete;
    Interp1D& operator=(Interp1D&&) = delete;
    ~Interp1D();

    template <size_t M>
    Interp1D(std::array<T, M> const& xs, std::array<T, M> const& zs);

    Interp1D(T const* xs, T const* zs, size_t cols);

    void swap(Interp1D<T>& other);
    __device__ __host__ T operator()(T x) const;
    __device__ __host__ T min_x() const;
    __device__ __host__ T max_x() const;
    __device__ __host__ T do_clamp(T v, T lo, T hi) const;
    __device__ __host__ T eval(T x) const;
    __device__ __host__ T clamp(T x) const;

	template<typename TT>
    friend std::istream& operator>>(std::istream& is, Interp1D<TT>& interp);
  };
}

template<typename T>
inline __device__ __host__ bool
quad::IndexRange<T>::is_valid() const
{
  return left < right;
}

template<typename T>
inline __device__ __host__ quad::IndexRange<T>
quad::IndexRange<T>::middle() const
{
  size_t const mid = static_cast<size_t>((left + right) * 0.5);
  return {mid, mid + 1};
}

template<typename T>
inline __device__ __host__ void
quad::IndexRange<T>::adjust_edges(T const* xs, T val, IndexRange<T> middle)
{

  if (xs[middle.left] > val) {
    right = middle.left; // shrink the right side
  } else {
    left = middle.right; // shrink the left side
  }
}

template<typename T>
inline void
quad::Interp1D<T>::_initialize(T const* x, T const* z)
{
  if (_cols > 1000000) {
    std::cerr << "Interp1D::_initilize called when _cols=" << _cols << '\n';
    std::abort();
  }
  _xs = cuda_malloc<T>(_cols);
  cuda_memcpy_to_device<T>(_xs, x, _cols);
  _zs = cuda_malloc<T>(_cols);
  cuda_memcpy_to_device<T>(_zs, z, _cols);
}

template<typename T>
inline quad::Interp1D<T>::Interp1D() {}

template<typename T>
inline quad::Interp1D<T>::Interp1D(const Interp1D<T>& source) : _cols(source._cols)
{
  _initialize(source._xs, source._zs);
}

template<typename T>
inline quad::Interp1D<T>&
quad::Interp1D<T>::operator=(Interp1D<T> const& rhs)
{
  Interp1D<T> tmp(rhs);
  swap(tmp);
  return *this;
}

template<typename T>
inline quad::Interp1D<T>::~Interp1D()
{
  cudaFree(_zs);
  cudaFree(_xs);
}

template<typename T>
template <size_t M>
quad::Interp1D<T>::Interp1D(std::array<T, M> const& xs,
                         std::array<T, M> const& zs)
  : _cols(M)
{
  _initialize(xs.data(), zs.data());
}

template<typename T>
inline quad::Interp1D<T>::Interp1D(T const* xs, T const* zs, size_t cols)
  : _cols(cols)
{
  _initialize(xs, zs);
}

template<typename T>
inline void
quad::Interp1D<T>::swap(Interp1D& other)
{
  std::swap(_cols, other._cols);
  std::swap(_zs, other._zs);
  std::swap(_xs, other._xs);
}

template<typename T>
inline __device__ __host__ bool
quad::Interp1D<T>::_in_range(const T val, IndexRange<T> const range) const
{
  return (_xs[range.left] <= val) && (_xs[range.right] >= val);
}

template<typename T>
inline __device__ __host__ quad::IndexRange<T>
quad::Interp1D<T>::_find_smallest__index_range(const T val) const
{
  // we don't check if val is in the current range. clamp makes sure we dont
  // pass values that exceed min/max, right?
  IndexRange<T> current_range{0, _cols - 1};

  while (current_range.is_valid()) {
    IndexRange<T> smaller_candidate_range = current_range.middle();
    if (_in_range(val, smaller_candidate_range)) {
      return smaller_candidate_range;
    }
    current_range.adjust_edges(_xs, val, smaller_candidate_range);
  }
  return current_range;
}

template<typename T>
inline __device__ __host__ T
quad::Interp1D<T>::operator()(T x) const
{
  auto [x0_index, x1_index] = _find_smallest__index_range(x);
  const T y0 = _zs[x0_index];
  const T y1 = _zs[x1_index];
  const T x0 = _xs[x0_index];
  const T x1 = _xs[x1_index];
  const T y = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0);
  return y;
}

template<typename T>
inline __device__ __host__ T
quad::Interp1D<T>::min_x() const
{
  return _xs[0];
}

template<typename T>
inline __device__ __host__ T
quad::Interp1D<T>::max_x() const
{
  return _xs[_cols - 1];
}

template<typename T>
inline __device__ __host__ T
quad::Interp1D<T>::do_clamp(T v, T lo, T hi) const
{
  assert(!(hi < lo));
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

template<typename T>
inline __device__ __host__ T
quad::Interp1D<T>::eval(T x) const
{
  return this->operator()(x);
}

template<typename T>
__device__ __host__ T
quad::Interp1D<T>::clamp(T x) const
{
  return eval(do_clamp(x, min_x(), max_x()));
}

namespace quad {
	
  template<typename T>
  inline std::istream&
  operator>>(std::istream& is, quad::Interp1D<T>& interp)
  {
    assert(is.good());
    std::string buffer;
    std::getline(is, buffer);
    std::vector<T> xs = str_to_doubles(buffer);
    std::getline(is, buffer);
    std::vector<T> zs = str_to_doubles(buffer);

    cudaMallocManaged((void**)&(*&interp), sizeof(quad::Interp1D<T>));
    cudaDeviceSynchronize();

    interp._cols = xs.size();

    cudaMallocManaged((void**)&interp._xs, sizeof(T) * xs.size());
    cudaDeviceSynchronize();
    cudaMallocManaged((void**)&interp._zs, sizeof(T) * zs.size());
    cudaDeviceSynchronize();

    memcpy(interp._xs, xs.data(), sizeof(T) * xs.size());
    memcpy(interp._zs, zs.data(), sizeof(T) * zs.size());

    return is;
  }
}
#endif
