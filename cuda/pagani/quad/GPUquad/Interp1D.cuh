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
  struct IndexRange {
    size_t left = 0;
    size_t right = 0;

    __device__ __host__ bool is_valid() const;
    __device__ __host__ IndexRange middle() const;
    __device__ __host__ void adjust_edges(double const* xs,
                                          double val,
                                          IndexRange middle);
  };

  class Interp1D {

    size_t _cols = 0;
    double* _xs = nullptr;
    double* _zs = nullptr;

    // Copy the pointed-to arrays into managed memory.
    // The class interface guarantees that the array lengths
    // match the memory allocation done.
    void _initialize(double const* x, double const* z);

    __device__ __host__ bool _in_range(double val, IndexRange range) const;
    __device__ __host__ IndexRange
    _find_smallest__index_range(double val) const;

  public:
    Interp1D();
    Interp1D(const Interp1D& source);
    Interp1D& operator=(Interp1D const& rhs);
    Interp1D(Interp1D&&) = delete;
    Interp1D& operator=(Interp1D&&) = delete;
    ~Interp1D();

    template <size_t M>
    Interp1D(std::array<double, M> const& xs, std::array<double, M> const& zs);

    Interp1D(double const* xs, double const* zs, size_t cols);

    void swap(Interp1D& other);
    __device__ __host__ double operator()(double x) const;
    __device__ __host__ double min_x() const;
    __device__ __host__ double max_x() const;
    __device__ __host__ double do_clamp(double v, double lo, double hi) const;
    __device__ __host__ double eval(double x) const;
    __device__ __host__ double clamp(double x) const;

    friend std::istream& operator>>(std::istream& is, Interp1D& interp);
  };
}

inline __device__ __host__ bool
quad::IndexRange::is_valid() const
{
  return left < right;
}

inline __device__ __host__ quad::IndexRange
quad::IndexRange::middle() const
{
  size_t const mid = static_cast<size_t>((left + right) * 0.5);
  return {mid, mid + 1};
}

inline __device__ __host__ void
quad::IndexRange::adjust_edges(double const* xs, double val, IndexRange middle)
{

  if (xs[middle.left] > val) {
    right = middle.left; // shrink the right side
  } else {
    left = middle.right; // shrink the left side
  }
}

inline void
quad::Interp1D::_initialize(double const* x, double const* z)
{
  if (_cols > 1000000) {
    std::cerr << "Interp1D::_initilize called when _cols=" << _cols << '\n';
    std::abort();
  }
  //size_t const nbytes = sizeof(double) * _cols;
  _xs = cuda_malloc<double>(_cols);
  //cudaMallocManaged(&_xs, nbytes);
  //memcpy(_xs, x, nbytes);
  cuda_memcpy_to_device<double>(_xs, x, _cols);
  _zs = cuda_malloc<double>(_cols);
  //cudaMallocManaged(&_zs, nbytes);
  //memcpy(_zs, z, nbytes);
  cuda_memcpy_to_device<double>(_zs, z, _cols);
}

inline quad::Interp1D::Interp1D() {}

inline quad::Interp1D::Interp1D(const Interp1D& source) : _cols(source._cols)
{
  _initialize(source._xs, source._zs);
}

inline quad::Interp1D&
quad::Interp1D::operator=(Interp1D const& rhs)
{
  Interp1D tmp(rhs);
  swap(tmp);
  return *this;
}

inline quad::Interp1D::~Interp1D()
{
  cudaFree(_zs);
  cudaFree(_xs);
}

template <size_t M>
quad::Interp1D::Interp1D(std::array<double, M> const& xs,
                         std::array<double, M> const& zs)
  : _cols(M)
{
  _initialize(xs.data(), zs.data());
}

inline quad::Interp1D::Interp1D(double const* xs, double const* zs, size_t cols)
  : _cols(cols)
{
  _initialize(xs, zs);
}

inline void
quad::Interp1D::swap(Interp1D& other)
{
  std::swap(_cols, other._cols);
  std::swap(_zs, other._zs);
  std::swap(_xs, other._xs);
}

inline __device__ __host__ bool
quad::Interp1D::_in_range(const double val, IndexRange const range) const
{
  return (_xs[range.left] <= val) && (_xs[range.right] >= val);
}

inline __device__ __host__ quad::IndexRange
quad::Interp1D::_find_smallest__index_range(const double val) const
{
  // we don't check if val is in the current range. clamp makes sure we dont
  // pass values that exceed min/max, right?
  IndexRange current_range{0, _cols - 1};

  while (current_range.is_valid()) {
    IndexRange smaller_candidate_range = current_range.middle();
    if (_in_range(val, smaller_candidate_range)) {
      return smaller_candidate_range;
    }
    current_range.adjust_edges(_xs, val, smaller_candidate_range);
  }
  return current_range;
}

inline __device__ __host__ double
quad::Interp1D::operator()(double x) const
{
  auto [x0_index, x1_index] = _find_smallest__index_range(x);
  const double y0 = _zs[x0_index];
  const double y1 = _zs[x1_index];
  const double x0 = _xs[x0_index];
  const double x1 = _xs[x1_index];
  const double y = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0);
  return y;
}

inline __device__ __host__ double
quad::Interp1D::min_x() const
{
  return _xs[0];
}

inline __device__ __host__ double
quad::Interp1D::max_x() const
{
  return _xs[_cols - 1];
}

inline __device__ __host__ double
quad::Interp1D::do_clamp(double v, double lo, double hi) const
{
  assert(!(hi < lo));
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

inline __device__ __host__ double
quad::Interp1D::eval(double x) const
{
  return this->operator()(x);
}

__device__ __host__ double
quad::Interp1D::clamp(double x) const
{
  return eval(do_clamp(x, min_x(), max_x()));
}

namespace quad {
  inline std::istream&
  operator>>(std::istream& is, quad::Interp1D& interp)
  {
    std::cout<<"interp1D>>"<<std::endl;
    assert(is.good());
    std::string buffer;
    std::getline(is, buffer);
    std::vector<double> xs = str_to_doubles(buffer);
    std::getline(is, buffer);
    std::vector<double> zs = str_to_doubles(buffer);

    cudaMallocManaged((void**)&(*&interp), sizeof(quad::Interp1D));
    cudaDeviceSynchronize();

    interp._cols = xs.size();

    cudaMallocManaged((void**)&interp._xs, sizeof(double) * xs.size());
    cudaDeviceSynchronize();
    cudaMallocManaged((void**)&interp._zs, sizeof(double) * zs.size());
    cudaDeviceSynchronize();

    memcpy(interp._xs, xs.data(), sizeof(double) * xs.size());
    memcpy(interp._zs, zs.data(), sizeof(double) * zs.size());

    return is;
  }
}
#endif
