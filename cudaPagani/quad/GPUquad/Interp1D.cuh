#ifndef GPUQUADINTERP1D_H
#define GPUQUADINTERP1D_H

#include "cudaPagani/quad/quad.h"
#include "cudaPagani/quad/util/cudaArray.cuh"
#include "cudaPagani/quad/util/cudaMemoryUtil.h"
#include "cudaPagani/quad/util/cudaTimerUtil.h"
#include "cudaPagani/quad/util/str_to_doubles.hh"
#include <assert.h>
#include <utility>

namespace quad {

  class Interp1D : public Managed {

    // Helper struct
    struct index_t {
      size_t left = 0;
      size_t right = 0;
    };

    size_t _cols = 0;
    double* _xs = nullptr;
    double* _zs = nullptr;

    // Copy the pointed-to arrays into managed memory.
    // The class interface guarantees that the array lengths
    // match the memory allocation done.
    void _initialize(double const* x, double const* z);

    __device__ __host__ bool _in_range(double val, index_t range) const;
	
	__device__ __host__ bool _is_valid_range(index_t range) const;

    __device__ __host__ index_t _find_smallest__index_range(double val) const;
	
	 
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

inline void
quad::Interp1D::_initialize(double const* x, double const* z)
{
  size_t const nbytes = sizeof(double) * _cols;
  cudaMallocManaged(&_xs, nbytes);
  memcpy(_xs, x, nbytes);
  cudaMallocManaged(&_zs, nbytes);
  memcpy(_zs, z, nbytes);
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
quad::Interp1D::_in_range(const double val, index_t const range) const
{
  return (_xs[range.left] <= val) && (_xs[range.right] >= val);
}

inline __device__ __host__ bool
quad::Interp1D::_is_valid_range(index_t const range) const
{
  if(range.left < range.right)
		return true;
	return false;
}

inline __device__ __host__ quad::Interp1D::index_t
quad::Interp1D::_find_smallest__index_range(const double val) const
{
  //we don't check if val is in the current range. clamp makes sure we dont pass values that exceed min/max, right?
  index_t current_range{0, _cols-1};
  size_t& leftIndex = current_range.left;
  size_t& rightIndex = current_range.right;
  
  while(_is_valid_range(current_range))
  {
	  size_t middle_index = static_cast<size_t>((rightIndex + leftIndex) * 0.5);
	  index_t smaller_candidate_range{middle_index, middle_index + 1};
	  
	  if(_in_range(val, smaller_candidate_range)){
		return smaller_candidate_range; 
	  }
	  
	  if (_xs[middle_index] > val) {
		rightIndex = middle_index;
	  } else {
		leftIndex = middle_index;
      }
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
