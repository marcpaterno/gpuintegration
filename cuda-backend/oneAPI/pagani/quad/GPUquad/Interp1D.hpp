#ifndef ONEAPI_GPUQUADINTERP1D_H
#define ONEAPI_GPUQUADINTERP1D_H

#include "oneAPI/pagani/quad/quad.h"
#include "oneAPI/pagani/quad/util/cudaArray.dp.hpp"
#include "oneAPI/pagani/quad/util/cudaMemoryUtil.h"
#include "oneAPI/pagani/quad/util/str_to_doubles.hh"
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace quad {

  // Helper struct to describe an index range.
  struct IndexRange {
    size_t left = 0;
    size_t right = 0;

    bool is_valid() const;
    IndexRange middle() const;
    void adjust_edges(double const* xs,
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

    bool _in_range(double val, IndexRange range) const;
    IndexRange
    _find_smallest__index_range(double val) const;

  public:
    size_t get_device_mem_footprint(){
      return 8*_cols*_cols;
    }
    
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
    double operator()(double x) const;
    double min_x() const;
    double max_x() const;
    double do_clamp(double v, double lo, double hi) const;
    double eval(double x) const;
    double clamp(double x) const;

    friend std::istream& operator>>(std::istream& is, Interp1D& interp);
  };
}

inline bool
quad::IndexRange::is_valid() const
{
  return left < right;
}

inline quad::IndexRange
quad::IndexRange::middle() const
{
  size_t const mid = static_cast<size_t>((left + right) * 0.5);
  return {mid, mid + 1};
}

inline void
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
  
  _xs = cuda_malloc<double>(_cols);
  cuda_memcpy_to_device<double>(_xs, x, _cols);
  _zs = cuda_malloc<double>(_cols);
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
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
	sycl::free(_xs, q_ct1);
	sycl::free(_zs, q_ct1);
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

inline bool
quad::Interp1D::_in_range(const double val, IndexRange const range) const
{
  return (_xs[range.left] <= val) && (_xs[range.right] >= val);
}

inline quad::IndexRange
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

inline double
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

inline double
quad::Interp1D::min_x() const
{
  return _xs[0];
}

inline double
quad::Interp1D::max_x() const
{
  return _xs[_cols - 1];
}

inline double
quad::Interp1D::do_clamp(double v, double lo, double hi) const
{
  assert(!(hi < lo));
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

inline double
quad::Interp1D::eval(double x) const
{
  return this->operator()(x);
}

double
quad::Interp1D::clamp(double x) const
{
  return eval(do_clamp(x, min_x(), max_x()));
}

/*namespace quad {
  inline std::istream&
  operator>>(std::istream& is, quad::Interp1D* interp)
  {
	friend class quad::Interp1D;
    assert(is.good());
    std::string buffer;
    std::getline(is, buffer);
    std::vector<double> xs = str_to_doubles(buffer);
    std::getline(is, buffer);
    std::vector<double> zs = str_to_doubles(buffer);

	interp = cuda_malloc_managed<quad::Interp1D>();
	
    interp->_cols = xs.size();
	
	interp->_xs = cuda_malloc_managed<double>(xs.size());
	interp->_zs = cuda_malloc_managed<double>(zs.size());
	
	cuda_memcpy_to_device<double>(interp->_xs, xs.data(), xs.size());
    cuda_memcpy_to_device<double>(interp->_zs, zs.data(), zs.size());
    return is;
  }
}*/
#endif
