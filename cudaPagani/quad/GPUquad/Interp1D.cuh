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
    size_t _cols = 0;
    double* _interpT = nullptr;
    double* _interpC = nullptr;

    void Alloc(size_t cols);

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

    __device__ __host__ bool AreNeighbors(double val,
                                          double const* arr,
                                          size_t lidx,
                                          size_t ridx) const;

    __device__ __host__ void FindNeighbourIndices(double val,
                                                  double const* arr,
                                                  size_t size,
                                                  size_t& lidx,
                                                  size_t& ridx) const;

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
quad::Interp1D::Alloc(std::size_t cols)
{
  _cols = cols;
  cudaMallocManaged((void**)&_interpC, sizeof(double) * _cols);
  cudaMallocManaged((void**)&_interpT, sizeof(double) * _cols);
}

inline quad::Interp1D::Interp1D() {}

inline quad::Interp1D::Interp1D(const Interp1D& source)
{
  _cols = source._cols;
  Alloc(source._cols);
  memcpy(_interpC, source._interpC, sizeof(double) * _cols);
  memcpy(_interpT, source._interpT, sizeof(double) * _cols);
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
  cudaFree(_interpT);
  cudaFree(_interpC);
}

template <size_t M>
quad::Interp1D::Interp1D(std::array<double, M> const& xs,
                         std::array<double, M> const& zs)
{
  Alloc(M);
  memcpy(_interpC, xs.data(), sizeof(double) * M);
  memcpy(_interpT, zs.data(), sizeof(double) * M);
}

inline quad::Interp1D::Interp1D(double const* xs, double const* zs, size_t cols)
{
  Alloc(cols);
  memcpy(_interpC, xs, sizeof(double) * cols);
  memcpy(_interpT, zs, sizeof(double) * cols);
}

inline void
quad::Interp1D::swap(Interp1D& other)
{
  std::swap(_cols, other._cols);
  std::swap(_interpT, other._interpT);
  std::swap(_interpC, other._interpC);
}

inline __device__ __host__ bool
quad::Interp1D::AreNeighbors(const double val,
                             double const* arr,
                             const size_t leftIndex,
                             const size_t RightIndex) const
{
  return (arr[leftIndex] <= val && arr[RightIndex] >= val);
}

inline __device__ __host__ void
quad::Interp1D::FindNeighbourIndices(const double val,
                                     double const* arr,
                                     const size_t size,
                                     size_t& leftI,
                                     size_t& rightI) const
{

  size_t currentIndex = size / 2;
  leftI = 0;
  rightI = size - 1;

  while (leftI <= rightI) {
    currentIndex = (rightI + leftI) * 0.5;
    if (AreNeighbors(val, arr, currentIndex, currentIndex + 1)) {
      leftI = currentIndex;
      rightI = currentIndex + 1;
      return;
    }

    if (arr[currentIndex] > val) {
      rightI = currentIndex;
    } else {
      leftI = currentIndex;
    }
  }
}

inline __device__ __host__ double
quad::Interp1D::operator()(double x) const
{
  size_t x0_index = 0, x1_index = 0;
  FindNeighbourIndices(x, _interpC, _cols, x0_index, x1_index);
  const double y0 = _interpT[x0_index];
  const double y1 = _interpT[x1_index];
  const double x0 = _interpC[x0_index];
  const double x1 = _interpC[x1_index];
  const double y = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0);
  return y;
}

inline __device__ __host__ double
quad::Interp1D::min_x() const
{
  return _interpC[0];
}

inline __device__ __host__ double
quad::Interp1D::max_x() const
{
  return _interpC[_cols - 1];
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

    cudaMallocManaged((void**)&interp._interpC, sizeof(double) * xs.size());
    cudaDeviceSynchronize();
    cudaMallocManaged((void**)&interp._interpT, sizeof(double) * zs.size());
    cudaDeviceSynchronize();

    memcpy(interp._interpC, xs.data(), sizeof(double) * xs.size());
    memcpy(interp._interpT, zs.data(), sizeof(double) * zs.size());

    return is;
  }
}
#endif
