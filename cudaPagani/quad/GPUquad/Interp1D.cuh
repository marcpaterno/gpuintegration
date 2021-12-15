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
    // change names to xs, ys, zs to fit with y3_cluster_cpp::Interp2D
    size_t _cols = 0;
    double* interpT = nullptr;
    double* interpC = nullptr;

    void
    Alloc(size_t cols)
    {
      _cols = cols;
      cudaMallocManaged((void**)&interpC, sizeof(double) * _cols);
      cudaMallocManaged((void**)&interpT, sizeof(double) * _cols);
    }

  public:

    void swap(Interp1D& other)
    {
      std::swap(_cols, other._cols);
      std::swap(interpT, other.interpT);
      std::swap(interpC, other.interpC);
    }

    __host__ __device__
    Interp1D()
    { }

    Interp1D(const Interp1D& source)
    {
      _cols = source._cols;
      Alloc(source._cols);
      memcpy(interpC, source.interpC, sizeof(double) * _cols);
      memcpy(interpT, source.interpT, sizeof(double) * _cols);
    }

    Interp1D& operator=(Interp1D const& rhs)
    {
      Interp1D tmp(rhs);
      swap(tmp);
      return *this;
    }

    Interp1D(Interp1D&&) = delete;
    Interp1D& operator=(Interp1D&&) = delete;

    ~Interp1D()
    {
       cudaFree(interpT);
       cudaFree(interpC);
    }

     template <size_t M>
    Interp1D(std::array<double, M> const& xs, std::array<double, M> const& zs)
    {
      Alloc(M);
      memcpy(interpC, xs.data(), sizeof(double) * M);
      memcpy(interpT, zs.data(), sizeof(double) * M);
    }

    Interp1D(double const* xs, double const* zs, size_t cols)
    {
      Alloc(cols);
      memcpy(interpC, xs, sizeof(double) * cols);
      memcpy(interpT, zs, sizeof(double) * cols);
    }

   __device__ __host__ bool
    AreNeighbors(const double val,
                 double const* arr,
                 const size_t leftIndex,
                 const size_t RightIndex) const
    {
      return (arr[leftIndex] <= val && arr[RightIndex] >= val);
    }

    friend std::istream&
    operator>>(std::istream& is, Interp1D& interp)
    {
      assert(is.good());
      std::string buffer;
      std::getline(is, buffer);
      std::vector<double> xs = str_to_doubles(buffer);
      std::getline(is, buffer);
      std::vector<double> zs = str_to_doubles(buffer);

      cudaMallocManaged((void**)&(*&interp), sizeof(Interp1D));
      cudaDeviceSynchronize();

      interp._cols = xs.size();

      cudaMallocManaged((void**)&interp.interpC, sizeof(double) * xs.size());
      cudaDeviceSynchronize();
      cudaMallocManaged((void**)&interp.interpT, sizeof(double) * zs.size());
      cudaDeviceSynchronize();

      memcpy(interp.interpC, xs.data(), sizeof(double) * xs.size());
      memcpy(interp.interpT, zs.data(), sizeof(double) * zs.size());

      return is;
    }
    __device__ __host__ void
    FindNeighbourIndices(const double val,
                         double* arr,
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

    __device__ __host__ double
    operator()(double x) const
    {
      size_t x0_index = 0, x1_index = 0;
      FindNeighbourIndices(x, interpC, _cols, x0_index, x1_index);
      const double y0 = interpT[x0_index];
      const double y1 = interpT[x1_index];
      const double x0 = interpC[x0_index];
      const double x1 = interpC[x1_index];
      const double y = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0);
      return y;
    }

    __device__ __host__ double
    min_x() const
    {
      return interpC[0];
    }

    __device__ __host__ double
    max_x() const
    {
      return interpC[_cols - 1];
    }

    __device__ __host__ double
    do_clamp(double v, double lo, double hi) const
    {
      assert(!(hi < lo));
      return (v < lo) ? lo : (hi < v) ? hi : v;
    }

    __device__ __host__ double
    eval(double x) const
    {
      return this->operator()(x);
    };

    __device__ __host__ double
    clamp(double x) const
    {
      return eval(do_clamp(x, min_x(), max_x()));
    }
  };
}
#endif
