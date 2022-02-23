#ifndef GPUQUADINTERP2D_H
#define GPUQUADINTERP2D_H

#include "cuda/cudaPagani/quad/quad.h"
#include "cuda/cudaPagani/quad/util/cudaArray.cuh"
#include "cuda/cudaPagani/quad/util/cudaMemoryUtil.h"
#include "cuda/cudaPagani/quad/util/cudaTimerUtil.h"
#include "cuda/cudaPagani/quad/util/str_to_doubles.hh"
#include <assert.h>
#include <utility>

namespace quad {

  class Interp2D : public Managed {
    // change names to xs, ys, zs to fit with y3_cluster_cpp::Interp2D
    size_t _rows = 0;
    size_t _cols = 0;
    double* interpT = nullptr;
    double* interpR = nullptr;
    double* interpC = nullptr;

    void
    Alloc(size_t cols, size_t rows)
    {
      _rows = rows;
      _cols = cols;
      cudaMallocManaged((void**)&interpR, sizeof(double) * _rows);
      cudaMallocManaged((void**)&interpC, sizeof(double) * _cols);
      cudaMallocManaged((void**)&interpT, sizeof(double) * _rows * _cols);
      CudaCheckError();
    }

  public:
    void
    swap(Interp2D& other)
    {
      std::swap(_rows, other._rows);
      std::swap(_cols, other._cols);
      std::swap(interpT, other.interpT);
      std::swap(interpR, other.interpR);
      std::swap(interpC, other.interpC);
    }

    __host__ __device__
    Interp2D()
    {}

    Interp2D(const Interp2D& source)
    {
      _cols = source._cols;
      _rows = source._rows;
      Alloc(_cols, _rows);
      memcpy(interpT, source.interpT, sizeof(double) * _cols * _rows);
      memcpy(interpC, source.interpC, sizeof(double) * _cols);
      memcpy(interpR, source.interpR, sizeof(double) * _rows);
      CudaCheckError();
    }

    Interp2D&
    operator=(Interp2D const& rhs)
    {
      Interp2D tmp(rhs);
      CudaCheckError();
      swap(tmp);
      return *this;
    }

    Interp2D(Interp2D&&) = delete;
    Interp2D& operator=(Interp2D&&) = delete;

    ~Interp2D()
    {
      CudaCheckError();
      cudaFree(interpT);
      cudaFree(interpR);
      cudaFree(interpC);
      CudaCheckError();
    }

    template <size_t M, size_t N>
    Interp2D(std::array<double, M> const& xs,
             std::array<double, N> const& ys,
             std::array<double, (N) * (M)> const& zs)
    {
      CudaCheckError();
      Alloc(M, N);
      memcpy(interpR, ys.data(), sizeof(double) * N);
      memcpy(interpC, xs.data(), sizeof(double) * M);
      memcpy(interpT, zs.data(), sizeof(double) * N * M);
      CudaCheckError();
    }

    Interp2D(double const* xs,
             double const* ys,
             double const* zs,
             size_t cols,
             size_t rows)
    {
      CudaCheckError();
      Alloc(cols, rows);
      memcpy(interpR, ys, sizeof(double) * rows);
      memcpy(interpC, xs, sizeof(double) * cols);
      memcpy(interpT, zs, sizeof(double) * rows * cols);
      CudaCheckError();
    }

    Interp2D(std::vector<double> const& xs,
             std::vector<double> const& ys,
             std::vector<double> const& zs)
      : Interp2D(xs.data(), ys.data(), zs.data(), xs.size(), ys.size())
    {
      CudaCheckError();
    }

    template <size_t M, size_t N>
    Interp2D(std::array<double, M> xs,
             std::array<double, N> ys,
             std::array<std::array<double, N>, M> zs)
    {
      CudaCheckError();
      Alloc(M, N);
      memcpy(interpR, ys.data(), sizeof(double) * N);
      memcpy(interpC, xs.data(), sizeof(double) * M);

      for (std::size_t i = 0; i < M; ++i) {
        std::array<double, N> const& row = zs[i];
        for (std::size_t j = 0; j < N; ++j) {
          interpT[i + j * M] = row[j];
        }
      }
      CudaCheckError();
    }

    __host__ __device__ bool
    AreNeighbors(const double val,
                 double* arr,
                 const size_t leftIndex,
                 const size_t RightIndex) const
    {
      return (arr[leftIndex] <= val && arr[RightIndex] >= val);
    }

    friend std::istream&
    operator>>(std::istream& is, Interp2D& interp)
    {
      assert(is.good());
      std::string buffer;
      std::getline(is, buffer);
      std::vector<double> xs = str_to_doubles(buffer);
      std::getline(is, buffer);
      std::vector<double> ys = str_to_doubles(buffer);
      std::getline(is, buffer);
      std::vector<double> zs = str_to_doubles(buffer);

      cudaMallocManaged((void**)&(*&interp), sizeof(Interp2D));

      interp._cols = xs.size();
      interp._rows = ys.size();

      cudaMallocManaged((void**)&interp.interpR, sizeof(double) * ys.size());
      cudaMallocManaged((void**)&interp.interpC, sizeof(double) * xs.size());
      cudaMallocManaged((void**)&interp.interpT, sizeof(double) * zs.size());

      memcpy(interp.interpR, ys.data(), sizeof(double) * ys.size());
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
    operator()(double x, double y) const
    {
      // y1, y2, x1, x2, are the indices of where to find the four neighbouring
      // points in the z-table
      size_t y1 = 0, y2 = 0;
      size_t x1 = 0, x2 = 0;
      FindNeighbourIndices(y, interpR, _rows, y1, y2);
      FindNeighbourIndices(x, interpC, _cols, x1, x2);
      // this is how  zij is accessed by gsl2.6 Interp2D i.e. zij =
      // z[j*xsize+i], where i=0,...,xsize-1, j=0, ..., ysize-1
      const double q11 = interpT[y1 * _cols + x1];
      const double q12 = interpT[y2 * _cols + x1];
      const double q21 = interpT[y1 * _cols + x2];
      const double q22 = interpT[y2 * _cols + x2];

      const double x1_val = interpC[x1];
      const double x2_val = interpC[x2];
      const double y1_val = interpR[y1];
      const double y2_val = interpR[y2];

      const double f_x_y1 = q11 * (x2_val - x) / (x2_val - x1_val) +
                            q21 * (x - x1_val) / (x2_val - x1_val);
      const double f_x_y2 = q12 * (x2_val - x) / (x2_val - x1_val) +
                            q22 * (x - x1_val) / (x2_val - x1_val);

      double f_x_y = 0.;
      f_x_y = f_x_y1 * (y2_val - y) / (y2_val - y1_val) +
              f_x_y2 * (y - y1_val) / (y2_val - y1_val);
      return f_x_y;
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
    min_y() const
    {
      return interpR[0];
    }

    __device__ __host__ double
    max_y() const
    {
      return interpR[_rows - 1];
    }

    __device__ __host__ double
    do_clamp(double v, double lo, double hi) const
    {
      assert(!(hi < lo));
      return (v < lo) ? lo : (hi < v) ? hi : v;
    }

    __device__ __host__ double
    eval(double x, double y) const
    {
      return this->operator()(x, y);
    };

    __device__ __host__ double
    clamp(double x, double y) const
    {
      return eval(do_clamp(x, min_x(), max_x()), do_clamp(y, min_y(), max_y()));
    }
  };
}

#endif
