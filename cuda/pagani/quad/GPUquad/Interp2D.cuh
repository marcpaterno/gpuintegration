#ifndef GPUQUADINTERP2D_H
#define GPUQUADINTERP2D_H

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
  
  template<typename T>
  class Interp2D {
    // change names to xs, ys, zs to fit with y3_cluster_cpp::Interp2D
    size_t _rows = 0;
    size_t _cols = 0;

    T* interpT = nullptr;
    T* interpR = nullptr;
    T* interpC = nullptr;
    
    void
    Alloc(size_t cols, size_t rows)
    {
      CudaCheckError();
      if ((cols > 100000) || (rows > 100000)) {
        std::cerr << "InterpD::Alloc called with cols=" << cols
                  << " and rows=" << rows << '\n';
        std::abort();
      }
      if (cols * rows > 1000000) {
        std::cerr << "Interp2D::Alloc called with cols=" << cols
                  << " and rows=" << rows << '\n';
        std::abort();
      }
      _rows = rows;
      _cols = cols;
      interpR = cuda_malloc<T>(_rows);
      CudaCheckError();
      interpC = cuda_malloc<T>(_cols);
      CudaCheckError();
      interpT = cuda_malloc<T>(_rows * _cols);
      CudaCheckError();
    }

  public:
    size_t
    get_device_mem_footprint()
    {
      return 8 * (_cols * _rows + _cols + _rows);
    }

    size_t
    get_device_mem_footprint() const
    {
      return 8 * (_cols * _rows + _cols + _rows);
    }

    void
    swap(Interp2D<T>& other)
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

    Interp2D(const Interp2D<T>& source)
    {
      _cols = source._cols;
      _rows = source._rows;
      Alloc(_cols, _rows);
      
      cuda_memcpy_device_to_device<T>(interpT, source.interpT, _cols * _rows);
      cuda_memcpy_device_to_device<T>(interpC, source.interpC, _cols);
      cuda_memcpy_device_to_device<T>(interpR, source.interpR, _rows);
      CudaCheckError();
    }

    Interp2D<T>&
    operator=(Interp2D<T> const& rhs)
    {
      Interp2D tmp(rhs);
      CudaCheckError();
      swap(tmp);
      return *this;
    }

    
    Interp2D(Interp2D<T>&&) = delete;
    Interp2D& operator=(Interp2D<T>&&) = delete;

    ~Interp2D()
    {
      cudaFree(interpT);
      cudaFree(interpR);
      cudaFree(interpC);
    }

    template <size_t M, size_t N>
    Interp2D(std::array<T, M> const& xs,
             std::array<T, N> const& ys,
             std::array<T, (N) * (M)> const& zs)
    {
      CudaCheckError();
      Alloc(M, N);
      cuda_memcpy_to_device<T>(interpR, ys.data(), N);
      cuda_memcpy_to_device<T>(interpC, xs.data(), M);
      cuda_memcpy_to_device<T>(interpT, zs.data(), N * M);
      CudaCheckError();
    }

    Interp2D(T const* xs,
             T const* ys,
             T const* zs,
             size_t cols,
             size_t rows)
    {
      CudaCheckError();
      Alloc(cols, rows);
      cuda_memcpy_to_device<T>(interpR, ys, rows);
      cuda_memcpy_to_device<T>(interpC, xs, cols);
      cuda_memcpy_to_device<T>(interpT, zs, rows * cols);
      CudaCheckError();
    }

    Interp2D(std::vector<T> const& xs,
             std::vector<T> const& ys,
             std::vector<T> const& zs)
      : Interp2D(xs.data(), ys.data(), zs.data(), xs.size(), ys.size())
    {
      CudaCheckError();
    }

    template <size_t M, size_t N>
    Interp2D(std::array<T, M> xs,
             std::array<T, N> ys,
             std::array<std::array<T, N>, M> zs)
    {

      CudaCheckError();
      Alloc(M, N);
      cuda_memcpy_to_device<T>(interpR, ys.data(), N);
      cuda_memcpy_to_device<T>(interpC, xs.data(), M);


      std::vector<T> buffer(N*M);
      for (std::size_t i = 0; i < M; ++i) {
        std::array<T, N> const& row = zs[i];
        for (std::size_t j = 0; j < N; ++j) {
          buffer[i + j * M] = row[j];
        }
      }

      cuda_memcpy_to_device<T>(interpT, buffer.data(), N*M);
      CudaCheckError();
    }

    __host__ __device__ bool
    AreNeighbors(const T val,
                 T* arr,
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
      std::vector<T> xs = str_to_doubles(buffer);
      std::getline(is, buffer);
      std::vector<T> ys = str_to_doubles(buffer);
      std::getline(is, buffer);
      std::vector<T> zs = str_to_doubles(buffer);

      // why is this done?
      cudaMallocManaged((void**)&(*&interp), sizeof(Interp2D));

      interp._cols = xs.size();
      interp._rows = ys.size();

      cudaMallocManaged((void**)&interp.interpR, sizeof(T) * ys.size());
      cudaMallocManaged((void**)&interp.interpC, sizeof(T) * xs.size());
      cudaMallocManaged((void**)&interp.interpT, sizeof(T) * zs.size());

      memcpy(interp.interpR, ys.data(), sizeof(T) * ys.size());
      memcpy(interp.interpC, xs.data(), sizeof(T) * xs.size());
      memcpy(interp.interpT, zs.data(), sizeof(T) * zs.size());

      return is;
    }

    __device__ __host__ void
    FindNeighbourIndices(const T val,
                         T* arr,
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

    __device__ __host__ T
    operator()(T x, T y) const
    {
      // y1, y2, x1, x2, are the indices of where to find the four neighbouring
      // points in the z-table
      size_t y1 = 0, y2 = 0;
      size_t x1 = 0, x2 = 0;
      FindNeighbourIndices(y, interpR, _rows, y1, y2);
      FindNeighbourIndices(x, interpC, _cols, x1, x2);
      // this is how  zij is accessed by gsl2.6 Interp2D i.e. zij =
      // z[j*xsize+i], where i=0,...,xsize-1, j=0, ..., ysize-1
      const T q11 = interpT[y1 * _cols + x1];
      const T q12 = interpT[y2 * _cols + x1];
      const T q21 = interpT[y1 * _cols + x2];
      const T q22 = interpT[y2 * _cols + x2];

      const T x1_val = interpC[x1];
      const T x2_val = interpC[x2];
      const T y1_val = interpR[y1];
      const T y2_val = interpR[y2];

      const T f_x_y1 = q11 * (x2_val - x) / (x2_val - x1_val) +
                            q21 * (x - x1_val) / (x2_val - x1_val);
      const T f_x_y2 = q12 * (x2_val - x) / (x2_val - x1_val) +
                            q22 * (x - x1_val) / (x2_val - x1_val);

      T f_x_y = 0.;
      f_x_y = f_x_y1 * (y2_val - y) / (y2_val - y1_val) +
              f_x_y2 * (y - y1_val) / (y2_val - y1_val);
      return f_x_y;
    }

    __device__ __host__ T
    min_x() const
    {
      return interpC[0];
    }

    __device__ __host__ T
    max_x() const
    {
      return interpC[_cols - 1];
    }

    __device__ __host__ T
    min_y() const
    {
      return interpR[0];
    }

    __device__ __host__ T
    max_y() const
    {
      return interpR[_rows - 1];
    }

    __device__ __host__ T
    do_clamp(T v, T lo, T hi) const
    {
      assert(!(hi < lo));
      return (v < lo) ? lo : (hi < v) ? hi : v;
    }

    __device__ __host__ T
    eval(T x, T y) const
    {
      return this->operator()(x, y);
    };

    __device__ __host__ T
    clamp(T x, T y) const
    {
      return eval(do_clamp(x, min_x(), max_x()), do_clamp(y, min_y(), max_y()));
    }
  };
}

#endif
