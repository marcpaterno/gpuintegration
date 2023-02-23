#ifndef CUDA_INTERP_3D_CUH
#define CUDA_INTERP_3D_CUH

#include "cuda/pagani/quad/quad.h"
#include "common/cuda/cudaArray.cuh"
#include "common/cuda/cudaMemoryUtil.h"
#include "common/cuda/cudaTimerUtil.h"
#include "common/cuda/str_to_doubles.hh"
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace quad {

  class Interp3D {
    // change names to xs, ys, zs to fit with y3_cluster_cpp::Interp3D
    size_t size_x = 0;
    size_t size_y = 0;
    size_t size_z = 0;

    double* interpT = nullptr;
    double* _zs = nullptr;
    double* _ys = nullptr; // y -> interpR
    double* _xs = nullptr; // x -> interpC

    void
    Alloc(size_t x, size_t y, size_t z)
    {
      if (x > 100000 || y > 100000 || z > 100000) {
        std::cerr << "InterpD::Alloc called with x=" << x << " and y=" << y
                  << " and z=" << z << '\n';
        std::abort();
      }
      if (x * y * z > 1000000) {
        std::cerr << "Interp3D::Alloc called with x=" << x << " and y=" << y
                  << " and z=" << z << '\n';
        std::abort();
      }

      size_x = x;
      size_y = y;
      size_z = z;

      _xs = cuda_malloc<double>(x);
      _ys = cuda_malloc<double>(y);
      _zs = cuda_malloc<double>(z);
      interpT = cuda_malloc<double>(x * y * z);
      CudaCheckError();
    }

  public:
    size_t
    get_device_mem_footprint()
    {
      return 8 * (size_x * size_y * size_z + size_x + size_y + size_z);
    }

    size_t
    get_device_mem_footprint() const
    {
      return 8 * (size_x * size_y * size_z + size_x + size_y + size_z);
    }

    void
    swap(Interp3D& other)
    {
      std::swap(size_x, other.size_x);
      std::swap(size_y, other.size_y);
      std::swap(size_z, other.size_z);

      std::swap(interpT, other.interpT);
      std::swap(_xs, other._xs);
      std::swap(_ys, other._ys);
      std::swap(_zs, other._zs);
    }

    __host__ __device__
    Interp3D()
    {}

    Interp3D(const Interp3D& source)
    {
      size_x = source.size_x;
      size_y = source.size_y;
      size_z = source.size_z;
      Alloc(size_x, size_y, size_z);

      cuda_memcpy_device_to_device<double>(
        interpT, source.interpT, size_x * size_y * size_z);
      cuda_memcpy_device_to_device<double>(_xs, source._xs, size_x);
      cuda_memcpy_device_to_device<double>(_ys, source._ys, size_y);
      cuda_memcpy_device_to_device<double>(_zs, source._zs, size_z);
      CudaCheckError();
    }

    Interp3D&
    operator=(Interp3D const& rhs)
    {
      Interp3D tmp(rhs);
      CudaCheckError();
      swap(tmp);
      return *this;
    }

    Interp3D(Interp3D&&) = delete;
    Interp3D& operator=(Interp3D&&) = delete;

    ~Interp3D()
    {
      cudaFree(_xs);
      cudaFree(_ys);
      cudaFree(_zs);
      cudaFree(interpT);
    }

    template <size_t M, size_t N, size_t S>
    Interp3D(std::array<double, M> const& xs,
             std::array<double, N> const& ys,
             std::array<double, S> const& zs,
             std::array<double, N * M * S> const& vals)
    {
      CudaCheckError();
      Alloc(M, N, S);
      cuda_memcpy_to_device<double>(_xs, xs.data(), M);
      cuda_memcpy_to_device<double>(_ys, ys.data(), N);
      cuda_memcpy_to_device<double>(_zs, zs.data(), S);
      cuda_memcpy_to_device<double>(interpT, vals.data(), N * M * S);
      CudaCheckError();
    }

    Interp3D(double const* xs,
             double const* ys,
             double const* zs,
             double const* vs,
             size_t x_size,
             size_t y_size,
             size_t z_size)
    {
      CudaCheckError();
      Alloc(x_size, y_size, z_size);
      cuda_memcpy_to_device<double>(_xs, xs, x_size);
      cuda_memcpy_to_device<double>(_ys, ys, y_size);
      cuda_memcpy_to_device<double>(_zs, zs, z_size);
      cuda_memcpy_to_device<double>(interpT, vs, x_size * y_size * z_size);
      CudaCheckError();
    }

    Interp3D(std::vector<double> const& xs,
             std::vector<double> const& ys,
             std::vector<double> const& zs,
             std::vector<double> const& vs)
      : Interp3D(xs.data(),
                 ys.data(),
                 zs.data(),
                 vs.data(),
                 xs.size(),
                 ys.size(),
                 zs.size())
    {
      CudaCheckError();
    }

    __device__ bool
    AreNeighbors(const double val,
                 double* arr,
                 const size_t leftIndex,
                 const size_t RightIndex) const
    {
      return (arr[leftIndex] <= val && arr[RightIndex] >= val);
    }

    __device__ void
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

    __device__ size_t
    index(size_t x, size_t y, size_t z) const
    {
      return x + (size_x * y) + (size_x * size_y * z);
    }

    __device__ double
    operator()(double x, double y, double z) const
    {
      // y1, y2, x1, x2, are the indices of where to find the four neighbouring
      // points in the z-table
      size_t y0 = 0, y1 = 0;
      size_t x0 = 0, x1 = 0;
      size_t z0 = 0, z1 = 0;

      // separately find two nearest values to each of x, y, z
      FindNeighbourIndices(y, _ys, size_y, y0, y1);
      FindNeighbourIndices(x, _xs, size_x, x0, x1);
      FindNeighbourIndices(z, _zs, size_z, z0, z1);

      const double x_d = (x - _xs[x0]) / (_xs[x1] - _xs[x0]);
      const double y_d = (y - _ys[y0]) / (_ys[y1] - _ys[y0]);
      const double z_d = (z - _zs[z0]) / (_zs[z1] - _zs[z0]);

      // interpolate along x
      const double c00 = interpT[index(x0, y0, z0)] * (1 - x_d) +
                         interpT[index(x1, y0, z0)] * x_d;
      const double c01 = interpT[index(x0, y0, z1)] * (1 - x_d) +
                         interpT[index(x1, y0, z1)] * x_d;
      const double c10 = interpT[index(x0, y1, z0)] * (1 - x_d) +
                         interpT[index(x1, y1, z0)] * x_d;
      const double c11 = interpT[index(x0, y1, z1)] * (1 - x_d) +
                         interpT[index(x1, y1, z1)] * x_d;

      // interpolate values along y
      const double c0 = c00 * (1 - y_d) + c10 * y_d;
      const double c1 = c01 * (1 - y_d) + c11 * y_d;

      // interpolate along z
      const double f_xyz = c0 * (1 - z_d) + c1 * z_d;
      return f_xyz;
    }

    __device__ double
    min_x() const
    {
      return _xs[0];
    }

    __device__ double
    max_x() const
    {
      return _xs[size_x - 1];
    }

    __device__ double
    min_y() const
    {
      return _ys[0];
    }

    __device__ double
    max_y() const
    {
      return _ys[size_y - 1];
    }

    __device__ double
    min_z() const
    {
      return _zs[0];
    }

    __device__ double
    max_z() const
    {
      return _zs[size_z - 1];
    }

    __device__ double
    do_clamp(double v, double lo, double hi) const
    {
      assert(!(hi < lo));
      return (v < lo) ? lo : (hi < v) ? hi : v;
    }

    __device__ double
    eval(double x, double y, double z) const
    {
      return this->operator()(x, y, z);
    };

    __device__ double
    clamp(double x, double y, double z) const
    {
      return eval(do_clamp(x, min_x(), max_x()),
                  do_clamp(y, min_y(), max_y()),
                  do_clamp(z, min_z(), max_z()));
    }
  };
}

#endif
