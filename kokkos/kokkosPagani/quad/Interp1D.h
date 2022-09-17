#ifndef GPUQUADINTERP1D_H
#define GPUQUADINTERP1D_H

#include "kokkos/kokkosPagani/quad/quad.h"
#include "kokkos/kokkosPagani/quad/util/str_to_doubles.hh"
#include <assert.h>

/*
    interpC is the coordinate list
    interpT is the value list at the respective coordinates
*/

// typedef Kokkos::View<double*, Kokkos::CudaUVMSpace> ViewDouble;

namespace quad {

  class Interp1D {
  public:
    __host__ __device__
    Interp1D()
    {}

    ViewDouble interpT;
    ViewDouble interpC;

    size_t _cols;

    Interp1D(HostVectorDouble xs, HostVectorDouble ys)
    {

      assert(xs.extent(0) == ys.extent(0));
      _cols = xs.extent(0);

      interpT = ViewDouble("interpT", _cols);
      interpC = ViewDouble("interpC", _cols);

      deep_copy(interpC, xs);
      deep_copy(interpT, ys);
    }

    template <size_t M>
    Interp1D(std::array<double, M> const& xs, std::array<double, M> const& zs)
    {
      assert(xs.size() == zs.size());
      AllocateAndSet<M>(xs, zs);
    }

    Interp1D(double* xs, double* zs, size_t cols)
    {
      AllocateAndSet(xs, zs, cols);
    }

    void
    AllocateAndSet(double* xs, double* zs, size_t cols)
    {
      _cols = cols;
      interpT = ViewDouble("interpT", _cols);
      interpC = ViewDouble("interpC", _cols);

      ViewDouble::HostMirror x = Kokkos::create_mirror(interpC);
      ViewDouble::HostMirror y = Kokkos::create_mirror(interpT);

      for (size_t i = 0; i < _cols; ++i) {
        x[i] = xs[i];
        y[i] = zs[i];
      }

      Kokkos::deep_copy(interpC, x);
      Kokkos::deep_copy(interpT, y);
    }

    template <size_t M>
    void
    AllocateAndSet(std::array<double, M> const& xs,
                   std::array<double, M> const& zs)
    {
      _cols = M;
      interpT = ViewDouble("interpT", _cols);
      interpC = ViewDouble("interpC", _cols);

      HostVectorDouble x("x", _cols);
      HostVectorDouble y("x", _cols);

      Kokkos::parallel_for("Copy_from_stdArray",
                           _cols,
                           [=, *this] __host__ __device__(const int64_t index) {
                             interpT(index) = zs[index];
                             interpC(index) = xs[index];
                           });
    }

    __device__ bool
    AreNeighbors(const double val,
                 ViewDouble arr,
                 const size_t leftIndex,
                 const size_t RightIndex) const
    {
      if (arr(leftIndex) <= val && arr(RightIndex) >= val)
        return true;
      return false;
    }

    __device__ void
    FindNeighbourIndices(const double val,
                         ViewDouble arr,
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

        if (arr(currentIndex) > val) {
          rightI = currentIndex;
        } else {
          leftI = currentIndex;
        }
      }
    }

    __device__ double
    operator()(double x) const
    {
      size_t x0_index = 0, x1_index = 0;
      FindNeighbourIndices(x, interpC, _cols, x0_index, x1_index);
      const double y0 = interpT(x0_index);
      const double y1 = interpT(x1_index);
      const double x0 = interpC(x0_index);
      const double x1 = interpC(x1_index);
      const double y = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0);
      return y;
    }

    __device__ double
    min_x() const
    {
      return interpC(0);
    }

    __device__ double
    max_x() const
    {
      return interpC(_cols - 1);
    }

    __device__ double
    do_clamp(double v, double lo, double hi) const
    {
      assert(!(hi < lo));
      return (v < lo) ? lo : (hi < v) ? hi : v;
    }

    __device__ double
    eval(double x) const
    {
      return this->operator()(x);
    };

    __device__ double
    clamp(double x) const
    {
      return eval(do_clamp(x, min_x(), max_x()));
    }
  };
}

#endif