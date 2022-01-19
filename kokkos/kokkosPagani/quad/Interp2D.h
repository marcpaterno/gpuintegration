#ifndef GPUQUADINTERP2D_H
#define GPUQUADINTERP2D_H

#include "kokkos/kokkosPagani/quad/quad.h"
#include "kokkos/kokkosPagani/quad/util/str_to_doubles.hh"
#include <assert.h> 

/*
    interpC is x for cols
    interpR is y for rows
    interpT is z for the values at the row/col coordinates
*/

//typedef Kokkos::View<double*, Kokkos::CudaUVMSpace> ViewDouble;

namespace quad {

class Interp2D {
  public:
    __host__ __device__
    Interp2D()
    {}
    
    ViewDouble interpT;
    ViewDouble interpR;
    ViewDouble interpC;
    
    size_t _cols, _rows;
    
    Interp2D(ViewDouble xs, ViewDouble ys, ViewDouble zs){
        assert(xs.extent(0) * ys.extent(0) == zs.extent(0));
        _cols = xs.extent(0);
        _rows = ys.extent(0);
        
        interpT = ViewDouble("interpT", _cols*_rows);
        interpC = ViewDouble("interpC", _cols);
        interpR = ViewDouble("interpR", _rows);
        
        interpC = xs;
        interpR = ys;
        interpT = zs;
    }
    
    template <size_t M, size_t N>
    Interp2D(std::array<double, M> const& xs, 
        std::array<double, N> const& ys, 
        std::array<double, M*N> const& zs)
    {
      assert(xs.size() * ys.size() == zs.size());
      AllocateAndSet<M, N>(xs, ys, zs);
    }

    Interp2D(double* xs, double* ys, double* zs, size_t cols, size_t rows)
    {
        AllocateAndSet(xs, ys, zs, cols, rows);
    }
    
    void
    AllocateAndSet(double* xs, double* ys, double* zs, size_t cols, size_t rows)
    {
      _cols = cols;
      _rows = rows;
      
      interpT = ViewDouble("interpT", _cols*_rows);
      interpC = ViewDouble("interpC", _cols);
      interpR = ViewDouble("interpR", _rows);
      
      ViewDouble::HostMirror x = Kokkos::create_mirror(interpC);
      ViewDouble::HostMirror y = Kokkos::create_mirror(interpR);
      ViewDouble::HostMirror z = Kokkos::create_mirror(interpT);
      
      for(size_t i = 0; i < _cols*_rows; ++i){
        if(i < _cols)  
            x[i] = xs[i];
        if(i < _rows)
            y[i] = ys[i];
        z[i] = zs[i];
      }
      
      Kokkos::deep_copy(interpC, x);
      Kokkos::deep_copy(interpR, y);
      Kokkos::deep_copy(interpT, z);
    }
    
    
    template <size_t M, size_t N>
    void
    AllocateAndSet(std::array<double, M> const& xs, 
        std::array<double, N> const& ys, 
        std::array<double, M*N> const& zs)
    {
      _cols = M;
      _rows = N;
      
      interpT = ViewDouble("interpT", _cols*_rows);
      interpC = ViewDouble("interpC", _cols);
      interpR = ViewDouble("interpC", _rows);
      
      Kokkos::parallel_for(
        "Copy_from_stdArray", _cols*_rows, [=,*this] __host__ __device__ (const size_t index) {
          if(index < _rows){
            interpR(index) = ys[index];
          }
          if(index < _cols){
            interpC(index) = xs[index]; 
          }
          interpT(index) = zs[index];
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
      const double q11 = interpT(y1 * _cols + x1);
      const double q12 = interpT(y2 * _cols + x1);
      const double q21 = interpT(y1 * _cols + x2);
      const double q22 = interpT(y2 * _cols + x2);

      const double x1_val = interpC(x1);
      const double x2_val = interpC(x2);
      const double y1_val = interpR(y1);
      const double y2_val = interpR(y2);

      const double f_x_y1 = q11 * (x2_val - x) / (x2_val - x1_val) +
                            q21 * (x - x1_val) / (x2_val - x1_val);
      const double f_x_y2 = q12 * (x2_val - x) / (x2_val - x1_val) +
                            q22 * (x - x1_val) / (x2_val - x1_val);

      double f_x_y = 0.;
      f_x_y = f_x_y1 * (y2_val - y) / (y2_val - y1_val) +
              f_x_y2 * (y - y1_val) / (y2_val - y1_val);
      return f_x_y;
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
    min_y() const
    {
      return interpR(0);
    }

    __device__ double
    max_y() const
    {
      return interpR(_rows - 1);
    }

    __device__ __host__ double
    do_clamp(double v, double lo, double hi) const
    {
      assert(!(hi < lo));
      return (v < lo) ? lo : (hi < v) ? hi : v;
    }

    __device__ double
    eval(double x, double y) const
    {
      return this->operator()(x, y);
    };

    __device__ double
    clamp(double x, double y) const
    {
      return eval(do_clamp(x, min_x(), max_x()), do_clamp(y, min_y(), max_y()));
    }
  };
}

#endif