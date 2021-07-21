#ifndef GPUQUADINTERP1D_H
#define GPUQUADINTERP1D_H

#include "quad.h"
#include "util/str_to_doubles.hh"
#include <assert.h> 

/*
    interpC is the coordinate list
    interpT is the value list at the respective coordinates
*/

namespace quad {

class Interp1D {
  public:
  
    Interp1D()
    {}
    
    ViewVectorDouble interpT;
    ViewVectorDouble interpC;

    size_t _cols;
    
    Interp1D(HostVectorDouble xs, HostVectorDouble ys){
        
        assert(xs.extent(0) == ys.extent(0));
        _cols = xs.extent(0);
        
        interpT = ViewVectorDouble("interpT", _cols);
        interpC = ViewVectorDouble("interpC", _cols);
        
        deep_copy(interpC, xs);
        deep_copy(interpT, ys);
        
        //interpT = xs;
        //interpC = ys;
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
      interpT = ViewVectorDouble("interpT", _cols);
      interpC = ViewVectorDouble("interpC", _cols);
      
      //HostVectorDouble x("x", _cols);
      //HostVectorDouble y("x", _cols);
      Kokkos::View<double*>::HostMirror x = Kokkos::create_mirror(interpC);
      Kokkos::View<double*>::HostMirror y = Kokkos::create_mirror(interpT);
      
      for(size_t i = 0; i < _cols; ++i){
        x[i] = xs[i];  
        y[i] = zs[i];
      }
      
      Kokkos::deep_copy(interpC, x);
      Kokkos::deep_copy(interpT, y);
      
      /*Kokkos::parallel_for(
        "Copy_from_stdArray", _cols, [=,*this] __host__ __device__ (const int64_t index) {
          interpT(index) = zs[index];
          interpC(index) = xs[index];
        });*/
    }
    
    
    template <size_t M>
    void
    AllocateAndSet(std::array<double, M> const& xs, std::array<double, M> const& zs)
    {
      _cols = M;
      interpT = ViewVectorDouble("interpT", _cols);
      interpC = ViewVectorDouble("interpC", _cols);
      
      HostVectorDouble x("x", _cols);
      HostVectorDouble y("x", _cols);

      
      Kokkos::parallel_for(
        "Copy_from_stdArray", _cols, [=,*this] __host__ __device__ (const int64_t index) {
          interpT(index) = zs[index];
          interpC(index) = xs[index];
        });
    }
    
    __device__ bool
    AreNeighbors(const double val,
                 ViewVectorDouble arr,
                 const size_t leftIndex,
                 const size_t RightIndex) const
    {
      if (arr(leftIndex) <= val && arr(RightIndex) >= val)
        return true;
      return false;
    }

    /*friend std::istream&
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

      memcpy(interp.interpC.data(), xs.data(), sizeof(double) * xs.size());
      memcpy(interp.interpT.data(), zs.data(), sizeof(double) * zs.size());

      return is;
    }*/

    /*Interp1D(const Interp1D& source)
    {   
      Alloc(source._cols);
      interpT = source.interpT;
      interpC = source.interpC;
      _cols = source._cols;
    }*/

    __device__ void
    FindNeighbourIndices(const double val,
                         ViewVectorDouble arr,
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