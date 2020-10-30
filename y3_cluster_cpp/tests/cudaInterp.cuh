#include "catch2/catch.hpp"

#include "modules/sigma_miscent_y1_scalarintegrand.hh"
#include "../cudaCuhre/quad/util/cudaArray.cuh"

#include <iostream>
#include <chrono>					
#include "utils/str_to_doubles.hh"  
#include <vector> 					

#include <fstream>
#include <stdexcept>
#include <string>
#include <array>

#ifndef CUDAINTERP_CUH
#define CUDAINTERP_CUH

#include <stdio.h>
#include <vector> 
#include <array>
#include "utils/str_to_doubles.hh"  

namespace quad {	
  class Managed 
  {
  public:
    void *operator new(size_t len) {
      void *ptr;
      cudaMallocManaged(&ptr, len);
      cudaDeviceSynchronize();
      return ptr;
    }

    void operator delete(void *ptr) {
      cudaDeviceSynchronize();
      cudaFree(ptr);
    }
  };	
	
  class Interp2D : public Managed{
  public:
    __host__ __device__
    Interp2D(){}
    //change names to xs, ys, zs to fit with y3_cluster_cpp::Interp2D
    double* interpT;
    double* interpR;
    double* interpC;
    size_t _rows;
    size_t _cols;
		
    ~Interp2D(){
      //cudaFree(interpT);
      //cudaFree(interpR);
      //cudaFree(interpC);
    }
		
    void Alloc(size_t cols, size_t rows){
      _rows = rows;
      _cols = cols;
      cudaMallocManaged((void**)&interpR, sizeof(double)*_rows);
      cudaMallocManaged((void**)&interpC, sizeof(double)*_cols);
      cudaMallocManaged((void**)&interpT, sizeof(double)*_rows*_cols);
    }
		
    template<size_t M, size_t N>
    Interp2D(std::array<double, M> const& xs, std::array<double, N> const& ys, std::array<double, (N)*(M)> const& zs)
    {
      Alloc(M, N);
      memcpy(interpR, ys.data(), sizeof(double)*N);
      memcpy(interpC, xs.data(), sizeof(double)*M);
      memcpy(interpT, zs.data(), sizeof(double)*N*M);
    }
		
    Interp2D(double* xs, double* ys, double* zs, size_t cols, size_t rows)
    {
      Alloc(cols, rows);
      memcpy(interpR, ys, sizeof(double)*rows);
      memcpy(interpC, xs, sizeof(double)*cols);
      memcpy(interpT, zs, sizeof(double)*rows*cols);
    }
		
    //Interp2D(std::vector<double>&& xs, std::vector<double>&& ys, std::vector<std::vector<double>> const& zs)
    //Look at mor_des_t model, is the data in .cc file given in wrong format?
    template<size_t M, size_t N>
    Interp2D(std::array<double, M> xs, std::array<double, N> ys, std::array<std::array<double, N>, M> zs)
    {		
      Alloc(M, N);
      memcpy(interpR, ys.data(), sizeof(double)*N);
      memcpy(interpC, xs.data(), sizeof(double)*M);
			
			
      for (std::size_t i = 0; i < M; ++i) {
	std::array<double, N> const& row = zs[i];
	for (std::size_t j = 0; j < N; ++j) {
	  interpT[i + j * M] = row[j];
	  //interpT[i + j * N] = zs[i][j];
	  //printf("custom: row[%lu]:%f -> InterpT[%lu]\n", j, row[j], i+j*N);
	}
      }
      //printf("%f, %f, %f, %f\n", min_x(), max_x(), min_y(), max_y());
    }
		
    __device__ 
    bool AreNeighbors(const double val, double* arr, const size_t leftIndex, const size_t RightIndex) const{
      //printf("[%i](%i) evaluating neighbors %f, %f against val:%f index:%lu\n", blockIdx.x, threadIdx.x, arr[leftIndex], arr[RightIndex], val, leftIndex);
      if(arr[leftIndex] <= val && arr[RightIndex] >= val)
	return true;
      return false;
    }
		
    friend std::istream&
    operator>>(std::istream& is, Interp2D& interp)
    {
      assert(is.good());
      std::string buffer;
      std::getline(is, buffer);
      std::vector<double> xs = cosmosis::str_to_doubles(buffer);
      std::getline(is, buffer);
      std::vector<double> ys = cosmosis::str_to_doubles(buffer);
      std::getline(is, buffer);
      std::vector<double> zs  = cosmosis::str_to_doubles(buffer);
		  
      cudaMallocManaged((void**)&(*&interp), sizeof(Interp2D));
		  
      interp._cols = xs.size();
      interp._rows = ys.size();
		  
		  
      cudaMallocManaged((void**)&interp.interpR, sizeof(double)*ys.size());
      cudaDeviceSynchronize();
      cudaMallocManaged((void**)&interp.interpC, sizeof(double)*xs.size());
      cudaDeviceSynchronize();
      cudaMallocManaged((void**)&interp.interpT, sizeof(double)*zs.size());
      cudaDeviceSynchronize();
		  
      memcpy(interp.interpR, ys.data(), sizeof(double)*ys.size());
      memcpy(interp.interpC, xs.data(), sizeof(double)*xs.size());
      memcpy(interp.interpT, zs.data(), sizeof(double)*zs.size());
		  
      return is;
    }
		
    Interp2D(const Interp2D &source) {
      //cudaMallocManaged((void**)source, sizeof(Interp2D));
      Alloc(source._cols, source._rows);
      interpT = source.interpT;
      interpC = source.interpC;
      interpR = source.interpR;
      _cols = source._cols;
      _rows = source._rows;
      //printf("Interp2D copy constructor called: %lu vs %lu\n", _cols, source._cols);
    } 
		
    __device__
    void FindNeighbourIndices(const double val, double* arr, const size_t size, size_t& leftI, size_t& rightI) const{
	
      size_t currentIndex = size/2;
      leftI = 0;
      rightI = size - 1;

      //while(currentIndex != 0 && currentIndex != lastIndex){
      while(leftI<=rightI){
	//currentIndex = leftI + (rightI - leftI)/2;
	currentIndex = (rightI+leftI)*0.5;
	/*if(val == 244.)
	  printf("[%i](%i) looking for %f l:%lu r:%lu checking:%f & %f currentIndex:%lu\n", blockIdx.x, 
	  threadIdx.x,  
	  val,
	  leftI, 
	  rightI,  
	  arr[leftI], 
	  arr[currentIndex],
	  currentIndex);*/
	/*printf("[%i](%i) looking for %f l:%lu r:%lu checking:%f & %f currentIndex:%lu\n", blockIdx.x, 
	  threadIdx.x,  
	  val,
	  leftI, 
	  rightI,  
	  arr[leftI], 
	  arr[rightI],
	  currentIndex);*/
	//if(AreNeighbors(val, arr, currentIndex-1, currentIndex)){
	if(AreNeighbors(val, arr, currentIndex, currentIndex+1)){
	  //leftI = currentIndex -1;
	  //rightI = currentIndex;
	  leftI = currentIndex;
	  rightI = currentIndex+1;
	  //printf("returning indices :%lu, %lu\n", leftI, rightI);
	  return;
	}
				
	if(arr[currentIndex] > val){
	  rightI = currentIndex;
	}
	else{
	  leftI = currentIndex;
	}
	//printf("[%i](%i) currentIndex:%lu\n", currentIndex);
      }
    }
		
    __device__ double
    operator()(double x, double y) const
    {
      //printf("custom: interpolating on %.20f, %.20f\n", x, y);
      // for(int i=0; i< _rows*_cols; ++i)
      //	printf("interpT[%i]:%f\n", i, interpT[i]);
	
      //y1, y2, x1, x2, are the indices of where to find the four neighbouring points in the z-table
      size_t y1 = 0, y2 = 0;
      size_t x1 = 0, x2 = 0;
			
      FindNeighbourIndices(y, interpR, _rows, y1, y2);
      FindNeighbourIndices(x, interpC, _cols, x1, x2);
      //printf("interpolation indices(%lu, %lu), (%lu, %lu)\n", x1, x2, y1, y2);
      //this is how  zij is accessed by gsl2.6 Interp2D i.e. zij = z[j*xsize+i], where i=0,...,xsize-1, j=0, ..., ysize-1
      const double q11 = interpT[y1*_cols + x1];
      const double q12 = interpT[y2*_cols + x1];
      const double q21 = interpT[y1*_cols + x2];
      const double q22 = interpT[y2*_cols + x2];
		  
      const double x1_val = interpC[x1];
      const double x2_val = interpC[x2];
      const double y1_val = interpR[y1];
      const double y2_val = interpR[y2];
		  
      const double f_x_y1 = q11*(x2_val-x)/(x2_val-x1_val) + q21*(x-x1_val)/(x2_val-x1_val);
      const double f_x_y2 = q12*(x2_val-x)/(x2_val-x1_val) + q22*(x-x1_val)/(x2_val-x1_val);
		  
      //printf("indices of neighbors:x1:%lu, x2:%lu, y1:%lu, y2:%lu\n", x1, x2, y1, y2);
      //printf("q11:%f, q12:%f, q21:%f, q22:%f\n", q11, q12, q21, q22);
		  
      double f_x_y = 0.;
      f_x_y = f_x_y1*(y2_val-y)/(y2_val-y1_val) + f_x_y2*(y-y1_val)/(y2_val-y1_val); 
      return f_x_y;
    }
		
    __device__ double
    min_x() const{ 
      //printf("min_x:%f\n", interpC[0]);
      //printf("min_x:\n");
      //printf("min _cols:%lu\n", _cols);
      //printf("interpC[0]:%lu\n", interpC[0]);
      return interpC[0]; }
		
    __device__ double
    max_x() const{ 
      //printf("cols:%lu max_x:%f\n", _cols, interpC[_cols-1]);
      //printf("max_x\n");
      return interpC[_cols-1]; }
		
    __device__   double
    min_y() const{ 
      //printf("min_y\n");
      //printf("min_y\n");
      return interpR[0]; }
		
    __device__ double
    max_y() const{ 
      //printf("max_y\n");
      return interpR[_rows-1]; }
		
    __device__  double
    do_clamp(double v, double lo, double hi) const
    {
      //printf("inside do_clamp\n");
      //printf("[%i](%i)do clamp (%f, %f, %f)\n", blockIdx.x, threadIdx.x, v, lo, hi);
      assert(!(hi < lo));
      return (v < lo) ? lo : (hi < v) ? hi : v;
    }
		
    __device__ double
    eval(double x, double y) const
    {
      //printf("[%i](%i)evaluate interpolator at  (%f, %f)\n", blockIdx.x, threadIdx.x, x, y);
      return this->operator()(x, y);
    };
		
    __device__ 
    double
    clamp(double x, double y) const
    {
      //printf("[%i](%i)clamp (%f, %f)\n", blockIdx.x, threadIdx.x, x, y);
      return eval(do_clamp(x, min_x(), max_x()), do_clamp(y, min_y(), max_y()));
    }
  };
	
  class Interp1D : public Managed{
  public:
    __host__ __device__
    Interp1D(){}
    //change names to xs, ys, zs to fit with y3_cluster_cpp::Interp2D
    double* interpT;
    double* interpC;
    size_t _cols;
		
    ~Interp1D(){
      //cudaFree(interpT);
      //cudaFree(interpC);
    }
		
    void Alloc(size_t cols){
      _cols = cols;
      cudaMallocManaged((void**)&interpC, sizeof(double)*_cols);
      cudaMallocManaged((void**)&interpT, sizeof(double)*_cols);
    }
		
    template<size_t M>
    Interp1D(std::array<double, M> const& xs, std::array<double,M> const& zs)
    {
      Alloc(M);
      memcpy(interpC, xs.data(), sizeof(double)*M);
      memcpy(interpT, zs.data(), sizeof(double)*M);
    }
		
    Interp1D(double* xs, double* ys, double* zs, size_t cols)
    {
      Alloc(cols);
      memcpy(interpC, xs, sizeof(double)*cols);
      memcpy(interpT, zs, sizeof(double)*cols);
    }
		
    __device__ 
    bool AreNeighbors(const double val, double* arr, const size_t leftIndex, const size_t RightIndex) const{
      if(arr[leftIndex] <= val && arr[RightIndex] >= val)
	return true;
      return false;
    }
		
    friend std::istream&
    operator>>(std::istream& is, Interp1D& interp)
    {
      assert(is.good());
      std::string buffer;
      std::getline(is, buffer);
      std::vector<double> xs = cosmosis::str_to_doubles(buffer);
      std::getline(is, buffer);
      std::vector<double> zs  = cosmosis::str_to_doubles(buffer);
		  
      cudaMallocManaged((void**)&(*&interp), sizeof(Interp1D));
      cudaDeviceSynchronize();
		  
      interp._cols = xs.size();
		  
      cudaMallocManaged((void**)&interp.interpC, sizeof(double)*xs.size());
      cudaDeviceSynchronize();
      cudaMallocManaged((void**)&interp.interpT, sizeof(double)*zs.size());
      cudaDeviceSynchronize();
		  
      memcpy(interp.interpC, xs.data(), sizeof(double)*xs.size());
      memcpy(interp.interpT, zs.data(), sizeof(double)*zs.size());
		  
      return is;
    }
		
    Interp1D(const Interp1D &source) {
      Alloc(source._cols);
      interpT = source.interpT;
      interpC = source.interpC;
      _cols = source._cols;
    } 
		
    __device__
    void FindNeighbourIndices(const double val, double* arr, const size_t size, size_t& leftI, size_t& rightI) const{

      size_t currentIndex = size/2;
      leftI = 0;
      rightI = size - 1;

      //while(currentIndex != 0 && currentIndex != lastIndex){
      while(leftI<=rightI){
	//currentIndex = leftI + (rightI - leftI)/2;
	currentIndex = (rightI + leftI)*0.5;
	if(AreNeighbors(val, arr, currentIndex, currentIndex+1)){
	  leftI = currentIndex;
	  rightI = currentIndex+1;
	  return;
	}
				
	if(arr[currentIndex] > val){
	  rightI = currentIndex;
	}
	else{
	  leftI = currentIndex;
	}
      }
    }
		
    __device__ double
    operator()(double x) const
    {
      size_t x0_index = 0, x1_index = 0;
      FindNeighbourIndices(x, interpC, _cols, x0_index, x1_index);
      const double y0 = interpT[x0_index];
      const double y1 = interpT[x1_index];
      const double x0 = interpC[x0_index];
      const double x1 = interpC[x1_index];
      const double y = (y0*(x1 - x) + y1*(x - x0))/(x1 - x0);
      return y;
    }
		
    __device__ double
    min_x() const{ return interpC[0]; }
		
    __device__ double
    max_x() const{ return interpC[_cols-1]; }
		
    __device__  double
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
		
    __device__ 
    double
    clamp(double x) const
    {
      return eval(do_clamp(x, min_x(), max_x()));
    }
  };
	
  template<size_t Order>
  class polynomial{
  private:
    const gpu::cudaArray<double, Order> coeffs;
  public:
    __host__ __device__
    polynomial(gpu::cudaArray<double, Order> coeffs) : coeffs(coeffs) {}
    
    __host__ __device__
    constexpr double
    operator()(const double x) const{
      double out = 0.0;
      for (auto i = 0u; i < Order; i++)
	out = coeffs[i] + x * out;
      return out;
    }
  };
}

struct GPU {
  template<size_t order>
  using polynomial = quad::polynomial<order>;
  typedef quad::Interp2D Interp2D;
  typedef quad::Interp1D Interp1D;
  //typedef quad::ez ez;
};

struct CPU {
  template<size_t order>
  using polynomial = y3_cluster::polynomial<order>;
  typedef y3_cluster::Interp2D Interp2D;
  typedef y3_cluster::Interp1D Interp1D;
};

#endif
