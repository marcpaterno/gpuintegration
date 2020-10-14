#ifndef GPUQUADINTERP2D_H
#define GPUQUADINTERP2D_H

#include "../util/cudaMemoryUtil.h"
#include "../util/cudaTimerUtil.h"
#include "../util/cudaArray.cuh"

namespace old_quad {
	
  template<typename T>	
  class Interp2D {
  
  public:
    Interp2D(){};
	
    T* interpT;
    T* interpR;
    T* interpC;
    size_t _rows;
    size_t _cols;
	
    __host__
	Interp2D(T* xs, T* ys, T* zs, size_t cols, size_t rows){
		cudaMalloc((void**)&interpR, sizeof(T)*rows);
		cudaMalloc((void**)&interpC, sizeof(T)*cols);
		cudaMalloc((void**)&interpT, sizeof(T)*rows*cols);
		
		cudaMemcpy(interpR, ys, sizeof(T)*rows, cudaMemcpyHostToDevice);
		cudaMemcpy(interpC, xs, sizeof(T)*cols, cudaMemcpyHostToDevice);
		cudaMemcpy(interpT, zs, sizeof(T)*rows*cols, cudaMemcpyHostToDevice);
		
		_rows = rows;
		_cols = cols;
    }
	
	__device__ 
	bool AreNeighbors(const T val, T* arr, const size_t leftIndex, const size_t RightIndex){
		if(arr[leftIndex] < val && arr[RightIndex] > val)
			return true;
		return false;
	}
	
	//what to return if value  is beyond left or right boundary?
	__device__
	void FindNeighbourIndices(const T val, T* arr, const size_t size, size_t& leftI, size_t& rightI){
		size_t currentIndex = size/2;
		size_t lastIndex = size - 1;
		
		while(currentIndex != 0 && currentIndex != lastIndex){
			if(AreNeighbors(val, arr, currentIndex-1, currentIndex)){
				leftI = currentIndex -1;
				rightI = currentIndex;
				return;
			}
			
			currentIndex = arr[currentIndex] > val ? currentIndex /= 2 : currentIndex + (size-currentIndex)/2;
		}
		//values can't be found, how to handle?
		leftI  = 0;
		rightI = 0;
	}
	
    __device__ double
    operator()(double x, double y)
    {
	  size_t y1 = 0, y2 = 0;
	  size_t x1 = 0, x2 = 0;
	 
	  FindNeighbourIndices(y, interpR, _rows, y1, y2);
	  FindNeighbourIndices(x, interpC, _cols, x1, x2);
	  
	  /*printf("x1:%lu\n", x1);
	  printf("x2:%lu\n", x2);
	  printf("y1:%lu\n", y1);
	  printf("y2:%lu\n", y2);*/
	  
	  T q11 = __ldg(&interpT[x1*_cols + y1]);
	  T q12 = __ldg(&interpT[x1*_cols + y2]);
	  T q21 = __ldg(&interpT[x2*_cols + y1]);
	  T q22 = __ldg(&interpT[x2*_cols + y2]);
	  
	  double t1 = (x2 - x) / ((x2 - x1) * (y2 - y1));
      double t2 = (x - x1) / ((x2 - x1) * (y2 - y1));
	  
      return ((q11 * (y2 - y) + q12 * (y - y1)) * t1 + (q21 * (y2 - y) + q22 * (y - y1)) * t2);
    }
  };
}

#endif
