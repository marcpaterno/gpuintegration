#include <numeric>
#include <vector>
#include "cuda/cudaPagani/quad/GPUquad/Interp1D.cuh"
#include <iostream>

__global__ void
Evaluate(quad::Interp1D interpolator,
         size_t size,
         double* results)
{
    double val = 1.5;
    results[0] = interpolator(val);
}

int main(){
  const size_t s = 10000000;
  std::vector<double> xs(s);
  std::vector<double> ys(s);
   
  std::iota(xs.begin(), xs.end(), 1.);
  std::iota(ys.begin(), ys.end(), 2.);
  double* results = quad::cuda_malloc_managed<double>(s);

  for(int i=0; i<1000; i++)
  {
    quad::Interp1D interpObj(xs.data(), ys.data(), s);  
    Evaluate<<<1,1>>>(interpObj, s, results);
    cudaDeviceSynchronize();
    size_t free_physmem, total_physmem;
    cudaMemGetInfo(&free_physmem, &total_physmem);
    std::cout<<free_physmem<<"\n"; 
    //std::cout<<"total:"<<total_physmem<<"\n"; 
  }

  cudaFree(results);

}