#include <numeric>
#include <vector>
#include "cuda/pagani/quad/GPUquad/Interp1D.cuh"
#include <iostream>
#include "cuda/pagani/quad/GPUquad/Interp2D.cuh"
#include "cuda/pagani/quad/util/cudaMemoryUtil.h"

__global__ void
Evaluate(quad::Interp1D interpolator,
	 
         size_t size,
         double* results)
{
    double val = 1.5;
    results[0] = interpolator(val);
}


template<size_t s, size_t nx, size_t ny>
class Test_object{
public:

  Test_object(double* xs_1D,
	      double* ys_1D,
	      std::array<double, nx> xs_2D,
	      std::array<double, ny> ys_2D,
	      std::array<double, ny*nx> zs_2D):
    obj_1D(xs_1D, ys_1D, s),
    obj_2D(xs_2D, ys_2D, zs_2D){};

  __device__ __host__
  double
  operator()(){
    return obj_1D(1.5)*obj_2D(2.6, 4.1);
  }


  
  quad::Interp1D obj_1D;
  quad::Interp2D obj_2D;

};

/*void interp_on_cpu_stack(){
  const size_t s = 10000000;
  std::vector<double> xs_1D(s);
  std::vector<double> ys_1D(s);
   
  std::iota(xs_1D.begin(), xs_1D.end(), 1.);
  std::iota(ys_1D.begin(), ys_1D.end(), 2.);
  double* results = quad::cuda_malloc<double>(s);


  constexpr std::size_t nx = 3; // rows
  constexpr std::size_t ny = 2; // cols                                                                                                
  std::array<double, nx> xs_2D = {1., 2., 3.};
  std::array<double, ny> ys_2D = {4., 5.};
  std::array<double, ny * nx> zs_2D;
  auto fxy = [](double x, double y) { return 3 * x * y + 2 * x + 4 * y; };

  for (std::size_t i = 0; i != nx; ++i) {
    double x = xs[i];

    for (std::size_t j = 0; j != ny; ++j) {
      double y = ys[j];
      zs[j * nx + i] = fxy(x, y);
    }
  }

  quad::Interp2D 2D(xs_2D, ys_2D, zs_2D);
  
  for(int i=0; i<1000; i++)
    {
      quad::Interp1D 1D(xs.data(), ys.data(), s);  
      Evaluate<<<1,1>>>(1D, s, results);
      cudaDeviceSynchronize();
      size_t free_physmem, total_physmem;
      cudaMemGetInfo(&free_physmem, &total_physmem);
      std::cout<<free_physmem<<"\n"; 
      //std::cout<<"total:"<<total_physmem<<"\n"; 
    }

  cudaFree(results);
}*/

template<typename F>
__global__ void
Evaluate_test_obj(F* f, double* results)
{
    
  for(int i=0; i <1000; i++){
    results[i] = f->operator()();
  }
}


int main(){
  constexpr size_t s = 100000;
  std::vector<double> xs_1D(s);
  std::vector<double> ys_1D(s);
   
  std::iota(xs_1D.begin(), xs_1D.end(), 1.);
  std::iota(ys_1D.begin(), ys_1D.end(), 2.);
  
  constexpr std::size_t nx = 3; // rows
  constexpr std::size_t ny = 2; // cols                                                                                                
  std::array<double, nx> xs_2D = {1., 2., 3.};
  std::array<double, ny> ys_2D = {4., 5.};
  std::array<double, ny * nx> zs_2D;
  auto fxy = [](double x, double y) { return 3 * x * y + 2 * x + 4 * y; };
  
  for (std::size_t i = 0; i != nx; ++i) {
    double x = xs_2D[i];

    for (std::size_t j = 0; j != ny; ++j) {
      double y = ys_2D[j];
      zs_2D[j * nx + i] = fxy(x, y);
    }
  }

  double* results = quad::cuda_malloc<double>(1000);

  using IntegT = Test_object<s, nx, ny>;
  size_t free_physmem, total_physmem;
  
  {
    cudaMemGetInfo(&free_physmem, &total_physmem);
    std::cout << "free device mem before host object creation:"<< free_physmem << std::endl;
  
    
    IntegT host_obj(xs_1D.data(), ys_1D.data(), xs_2D, ys_2D, zs_2D);
  
    cudaMemGetInfo(&free_physmem, &total_physmem);
    std::cout << "free device mem post host object creation:"<< free_physmem << std::endl;
  
    IntegT* device_obj = quad::cuda_copy_to_device/*managed*/(host_obj);
    
    //IntegT* device_obj = quad::cuda_copy_to_managed(host_obj);
    
    cudaMemGetInfo(&free_physmem, &total_physmem);
    std::cout << "free device mem post device object creation:"<< free_physmem << std::endl;
  
    Evaluate_test_obj<IntegT><<<1,1>>>(device_obj, results);
    cudaDeviceSynchronize();
    cudaFree(device_obj);
  }
  
  cudaMemGetInfo(&free_physmem, &total_physmem);
  std::cout << "free device mem at end:"<< free_physmem << std::endl;
  
  cudaFree(results);
}
