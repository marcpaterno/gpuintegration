#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <numeric>
#include <vector>
#include "oneAPI/dpct_latest/pagani/quad/GPUquad/Interp1D.dp.hpp"
#include <iostream>
#include "oneAPI/dpct_latest/pagani/quad/GPUquad/Interp2D.dp.hpp"
#include "oneAPI/dpct_latest/pagani/quad/util/cudaMemoryUtil.h"

void
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
void
Evaluate_test_obj(F* f, double* results)
{
    
  for(int i=0; i <1000; i++){
    results[i] = f->operator()();
  }
}

int
main() {
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();
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
    /*
    DPCT1072:57: DPC++ currently does not support getting the available
     * memory on the current device. You may need to adjust the code.
    */
    total_physmem =
      dpct::get_current_device().get_device_info().get_global_mem_size();
    std::cout << "free device mem before host object creation:"<< free_physmem << std::endl;
  
    
    IntegT host_obj(xs_1D.data(), ys_1D.data(), xs_2D, ys_2D, zs_2D);

    /*
    DPCT1072:58: DPC++ currently does not support getting the available
     * memory on the current device. You may need to adjust the code.
    */
    total_physmem =
      dpct::get_current_device().get_device_info().get_global_mem_size();
    std::cout << "free device mem post host object creation:"<< free_physmem << std::endl;
  
    IntegT* device_obj = quad::cuda_copy_to_device/*managed*/(host_obj);
    
    //IntegT* device_obj = quad::cuda_copy_to_managed(host_obj);

    /*
    DPCT1072:59: DPC++ currently does not support getting the available
     * memory on the current device. You may need to adjust the code.
    */
    total_physmem =
      dpct::get_current_device().get_device_info().get_global_mem_size();
    std::cout << "free device mem post device object creation:"<< free_physmem << std::endl;

    q_ct1.parallel_for(
      sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
        Evaluate_test_obj<IntegT>(device_obj, results);
      });
    dev_ct1.queues_wait_and_throw();
    sycl::free(device_obj, q_ct1);
  }

  /*
  DPCT1072:56: DPC++ currently does not support getting the available
   * memory on the current device. You may need to adjust the code.
  */
  total_physmem =
    dpct::get_current_device().get_device_info().get_global_mem_size();
  std::cout << "free device mem at end:"<< free_physmem << std::endl;

  sycl::free(results, q_ct1);
}
