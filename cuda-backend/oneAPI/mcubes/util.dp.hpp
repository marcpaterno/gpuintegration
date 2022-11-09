#ifndef VEGAS_UTIL_UTIL_CUH
#define VEGAS_UTIL_UTIL_CUH

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

struct IntegrationResult {
  double estimate = 0;
  double errorest = 0;
  double chi_sq = 0.;
  int status = 1;
};

template <typename T>
void
PrintArray(T* array, int size, std::string label)
{
  for (int i = 0; i < size; i++)
    std::cout << label << "[" << i << "]:" << array[i] << "\n";
}

template <class T>
T*
cuda_malloc_managed(size_t size)
{
  T* temp = nullptr;
  
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
  auto rc = (temp = (T *)sycl::malloc_shared(sizeof(T) * size,
                                             q_ct1),
             0);
  if (rc != 0)
    throw std::bad_alloc();
  return temp;
}

template <class T> T *cuda_malloc_managed() try {
  T* temp = nullptr;
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
  auto rc =
      (temp = (T *)sycl::malloc_shared(sizeof(T), q_ct1),
       0);
  if (rc != 0)
    throw std::bad_alloc();
  return temp;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <class T>
T*
cuda_copy_to_managed(T const& on_host)
{
  T* buffer = cuda_malloc_managed<T>();
  try {
    new (buffer) T(on_host);
  }
  catch (...) {
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    sycl::free(buffer, q_ct1);
    throw;
  }
  return buffer;
}
#endif
