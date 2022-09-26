#ifndef VEGAS_UTIL_UTIL_CUH
#define VEGAS_UTIL_UTIL_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
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
  /*
  DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  auto rc = (temp = (T *)sycl::malloc_shared(sizeof(T) * size,
                                             dpct::get_default_queue()),
             0);
  if (rc != 0)
    throw std::bad_alloc();
  return temp;
}

template <class T> T *cuda_malloc_managed() try {
  T* temp = nullptr;
  /*
  DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  auto rc =
      (temp = (T *)sycl::malloc_shared(sizeof(T), dpct::get_default_queue()),
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
    sycl::free(buffer, dpct::get_default_queue());
    throw;
  }
  return buffer;
}
#endif