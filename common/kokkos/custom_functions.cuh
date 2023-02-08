#ifndef KOKKOS_QUAD_UTIL_CUDA_CUSTOM_FUNCTIONS_CUH
#define KOKKOS_QUAD_UTIL_CUDA_CUSTOM_FUNCTIONS_CUH

#include <iostream>
#include <limits>
#include "kokkos/pagani/quad/GPUquad/Sample.cuh"
#include "common/kokkos/cudaMemoryUtil.h"

/*
    require blocks to be equal to size
*/

template <typename T>
T
custom_reduce(Kokkos::View<T*, Kokkos::CudaSpace> arr, size_t size)
{
  T res = 0.;
  Kokkos::parallel_reduce(
    "Estimate computation",
    size,
    KOKKOS_LAMBDA(const int64_t index, T& valueToUpdate) {
      valueToUpdate += arr(index);
    },
    res);

  return res;
}

template <typename T1, typename T2>
T2
custom_inner_product(Kokkos::View<T1*, Kokkos::CudaSpace> arr1,
                     Kokkos::View<T2*, Kokkos::CudaSpace> arr2)
{
  size_t size = std::min(arr1.extent(0), arr2.extent(0));
  T2 res;
  Kokkos::parallel_reduce(
    "ProParRed1",
    size,
    KOKKOS_LAMBDA(const int64_t index, T2& valueToUpdate) {
      valueToUpdate += static_cast<T2>(arr1(index)) * arr2(index);
    },
    res);
  return res;
}

template <typename T>
double
ComputeMax(Kokkos::View<T*, Kokkos::CudaSpace> list)
{
  T max;
  Kokkos::parallel_reduce(
    list.extent(0),
    KOKKOS_LAMBDA(const int& index, T& lmax) {
      if (lmax < list(index))
        lmax = list(index);
    },
    Kokkos::Max<T>(max));
  return max;
}

template <typename T>
T
ComputeMin(Kokkos::View<T*, Kokkos::CudaSpace> list)
{
  T min;
  Kokkos::parallel_reduce(
    list.extent(0),
    KOKKOS_LAMBDA(const int& index, T& lmin) {
      if (lmin > list(index))
        lmin = list(index);
    },
    Kokkos::Min<T>(min));
  return min;
}

template <typename T>
std::pair<T, T>
min_max(Kokkos::View<T*, Kokkos::CudaSpace> input)
{
  return {ComputeMin<T>(input), ComputeMax<T>(input)};
}

#endif
