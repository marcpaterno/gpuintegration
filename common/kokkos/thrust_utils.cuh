#ifndef KOKKOS_QUAD_THRUST_UTILS_CUH
#define KOKKOS_QUAD_THRUST_UTILS_CUH

#include "common/kokkos/custom_functions.cuh"
#include "common/kokkos/cudaMemoryUtil.h"
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_team_dot.hpp>

template <typename T, bool use_custom = false>
T
dot_product(Kokkos::View<T*, Kokkos::CudaSpace> arr1,
            Kokkos::View<T*, Kokkos::CudaSpace> arr2)
{
  if constexpr (use_custom == false) {
    return KokkosBlas::dot(arr1, arr2);
  }

  T res = custom_inner_product<T, T>(arr1, arr2);
  return res;
}

template <typename T1, typename T2, bool use_custom = false>
T2
dot_product(Kokkos::View<T1*, Kokkos::CudaSpace> arr1,
            Kokkos::View<T2*, Kokkos::CudaSpace> arr2)
{
  T2 res = custom_inner_product<T1, T2>(arr1, arr2);
  return res;
}

template <typename T, bool use_custom = false>
T
reduction(Kokkos::View<T*, Kokkos::CudaSpace> arr, size_t size)
{
  if constexpr (use_custom == false) {
    std::cerr << "no library use for reduction in kokkos" << std::endl;
    exit(1);
  }
  return custom_reduce(arr, size);
}

template <typename T, bool use_custom = false>
T
exclusive_scan(Kokkos::View<T*, Kokkos::CudaSpace> input,
               Kokkos::View<T*, Kokkos::CudaSpace> output)
{
  if constexpr (use_custom == false) {
    std::cerr << "no library use for exclusive_scan in kokkos" << std::endl;
    exit(1);
    return -1;
  } else {

    int update = 0.;
    Kokkos::parallel_scan(
      input.extent(0),
      KOKKOS_LAMBDA(const int i, int& update, const bool final) {
        const int val_i = input(i);
        if (final) {
          output(i) = update;
        }
        update += val_i;
      });
    return update;
  }
}

template <typename T, bool use_custom = false>
quad::Range<T>
device_array_min_max(Kokkos::View<T*, Kokkos::CudaSpace> arr)
{
  quad::Range<T> range;
  if (use_custom == false) {
    std::cerr << "no library use for min_max in kokkos" << std::endl;
    exit(1);
    return range;
  }

  auto res = min_max<T>(arr);
  range.low = res.first;
  range.high = res.second;
  return range;
}

#endif