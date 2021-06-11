#ifndef KOKKOSCUHRE_UTIL
#define KOKKOSCUHRE_UTIL

#include "quad.h"
#include <iostream>

template <class Type>
void
EasyPrint(Kokkos::View<Type*, Kokkos::CudaSpace> list)
{
  // Kokkos::View<Type>::HostMirror cpulist  = Kokkos::create_mirror_view(list);
  // //left coordinate of bin should use mirror instead
  std::cout.precision(17);
  size_t list_size = list.extent(0);
  Kokkos::View<Type*> cpulist("cpulist", list_size);
  Kokkos::deep_copy(cpulist, list);

  for (size_t index = 0; index < list_size; ++index) {
    std::cout << "list[" << index << "]:" << cpulist(index) << std::endl;
  }
}

template <class Type>
void
constEasyPrint(Kokkos::View<const Type*, Kokkos::CudaSpace> list)
{
  // Kokkos::View<Type>::HostMirror cpulist  = Kokkos::create_mirror_view(list);
  // //left coordinate of bin should use mirror instead
  size_t list_size = list.extent(0);
  Kokkos::parallel_for(
    "Printing",
    team_policy(1, 1),
    KOKKOS_LAMBDA(const member_type team_member) {
      for (int i = 0; i < list_size; i++)
        printf("list[%i]:%.15f\n", i, list(i));
    });
}

#endif