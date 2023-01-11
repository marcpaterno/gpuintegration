#ifndef RANDOM_NUMBER_GENERATOR_CUH
#define RANDOM_NUMBER_GENERATOR_CUH

#include <curand_kernel.h>

namespace mcubes {

  template <typename T, typename U>
  __device__ __inline__ constexpr bool
  is_same()
  {
    return false;
  }

  template <typename Custom_generator>
  __device__ __inline__ constexpr bool
  is_same<Custom_generator, Custom_generator>()
  {
    return true;
  }

  // try the above to avoid class overhead

  template <typename T, typename U>
  struct TypeChecker {
    // static const bool value = false;
    __device__ static constexpr bool
    is_custom_generator()
    {
      return false;
    }
  };

  template <typename Custom_generator>
  struct TypeChecker<Custom_generator, Custom_generator> {
    // static const bool value = true;

    __device__ static constexpr bool
    is_custom_generator()
    {
      return true;
    }
  };
}

#endif
