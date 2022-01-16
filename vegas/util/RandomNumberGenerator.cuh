#ifndef RANDOM_NUMBER_GENERATOR_CUH
#define RANDOM_NUMBER_GENERATOR_CUH

#include <curand_kernel.h>

class Custom_generator {
  const uint32_t a = 1103515245;
  const uint32_t c = 12345;
  const uint32_t one = 1;
  const uint32_t expi = 31;
  uint32_t p = one << expi;
  uint32_t custom_seed = 0;
  uint64_t temp = 0;

public:
  __device__ Custom_generator(uint32_t seed) : custom_seed(seed){};

  __device__ double
  operator()()
  {
    temp = a * custom_seed + c;
    custom_seed = temp & (p - 1);
    return (double)custom_seed / (double)p;
  }

  __device__ void
  SetSeed(uint32_t seed)
  {
    custom_seed = seed;
  }
};

class Curand_generator {
  curandState localState;

public:
  __device__
  Curand_generator()
  {
    curand_init(0, blockIdx.x, threadIdx.x, &localState);
  }

  __device__
  Curand_generator(unsigned int seed)
  {
    curand_init(seed, blockIdx.x, threadIdx.x, &localState);
  }

  __device__ double
  operator()()
  {
    return curand_uniform_double(&localState);
  }
};

template <typename Generator>
class Random_num_generator {
  Generator generator;

public:
  __device__
  Random_num_generator(unsigned int seed)
    : generator(seed)
  {}
  __device__ double
  operator()()
  {
    return generator();
  }
};

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

  /*template<typename Custom_generator>
  __device__
  constexpr bool
  is_custom_generator(){
      return true;
  }

  template<typename Curand_generator>
  __device__
  constexpr bool
  is_custom_generator(){
      return false;
  }*/
}

#endif
