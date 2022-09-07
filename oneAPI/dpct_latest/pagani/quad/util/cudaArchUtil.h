#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef CUDACUHRE_QUAD_UTIL_CUDAARCH_UTIL_H
#define CUDACUHRE_QUAD_UTIL_CUDAARCH_UTIL_H
namespace quad {

  /// QUAD_PTX_ARCH reflects the PTX version targeted by the active compiler
  /// pass (or zero during the host pass).
#ifndef DPCT_COMPATIBILITY_TEMP
#define QUAD_PTX_ARCH 0
#else
#define QUAD_PTX_ARCH DPCT_COMPATIBILITY_TEMP
#endif
}

#endif
