#ifndef CUDACUHRE_QUAD_UTIL_CUDADEBUGUTIL_H
#define CUDACUHRE_QUAD_UTIL_CUDADEBUGUTIL_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "oneAPI/dpct_latest/mcubes/cudaArchUtil.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

namespace quad {

#if (defined(DEBUG) || defined(_DEBUG))
#define QUAD_STDERR
#endif

  //__host__ __device__
  int Debug(int error, const char *filename, int line, bool silent = false)
  {

#ifdef QUAD_STDERR
    if (error && !silent) {
#if (CUB_PTX_ARCH == 0)
      fprintf(stderr,
              "CUDA error %d [%s, %d]: %s\n",
              error,
              filename,
              line,
              cudaGetErrorString(error));
      fflush(stderr);
#elif (CUB_PTX_ARCH >= 200)
      printf("CUDA error %d [block %d, thread %d, %s, %d]\n",
             error,
             blockIdx.x,
             threadIdx.x,
             filename,
             line);
#endif
    }
#endif
    return error;
  }

  /**
   * \brief Debug macro
   */
#define QuadDebug(e) quad::Debug((e), __FILE__, __LINE__)

#define QuadDebugExit(e)                                                       \
  if (quad::Debug((e), __FILE__, __LINE__)) {                                  \
    exit(1);                                                                   \
  }

  void
  Println(std::ostream& out, std::string s)
  {
    out << s << std::endl;
    fflush(stdout);
  }
#define Print(s)                                                               \
  puts(s);                                                                     \
  fflush(stdout)

#define CUDA_ERROR_CHECK
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

  inline void __cudaCheckError(const char *file, const int line) try {
#ifdef CUDA_ERROR_CHECK
    /*
    
    *** This error check is not required. Left as omitted ~ Emmanuel ***
    
    DPCT1010:0: SYCL uses exceptions to report errors and does not use the error
    codes. The call was replaced with 0. You need to rewrite this code.
    */
    int err = 0;

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    /*
    DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    err = (dpct::get_current_device().queues_wait_and_throw(), 0);

#endif

    return;
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }
}

#endif