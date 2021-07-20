#ifndef CUDACUHRE_QUAD_UTIL_CUDADEBUGUTIL_H
#define CUDACUHRE_QUAD_UTIL_CUDADEBUGUTIL_H

#include "cudaPagani/quad/util/cudaArchUtil.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

namespace quad {

#if (defined(DEBUG) || defined(_DEBUG))
#define QUAD_STDERR
#endif

  //__host__ __device__
  cudaError_t
  Debug(cudaError_t error, const char* filename, int line, bool silent = false)
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

  inline void
  __cudaCheckError(const char* file, const int line)
  {
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr,
              "cudaCheckError() failed at %s:%i : %s\n",
              file,
              line,
              cudaGetErrorString(err));
      exit(-1);
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
      fprintf(stderr,
              "cudaCheckError() with sync failed at %s:%i : %s\n",
              file,
              line,
              cudaGetErrorString(err));
      exit(-1);
    }
#endif

    return;
  }
}

#endif
