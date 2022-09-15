#ifndef CUDACUHRE_QUAD_UTIL_CUDADEBUGUTIL_H
#define CUDACUHRE_QUAD_UTIL_CUDADEBUGUTIL_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "oneAPI/pagani/quad/util/cudaArchUtil.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>


#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>

template<int debug = 0>
class Recorder{
  public:
    std::ofstream outfile;
    
    Recorder() = default;
    
    Recorder(std::string filename){
        if constexpr(debug > 0)
            outfile.open(filename.c_str());
    }
    
    ~Recorder(){
        if constexpr(debug > 0)
            outfile.close();
    }
};


/* Obtain a backtrace and print it to stdout. */
/* This is a hideous C function taken from 
 * https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
 * and modified slightly.
 */
void print_trace() {
  int const MAX_FRAMES =100;
  void *array[MAX_FRAMES];
  char **strings;
  int size, i;

  size = backtrace (array, MAX_FRAMES);
  strings = backtrace_symbols (array, size);
  if (strings != NULL)
  {
    printf ("Obtained %d stack frames.\n", size);
    for (i = 0; i < size; i++)
      printf ("%s\n", strings[i]);
  }
  free (strings);
}

namespace quad {

#if (defined(DEBUG) || defined(_DEBUG))
#define QUAD_STDERR
#endif

  //__host__ __device__
  int
  Debug(int error, const char* filename, int line, bool silent = false)
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
   try {
#ifdef CUDA_ERROR_CHECK
    /*
    DPCT1010:4: SYCL uses exceptions to report errors and does not use the error
    codes. The call was replaced with 0. You need to rewrite this code.
    */
    int err = 0;
    /*
    DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
    */
    if (0 != err) {
      fprintf(stderr,
              "cudaCheckError() failed at %s:%i : %s\n",
              file,
              line,
              /*
              DPCT1009:5: SYCL uses exceptions to report errors and does not use
              the error codes. The original code was commented out and a warning
              string was inserted. You need to rewrite this code.
              */
              "cudaGetErrorString not supported" /*cudaGetErrorString(err)*/);
      /*
      DPCT1001:0: The statement could not be removed.
      */
      print_trace();
      abort();
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    /*
    DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    err = (dpct::get_current_device().queues_wait_and_throw(), 0);
    /*
    DPCT1000:3: Error handling if-stmt was detected but could not be rewritten.
    */
    if (0 != err) {
      fprintf(stderr,
              "cudaCheckError() with sync failed at %s:%i : %s\n",
              file,
              line,
              /*
              DPCT1009:7: SYCL uses exceptions to report errors and does not use
              the error codes. The original code was commented out and a warning
              string was inserted. You need to rewrite this code.
              */
              "cudaGetErrorString not supported" /*cudaGetErrorString(err)*/);
      /*
      DPCT1001:2: The statement could not be removed.
      */
      print_trace();
      abort();
    }
#endif

    return;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }
}

#endif
