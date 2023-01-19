#ifndef CUDACUHRE_QUAD_UTIL_CUDADEBUGUTIL_H
#define CUDACUHRE_QUAD_UTIL_CUDADEBUGUTIL_H

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
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

template <int debug = 0>
class Recorder {
public:
  std::ofstream outfile;

  Recorder() = default;

  Recorder(std::string filename)
  {
    if constexpr (debug > 0)
      outfile.open(filename.c_str());
  }

  ~Recorder()
  {
    if constexpr (debug > 0)
      outfile.close();
  }
};

/* Obtain a backtrace and print it to stdout. */
/* This is a hideous C function taken from
 * https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
 * and modified slightly.
 */
void
print_trace()
{
  int const MAX_FRAMES = 100;
  void* array[MAX_FRAMES];
  char** strings;
  int size, i;

  size = backtrace(array, MAX_FRAMES);
  strings = backtrace_symbols(array, size);
  if (strings != NULL) {
    printf("Obtained %d stack frames.\n", size);
    for (i = 0; i < size; i++)
      printf("%s\n", strings[i]);
  }
  free(strings);
}

namespace quad {

#if (defined(DEBUG) || defined(_DEBUG))
#define QUAD_STDERR
#endif


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
    
    int err = 0;
    
    if (0 != err) {
      fprintf(stderr,
              "cudaCheckError() failed at %s:%i : %s\n",
              file,
              line,
              
              "cudaGetErrorString not supported" /*cudaGetErrorString(err)*/);
      
      print_trace();
      abort();
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    
    //err = (dpct::get_current_device().queues_wait_and_throw(), 0);
    
    if (0 != err) {
      fprintf(stderr,
              "cudaCheckError() with sync failed at %s:%i : %s\n",
              file,
              line,
              
              "cudaGetErrorString not supported" /*cudaGetErrorString(err)*/);
      
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
