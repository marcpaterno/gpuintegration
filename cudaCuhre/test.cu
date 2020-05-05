#include "function.cu"
#include "quad/quad.h"
#include "quad/util/cudaUtil.h"
#include <iomanip>
#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "quad/GPUquad/GPUquad.cu"

#define EPSABS 1e-12

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose = false;

int
main(int argc, char** argv)
{
  return 0;
}
