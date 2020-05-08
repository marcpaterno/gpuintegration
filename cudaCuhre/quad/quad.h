
#define TIMING_DEBUG 1
#define BLOCK_SIZE 256
#define SM_REGION_POOL_SIZE 128

#define GLOBAL_ERROR 1
#define MAX_GLOBALPOOL_SIZE 2048

//#define TYPE double

static int FIRST_PHASE_MAXREGIONS = (1 << 14);

__constant__ TYPE errcoeff[] = {5, 1, 5};

// Utilities
#include "util/cudaArchUtil.h"
#include "util/cudaDebugUtil.h"

template <typename T>
class Structures {
public:
  __host__ __device__
  Structures()
    : _gpuG(NULL)
    , _cRuleWt(NULL)
    , _GPUScale(NULL)
    , _gpuGenPos(NULL)
    , _gpuGenPermGIndex(NULL)
    , _gpuGenPermVarCount(NULL)
    , _gpuGenPermVarStart(NULL)
    , _cGeneratorCount(NULL)
    , _GPUNorm(NULL)
  {}

  T* const __restrict__ _gpuG;
  T* const __restrict__ _cRuleWt;
  T* const __restrict__ _GPUScale;
  T* const __restrict__ _GPUNorm;
  int* const __restrict__ _gpuGenPos;
  int* const __restrict__ _gpuGenPermGIndex;
  int* const __restrict__ _gpuGenPermVarCount;
  int* const __restrict__ _gpuGenPermVarStart;
  size_t* const __restrict__ _cGeneratorCount;
};

typedef struct {
  double avg, err;
  int bisectdim;
} Result;

typedef struct {
  double lower, upper;
} Bounds;

typedef struct {
  double unScaledLower, unScaledUpper;
} GlobalBounds;

template<int dim>
struct Region{
  int div;
  Result result;
  Bounds bounds[dim];
};

/*typedef struct {
  int div;
  Result result;
  Bounds bounds[DIM];
} Region;*/

/*#if KEY == 13 && DIM == 2
#define RULE 13
#elif KEY == 1 && DIM == 3
#define RULE 13
#elif KEY == 9
#define RULE 9
#elif KEY == 7
#define RULE 7
#elif DIM == 2
#define RULE 9
#elif DIM == 3
#define RULE 11
#else
#define RULE 9
#endif

#if RULE == 13
#define NSETS 14
#elif RULE == 11
#define NSETS 13
#elif RULE == 9
#define NSETS 9
#define FEVAL (1 + 2*DIM + 2*DIM + 2*DIM + 2*DIM + 2*DIM*(DIM - 1) + 4*DIM*(DIM
- 1) + 4*DIM*(DIM - 1)*(DIM - 2)/3 + (1 << DIM)) #define
PERMUTATIONS_POS_ARRAY_SIZE (1+1*1 + 2*DIM*1 + 2*DIM*1 + 2*DIM*1 + 2*DIM*1 +
2*DIM*(DIM - 1)*2 + 4*DIM*(DIM - 1)*2 + 4*DIM*(DIM - 1)*(DIM - 2)*3/3 + DIM*(1
<< DIM)) #elif RULE == 7 #define NSETS 6 #endif

*/
#define NRULES 5
//__shared__ Region<6> sRegionPool[SM_REGION_POOL_SIZE];

//__shared__ GlobalBounds sBound[DIM];
extern __shared__ GlobalBounds sBound[];
__shared__ TYPE sdata[BLOCK_SIZE];

__shared__ TYPE* serror;
__shared__ size_t* serrorPos;

__shared__ TYPE ERR, RESULT;
__shared__ size_t gRegionPos[SM_REGION_POOL_SIZE / 2], gRegionPoolSize;
