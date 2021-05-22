#ifndef CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH

#include "../quad.h"
#include "../util/Volume.cuh"
#include "../util/cudaApply.cuh"
#include "../util/cudaArray.cuh"
#include "../util/cudaUtil.h"

namespace quad {

template <typename T>
__device__ T
Sq(T x)
{
    return x * x;
}

__device__
double warpReduceSum(double val) {
	val += __shfl_down_sync(0xffffffff, val, 16, 32);
	val += __shfl_down_sync(0xffffffff, val, 8, 32);
	val += __shfl_down_sync(0xffffffff, val, 4, 32);
	val += __shfl_down_sync(0xffffffff, val, 2, 32);
	val += __shfl_down_sync(0xffffffff, val, 1, 32);
	return val;
}

__device__
double blockReduceSum(double val) {

	static __shared__ double shared[8]; //why was this set to 8?
    //static __shared__ double shared[32]; 
	int lane = threadIdx.x % 32; //32 is for warp size
	int wid = threadIdx.x >> 5 /* threadIdx.x / 32  */;

	val = warpReduceSum(val);    
	if (lane == 0) {
        shared[wid] = val; 
    }
	__syncthreads();              // Wait for all partial reductions
    
	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x >> 5 ) ? shared[lane] : 0;
    
	if (wid == 0) 
        val = warpReduceSum(val); //Final reduce within first warp

	return val;
}
    
  template <typename T>
  __device__ T
  computeReduce(T sum, double* sdata)
  {
    sdata[threadIdx.x] = sum;

    __syncthreads();
    // is it wise to use shlf_down_sync, sdata[BLOCK_SIZE]
    // contiguous range pattern
    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        sdata[threadIdx.x] += sdata[threadIdx.x + offset];
      }
      __syncthreads();
    }
    return sdata[0];
  }

  template <typename IntegT, typename T, int NDIM>
  __device__ void
  ___computePermutation(IntegT* d_integrand,
                     int pIndex,
                     int perm,
                     Bounds* b,
                     T* g,
                     gpu::cudaArray<T, NDIM>& x,
                     T* sum,
                     const Structures<T>& constMem,
                     double range[],
                     double* jacobian,
                     double* generators,
                     int FEVAL,
                     int iteration,
                     double* sdata)
  {
    for (int dim = 0; dim < NDIM; ++dim) {
      x[dim] = 0;
    }
    
     int gIndex = __ldg(&constMem._gpuGenPermGIndex[pIndex]);
    
    for (int dim = 0; dim < NDIM; ++dim) {
      double generator = __ldg(&generators[FEVAL*dim + pIndex]);
      x[dim] = sBound[dim].unScaledLower +((.5 + generator) * b[dim].lower + (.5 - generator) * b[dim].upper)*range[dim];
    }
     
    T fun = gpu::apply(*d_integrand, x)* (*jacobian);
    sdata[pIndex] = fun; // target for reduction //this assumes each thread has a spot in sdata, but threads enter if they are larger than FEVAL, this would mean threads that arent' supposed to enter, enter this computation adn we collect garbage data
    
    for (int rul = 0; rul < NRULES; ++rul) {
      sum[rul] += fun * __ldg(&constMem._cRuleWt[gIndex * NRULES + rul]);
      /*if(rul == 0 && blockIdx.x == 0 && pIndex == 0 && iteration == 2)
      {
          printf("%i, %.15f x:%.15f, %.15f, %.15f, %.15f, %.15f, %.15f, %.15f, %.15f\n", threadIdx.x, fun * __ldg(&constMem._cRuleWt[gIndex * NRULES + rul]), x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
      }*/
    }
  }


 template <typename IntegT, typename T, int NDIM>
  __device__ void
  computePermutation(IntegT* d_integrand,
                     int pIndex,
                     Bounds* b,
                     T* g,
                     gpu::cudaArray<T, NDIM>& x,
                     T* sum,
                     const Structures<T>& constMem,
                     double range[],
                     double* jacobian,
                     double* generators,
                     int FEVAL,
                     int iteration,
                     double* sdata)
  {
 
    for (int dim = 0; dim < NDIM; ++dim) {
      x[dim] = 0;
    }
    
     int gIndex = __ldg(&constMem._gpuGenPermGIndex[pIndex]);
    
    for (int dim = 0; dim < NDIM; ++dim) {
      double generator = __ldg(&generators[FEVAL*dim + pIndex]);
      x[dim] = sBound[dim].unScaledLower +((.5 + generator) * b[dim].lower + (.5 - generator) * b[dim].upper)*range[dim];
    }
     
    T fun = gpu::apply(*d_integrand, x)* (*jacobian);
    sdata[threadIdx.x] = fun; // target for reduction
    
    for (int rul = 0; rul < NRULES; ++rul) {
      sum[rul] += fun * __ldg(&constMem._cRuleWt[gIndex * NRULES + rul]);
    }
  }

  template <typename IntegT, typename T, int NDIM, int blockdim>
  __device__ void
  ___SampleRegionBlock(IntegT* d_integrand,
                    int sIndex,
                    const Structures<T>& constMem,
                    int FEVAL,
                    int NSETS,
                    Region<NDIM> sRegionPool[], double* vol, int* maxdim,  double range[],  double* jacobian, double* generators, int iteration)
  {
    constexpr int feval = (1 + 2 * NDIM + 2 * NDIM + 2 * NDIM + 2 * NDIM +
                        2 * NDIM * (NDIM - 1) + 4 * NDIM * (NDIM - 1) +
                        4 * NDIM * (NDIM - 1) * (NDIM - 2) / 3 + (1 << NDIM));
    __shared__ double sdata[feval]; //unused memory at the end feval-4*NDIM-1 is unused, done this way to avoid branching when assigning
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[sIndex];
    T g[NDIM];
    gpu::cudaArray<double, NDIM> x;
    int perm = 0;
    
    T ratio = Sq(__ldg(&constMem._gpuG[2 * NDIM]) / __ldg(&constMem._gpuG[1 * NDIM]));
    int offset = 2 * NDIM;

    T sum[NRULES];
    Zap(sum);

    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * blockdim + threadIdx.x;

    //sdata[0] = 0.; 
    if (pIndex < FEVAL) {
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, perm, region->bounds, g, x, sum, constMem, range, jacobian, generators, FEVAL, iteration, sdata);
    }

    __syncthreads();

    for (perm = 1; perm < FEVAL / blockdim; ++perm) {
      int pIndex = perm * blockdim + threadIdx.x;
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, perm, region->bounds, g, x, sum, constMem, range, jacobian, generators, FEVAL, iteration, sdata);
    }
    __syncthreads(); //can we get rid of these, each thread writes to sdata[threadIdx.x], shouldn't be race condition
    // Balance permutations
    pIndex = perm * blockdim + threadIdx.x;
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + threadIdx.x;
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, perm, region->bounds, g, x, sum, constMem, range, jacobian, generators, FEVAL, iteration, sdata);
    }
    __syncthreads(); //warp reduction so is it ok to not use syncthreads
 
    for (int i = 0; i < NRULES; ++i) {
      sum[i] = blockReduceSum/*computeReduce*/(sum[i]);
    __syncthreads(); //do we really need this?
    }
       
    if (threadIdx.x == 0) {
      Result* r = &region->result;
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0;
        for (int s = 0; s < NSETS; ++s) {
          maxerr = max(maxerr,
                       fabs(sum[rul + 1] +
                            __ldg(&constMem._GPUScale[s * NRULES + rul]) * sum[rul]) *
                         __ldg(&constMem._GPUNorm[s * NRULES + rul]));
        }
        sum[rul] = maxerr;
      }

      r->avg = (*vol) * sum[0];
      r->err = (*vol) * ((errcoeff[0] * sum[1] <= sum[2] &&
                       errcoeff[0] * sum[2] <= sum[3]) ?
                        errcoeff[1] * sum[1] :
                        errcoeff[2] * max(max(sum[1], sum[2]), sum[3]));   
                        
        T* f = &sdata[0];   //should sdata be initialized?
        T* f1 = f;
        T base = *f1 * 2 * (1 - ratio);
        T maxdiff = 0;
        int bisectdim = *maxdim;
        for (int dim = 0; dim < NDIM; ++dim) {
            T* fp = f1 + 1;
            T* fm = fp + 1;
            T fourthdiff =
              fabs(base + ratio * (fp[0] + fm[0]) - (fp[offset] + fm[offset]));

            f1 = fm;
            if (fourthdiff > maxdiff) {
              maxdiff = fourthdiff;
              bisectdim = dim;
            }
        }
        r->bisectdim = bisectdim;                      
    }
  }

  // BLOCK SIZE has to be atleast 4*DIM+1 for the first IF
   template <typename IntegT, typename T, int NDIM, int blockdim>
  __device__ void
  SampleRegionBlock(IntegT* d_integrand,
                    int sIndex,
                    const Structures<T>& constMem,
                    int FEVAL,
                    int NSETS,
                    Region<NDIM> sRegionPool[], double* vol, int* maxdim,  double range[],  double* jacobian, double* generators, int iteration)
  {
    /*constexpr int feval = (1 + 2 * NDIM + 2 * NDIM + 2 * NDIM + 2 * NDIM +
                        2 * NDIM * (NDIM - 1) + 4 * NDIM * (NDIM - 1) +
                        4 * NDIM * (NDIM - 1) * (NDIM - 2) / 3 + (1 << NDIM));*/  
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[sIndex];
    __shared__ double sdata[blockdim];
    T g[NDIM];
    gpu::cudaArray<double, NDIM> x;
    int perm = 0;
    
    T ratio = Sq(__ldg(&constMem._gpuG[2 * NDIM]) / __ldg(&constMem._gpuG[1 * NDIM]));
    
    int offset = 2 * NDIM;

    T sum[NRULES];
    Zap(sum);

    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * blockdim + threadIdx.x;
    
    if (pIndex < FEVAL) {
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, region->bounds, g, x, sum, constMem, range, jacobian, generators, FEVAL, iteration, sdata);
    }
    
    __syncthreads();
   
    if (threadIdx.x == 0) {
      T* f = &sdata[0];
      Result* r = &region->result; 
      T* f1 = f;
      T base = *f1 * 2 * (1 - ratio);
      T maxdiff = 0;
      int bisectdim = *maxdim;
      for (int dim = 0; dim < NDIM; ++dim) {
        T* fp = f1 + 1;
        T* fm = fp + 1;
        T fourthdiff =
          fabs(base + ratio * (fp[0] + fm[0]) - (fp[offset] + fm[offset]));

        f1 = fm;
        if (fourthdiff > maxdiff) {
          maxdiff = fourthdiff;
          bisectdim = dim;
        }
      }

      r->bisectdim = bisectdim;
    }
    __syncthreads(); 

    for (perm = 1; perm < FEVAL / blockdim; ++perm) {
      int pIndex = perm * blockdim + threadIdx.x;
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, region->bounds, g, x, sum, constMem, range, jacobian, generators, FEVAL, iteration, sdata);
    }
    //__syncthreads(); 
    // Balance permutations
    pIndex = perm * blockdim + threadIdx.x;
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + threadIdx.x;
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, region->bounds, g, x, sum, constMem, range, jacobian, generators, FEVAL, iteration, sdata);
    }
   // __syncthreads(); 
    
    for (int i = 0; i < NRULES; ++i) {
      sum[i] = blockReduceSum/*computeReduce*/(sum[i]);
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      Result* r = &region->result;
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0;
        for (int s = 0; s < NSETS; ++s) {
          maxerr = max(maxerr,
                       fabs(sum[rul + 1] +
                            __ldg(&constMem._GPUScale[s * NRULES + rul]) * sum[rul]) *
                         __ldg(&constMem._GPUNorm[s * NRULES + rul]));
        }
        sum[rul] = maxerr;
      }

      r->avg = (*vol) * sum[0];
      r->err = (*vol) * ((errcoeff[0] * sum[1] <= sum[2] &&
                       errcoeff[0] * sum[2] <= sum[3]) ?
                        errcoeff[1] * sum[1] :
                        errcoeff[2] * max(max(sum[1], sum[2]), sum[3]));
    }
  }
    
}

#endif
