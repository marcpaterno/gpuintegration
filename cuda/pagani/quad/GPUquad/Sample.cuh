#ifndef CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH

#include <assert.h>
#include "cuda/pagani/quad/quad.h"
#include "cuda/pagani/quad/util/Volume.cuh"
#include "cuda/pagani/quad/util/cudaApply.cuh"
#include "cuda/pagani/quad/util/cudaArray.cuh"
#include "cuda/pagani/quad/util/cudaUtil.h"
#include "cuda/pagani/quad/GPUquad/Func_Eval.cuh"
#include <cmath>
#include <curand_kernel.h>

namespace pagani{
	
template<typename T>	
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

  __device__ T
  operator()()
  {
    return curand_uniform_double(&localState);
  }
};
}

namespace quad {
  template <typename T>
  __device__ T
  Sq(T x)
  {
    return x * x;
  }

  template <typename T>
  __device__ T
  warpReduceSum(T val)
  {
    val += __shfl_down_sync(0xffffffff, val, 16, 32);
    val += __shfl_down_sync(0xffffffff, val, 8, 32);
    val += __shfl_down_sync(0xffffffff, val, 4, 32);
    val += __shfl_down_sync(0xffffffff, val, 2, 32);
    val += __shfl_down_sync(0xffffffff, val, 1, 32);
    return val;
  }

  template <typename T>
  __device__ T
  blockReduceSum(T val)
  {
    static __shared__ T shared[8];     // why was this set to 8?
    const int lane = threadIdx.x % 32; // 32 is for warp size
    const int wid = threadIdx.x >> 5 /* threadIdx.x / 32  */;

    val = warpReduceSum(val);
    if (lane == 0) {
      shared[wid] = val;
    }
    __syncthreads(); // Wait for all partial reductions //I think it's safe to
                     // remove

    // read from shared memory only if that warp existed
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    __syncthreads();

    if (wid == 0)
      val = warpReduceSum(val); // Final reduce within first warp

    return val;
  }

  template <typename T>
  __device__ T
  computeReduce(T sum, T* sdata)
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

  template <typename IntegT, typename T, int NDIM, int debug = 0>
  __device__ void
  computePermutation(IntegT* d_integrand,
                     int pIndex,
                     Bounds* b,
                     GlobalBounds sBound[],
                     T* sum,
                     Structures<T>& constMem,
                     T range[],
                     T* jacobian,
                     T* generators,
                     T* sdata,
                     quad::Func_Evals<NDIM>& fevals)
  {
    gpu::cudaArray<T, NDIM> x;
    for (int dim = 0; dim < NDIM; ++dim) {
      const T generator = __ldg(
        &generators[pagani::CuhreFuncEvalsPerRegion<NDIM>() * dim + pIndex]);
      x[dim] = sBound[dim].unScaledLower + ((.5 + generator) * b[dim].lower +
                                            (.5 - generator) * b[dim].upper) *
                                             range[dim];
    }

    const T fun = gpu::apply(*d_integrand, x) * (*jacobian);
    sdata[threadIdx.x] = fun; // target for reduction
    const int gIndex = __ldg(&constMem.gpuGenPermGIndex[pIndex]);

    if constexpr (debug >= 2) {
      // assert(fevals != nullptr);
      fevals[blockIdx.x * pagani::CuhreFuncEvalsPerRegion<NDIM>() + pIndex]
        .store(x, sBound, b);
      fevals[blockIdx.x * pagani::CuhreFuncEvalsPerRegion<NDIM>() + pIndex]
        .store(gpu::apply(*d_integrand, x), pIndex);
    }

#pragma unroll 5
    for (int rul = 0; rul < NRULES; ++rul) {
      sum[rul] += fun * __ldg(&constMem.cRuleWt[gIndex * NRULES + rul]);
    }
  }


  // BLOCK SIZE has to be atleast 4*DIM+1 for the first IF
  template <typename IntegT, typename T, int NDIM, int blockdim, int debug>
  __device__ void
  SampleRegionBlock(IntegT* d_integrand,
                    Structures<T>& constMem,
                    Region<NDIM> sRegionPool[],
                    GlobalBounds sBound[],
                    T* vol,
                    int* maxdim,
                    T range[],
                    T* jacobian,
                    T* generators,
					quad::Func_Evals<NDIM>& fevals)
  {
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[0];
    __shared__ T sdata[blockdim];
    int perm = 0;
    constexpr int offset = 2 * NDIM;

    T sum[NRULES];
    Zap(sum);

    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * blockdim + threadIdx.x;
    constexpr int FEVAL = pagani::CuhreFuncEvalsPerRegion<NDIM>();
    if (pIndex < FEVAL) {
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 sum,
                                                 constMem,
                                                 range,
                                                 jacobian,
                                                 generators,
                                                 sdata,
                                                 fevals);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      const T ratio =
        Sq(__ldg(&constMem.gpuG[2 * NDIM]) / __ldg(&constMem.gpuG[1 * NDIM]));
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
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 sum,
                                                 constMem,
                                                 range,
                                                 jacobian,
                                                 generators,
                                                 sdata,
                                                 fevals);
    }
    //__syncthreads();
    // Balance permutations
    pIndex = perm * blockdim + threadIdx.x;
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + threadIdx.x;
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 sum,
                                                 constMem,
                                                 range,
                                                 jacobian,
                                                 generators,
                                                 sdata,
                                                 fevals);
    }

    for (int i = 0; i < NRULES; ++i) {
      sum[i] = blockReduceSum(sum[i]);
      //__syncthreads();
    }

    if (threadIdx.x == 0) {
      Result* r = &region->result; // ptr to shared Mem

#pragma unroll 4
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0.;

        constexpr int NSETS = 9;
#pragma unroll 9
        for (int s = 0; s < NSETS; ++s) {
          maxerr = max(maxerr,
                       fabs(sum[rul + 1] +
                            constMem.GPUScale[s * NRULES + rul] * sum[rul]) *
                         (constMem.GPUNorm[s * NRULES + rul]));
        }
        sum[rul] = maxerr;
      }

      r->avg = (*vol) * sum[0];
	  const T errcoeff[3] = {5., 1., 5.};
	  //branching twice for each thread 0
      r->err = (*vol) * ((errcoeff[0] * sum[1] <= sum[2] &&
                          errcoeff[0] * sum[2] <= sum[3]) ?
                           errcoeff[1] * sum[1] :
                           errcoeff[2] * max(max(sum[1], sum[2]), sum[3]));
    }
  }

  template<typename T>
  __device__
  T scale_point(const T val, T low, T high){
	  return low + (high - low) * val;
  }
  
  template<typename T>
  __device__
  void
rebin(T rc, int nd, T r[], T xin[], T xi[])
{
    int i, k = 0;
    T dr = 0.0, xn = 0.0, xo = 0.0;

    // dr is the cummulative contribution

    for (i = 1; i < nd; i++) {
      // rc is the average bin contribution
      while (rc > dr) {

        dr += r[++k];
      }

      if (k > 1)
        xo = xi[k - 1];
      xn = xi[k];
      dr -= rc;

      xin[i] = xn - (xn - xo) * dr / r[k];
    }

    for (i = 1; i < nd; i++)
      xi[i] = xin[i];
    xi[nd] = 1.0;
  }

  template <typename IntegT, typename T, int NDIM>
  __device__ void
  Vegas_assisted_computePermutation(IntegT* d_integrand,
					 size_t num_samples, 
					 size_t num_passes,
                     Bounds* b,
                     GlobalBounds sBound[],
                     T range[],
                     T* jacobian,
					 pagani::Curand_generator<T>& rand_num_generator,
					 T& sum,
					 T& sq_sum,
					 T vol)
  {
	
	//random number generation for bin selection
	gpu::cudaArray<T, NDIM> x_random;
	constexpr size_t nbins = 100;
	int ndmx_p1 = nbins+1;
	__shared__ T xi[(nbins+1)*(NDIM+1)];
	__shared__ T d[(nbins+1) * (NDIM+1)];
	T dt[NDIM+1];   
	const int mxdim_p1 = NDIM + 1;
	T r[nbins+1];
	T xin[nbins+1];
	
	//size_t FEVAL = CuhreFuncEvalsPerRegion<NDIM>();
	//size_t num_passes = num_samples/FEVAL;
	
	if(threadIdx.x == 0){
		
		for (int j = 0; j <= NDIM; j++) {
			xi[j * ndmx_p1 + 1] = 1.0; // this index is the first for each bin for each dimension
			
			for(int bin = 0; bin <= nbins; ++bin){
				d[bin * (NDIM + 1) + j] = 0.;
			}
		}
		
		for (int bin = 1; bin <= nbins; bin++)
		  r[bin] = 1.0;
	  
		for (int dim = 1; dim <= NDIM; dim++) {
		  rebin(1. / nbins, nbins, r, xin, &xi[dim * ndmx_p1]);
		}
		
		for (int dim = 1; dim <= NDIM; dim++){
			for(int bin=1; bin <= nbins; ++bin){
					
				size_t xi_i = (dim)*(nbins+1) + bin;
				size_t d_i = bin * mxdim_p1 + dim;
					
				if(threadIdx.x == 0 && blockIdx.x == 9)
					printf("xi %i, %i, %i, %i, %e, %e\n", 0, blockIdx.x, dim, bin, xi[xi_i], d[d_i]);
			}
		}		
	}
		
	__syncthreads();
		
	for(size_t pass = 0; pass < num_passes; ++pass){
		
		T local_sq_sum = 0;
		T local_sum = 0;
		
		for(size_t sample = 0; sample < num_samples; ++sample){
			//use one thread to see what contribution gets marked
			int bins[NDIM+1];
			
			T wgt = 1.;
			
			for (int dim = 1; dim <= NDIM; ++dim) {
				//draw random number
				const T random = (rand_num_generator)();
				
				//select bin with that random number
				const T probability = 1./static_cast<T>(nbins);
				const int bin = static_cast<int>(random/probability) + 1;
				bins[dim] = bin;
				
				const T bin_high = xi[(dim)*(nbins+1) + bin];
				const T bin_low = xi[(dim)*(nbins+1) + bin - 1];
						
				//get the true bounds
				const T region_scaled_b_high = scale_point(bin_high, b[dim-1].lower, b[dim-1].upper);
				const T region_scaled_b_low = scale_point(bin_low, b[dim-1].lower, b[dim-1].upper);
				//scale a random point at those bounds
				
				T rand_point_in_bin = scale_point((rand_num_generator)(), region_scaled_b_low, region_scaled_b_high);
				x_random[dim-1] = rand_point_in_bin;
				wgt *= nbins * (bin_high - bin_low);
			}
			
			T calls = 64. * num_passes * num_samples;
			T f =  gpu::apply(*d_integrand, x_random) * (*jacobian) * wgt/calls;
			
			local_sum += f;
			local_sq_sum +=  f*f;
			
			for(int dim = 1; dim <= NDIM; ++dim){
				atomicAdd(&d[bins[dim] * mxdim_p1 + dim], f * f);
			}
		}
		
		local_sq_sum += sqrt(local_sq_sum * num_samples);
		local_sq_sum += (local_sq_sum - local_sum) * (local_sq_sum + local_sum);
		
		if(threadIdx.x == 0 && blockIdx.x == 0)
			printf("per-pass %e / %e / %e\n", local_sum, local_sum*local_sum, local_sq_sum);
		
		if(local_sq_sum <= 0.)
			local_sq_sum = 1.e-100;
		
		sum += local_sum;
		sq_sum += local_sq_sum;
		
		T xo = 0.;
		T xn = 0.;
		T rc = 0.;
		
		__syncthreads();
		
		if(threadIdx.x == 0){
									
		for (int dim = 1; dim <= NDIM; dim++){
			for(int bin = 1; bin <= nbins; ++bin){

				size_t xi_i = (dim)*(nbins+1) + bin;
				size_t d_i = bin * mxdim_p1 + dim;
					
				if(threadIdx.x == 0 && blockIdx.x == 9)
					printf("xi %lu, %i, %i, %i, %e, %e\n", pass + 1, blockIdx.x, dim, bin, xi[xi_i], d[d_i]);
			}
		}		
			
			
		for (int j = 1; j <= NDIM; j++) {
				
			//avg contribution from first two bins 
			xo = d[1 * mxdim_p1 + j]; // bin 1 of dim j
			xn = d[2 * mxdim_p1 + j]; // bin 2 of dim j
			d[1 * mxdim_p1 + j] = (xo + xn) / 2.0;
				
			dt[j] = d[1 * mxdim_p1 + j]; // set dt sum to contribution of bin 1
				
			for (int i = 2; i < nbins; i++) {
					
				rc = xo + xn;//here xn is contr of previous bin
				xo = xn;
					
				xn = d[(i + 1) * mxdim_p1 + j];//here takes contr of next bin
				d[i * mxdim_p1 + j] = (rc + xn) / 3.0; //avg of three bins
				dt[j] += d[i * mxdim_p1 + j]; //running sum of all contributions
			}

			d[nbins * mxdim_p1 + j] = (xo + xn) / 2.0; // do bin nd last
			dt[j] += d[nbins * mxdim_p1 + j];
		}
			
		for (int j = 1; j <= NDIM; j++) {
			if (dt[j] > 0.0) { // enter if there is any contribution only
				rc = 0.0;
				for (int i = 1; i <= nbins; i++) {
					r[i] = pow((1.0 - d[i * mxdim_p1 + j] / dt[j]) /
						(log(dt[j]) - log(d[i * mxdim_p1 + j])), .5);
					rc += r[i];
				}
				
				
				rebin( rc / nbins, nbins, r, xin, &xi[j * ndmx_p1]); // first bin of each dimension is at a diff index
			}
		}
			
		for (int j = 1; j <= NDIM; j++) {
			for (int i = 1; i <= nbins; i++)
				d[i * mxdim_p1 + j] = 0.0;
		}
			
			
			
		}
		__syncthreads();
		
	}
  }

	template <typename IntegT, typename T, int NDIM, int blockdim, bool debug = false>
	__device__ void
	Vegas_assisted_SampleRegionBlock(IntegT* d_integrand,
                    Structures<T>& constMem,
                    Region<NDIM> sRegionPool[],
                    GlobalBounds sBound[],
                    T* vol,
                    int* maxdim,
                    T range[],
                    T* jacobian,
                    T* generators,
					quad::Func_Evals<NDIM>& fevals,
					unsigned int seed_init)
  {
	pagani::Curand_generator<T> rand_num_generator(seed_init);  
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[0];
    __shared__ T sdata[blockdim];

    int perm = 0;
    constexpr int offset = 2 * NDIM;

    T sum[NRULES];
    Zap(sum);

    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * blockdim + threadIdx.x;
    constexpr int FEVAL = pagani::CuhreFuncEvalsPerRegion<NDIM>();
    if (pIndex < FEVAL) {
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 sum,
                                                 constMem,
                                                 range,
                                                 jacobian,
                                                 generators,
                                                 sdata,
                                                 fevals);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      const T ratio =
        Sq(__ldg(&constMem.gpuG[2 * NDIM]) / __ldg(&constMem.gpuG[1 * NDIM]));
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
      computePermutation<IntegT, T, NDIM>(d_integrand,
                                          pIndex,
                                          region->bounds,
                                          sBound,
                                          // g,
                                          // x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          // FEVAL,
                                          sdata,
                                          fevals);
    }
    //__syncthreads();
    // Balance permutations
    pIndex = perm * blockdim + threadIdx.x;
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + threadIdx.x;
      computePermutation<IntegT, T, NDIM>(d_integrand,
                                          pIndex,
                                          region->bounds,
                                          sBound,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          sdata,
                                          fevals);
    }
    // __syncthreads();

    for (int i = 0; i < NRULES; ++i) {
      sum[i] = blockReduceSum(sum[i]);
      //__syncthreads();
    }

    if (threadIdx.x == 0) {
      Result* r = &region->result;

#pragma unroll 4
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0;

        constexpr int NSETS = 9;
#pragma unroll 9
        for (int s = 0; s < NSETS; ++s) {
          maxerr =
            max(maxerr,
                fabs(sum[rul + 1] +
                     __ldg(&constMem.GPUScale[s * NRULES + rul]) * sum[rul]) *
                  __ldg(&constMem.GPUNorm[s * NRULES + rul]));
        }
        sum[rul] = maxerr;
      }

      r->avg = (*vol) * sum[0];

	  const T errcoeff[3] = {5, 1, 5};
      r->err = (*vol) * ((errcoeff[0] * sum[1] <= sum[2] &&
                          errcoeff[0] * sum[2] <= sum[3]) ?
                           errcoeff[1] * sum[1] :
                           errcoeff[2] * max(max(sum[1], sum[2]), sum[3]));
	}
	__syncthreads();
	
	size_t num_samples = 100;
	size_t num_passes = 10;
	T ran_sum = 0.;
	T sq_sum = 0.;
	
	
	Vegas_assisted_computePermutation<IntegT, T, NDIM>(d_integrand,
					 num_samples,
					 num_passes,
                     region->bounds,
                     sBound,
                     range,
                     jacobian,
					 rand_num_generator,
					 ran_sum,
					 sq_sum,
					 vol[0]);
	
	__syncthreads();
	ran_sum = blockReduceSum(ran_sum);
	sq_sum = blockReduceSum(sq_sum);
	__syncthreads();
	
	if(threadIdx.x == 0){
		Result* r = &region->result;	
		//double mean = ran_sum / static_cast<double>(64* num_samples * num_passes);
		//double var = sq_sum / static_cast<double>(64*num_passes * num_samples) - mean* mean;
		//printf("region %i mcubes:%e +- %e (sum:%e) nsamples:%i\n", blockIdx.x, vol[0]*mean, var, ran_sum, num_samples*num_passes*64);
		
		T dxg = 1.0 / (num_passes*num_samples*64);
		T dv2g, i;
		T calls = num_passes*num_samples*64;
		for (dv2g = 1, i = 1; i <= NDIM; i++)
			dv2g *= dxg;
		dv2g = (calls * dv2g * calls * dv2g) / num_samples / num_samples / (num_samples - 1.0);
		
		
		printf("region %i pagani:%e +- %e mcubes:%e +- %e (ran_sum:%e)\n", 
			blockIdx.x, 
			r->avg, r->err, 
			ran_sum, sqrt(sq_sum * dv2g),
			//vol[0]*mean, sqrt(sq_sum * dv2g), 
			ran_sum);
		
		r->avg = vol[0] * ran_sum;
		r->err = sqrt(sq_sum * dv2g);
	}
	
  }	
}

#endif
