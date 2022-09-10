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

    static __shared__ T shared[8]; // why was this set to 8?
    int lane = threadIdx.x % 32;   // 32 is for warp size
    int wid = threadIdx.x >> 5 /* threadIdx.x / 32  */;

    val = warpReduceSum(val);
    if (lane == 0) {
      shared[wid] = val;
    }
    __syncthreads(); // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;

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
  

  template <typename IntegT, typename T, int NDIM, bool debug = false>
  __device__ void
  computePermutation(IntegT* d_integrand,
                     int pIndex,
                     Bounds* b,
                     GlobalBounds sBound[],
                     //T* g,
                     //gpu::cudaArray<T, NDIM>& x,
                     T* sum,
                     const Structures<double>& constMem,
                     T range[],
                     T* jacobian,
                     double* generators,
                     //int FEVAL,
                     T* sdata,
					 quad::Func_Evals<NDIM>& fevals)
  {
	gpu::cudaArray<T, NDIM> x;
    for (int dim = 0; dim < NDIM; ++dim) {
      const T generator = __ldg(&generators[pagani::CuhreFuncEvalsPerRegion<NDIM>() * dim + pIndex]);
      x[dim] = sBound[dim].unScaledLower + ((.5 + generator) * b[dim].lower +
                                            (.5 - generator) * b[dim].upper) *
                                             range[dim];
    }
		
    const T fun = gpu::apply(*d_integrand, x) * (*jacobian);
    sdata[threadIdx.x] = fun; // target for reduction
	const int gIndex = __ldg(&constMem._gpuGenPermGIndex[pIndex]);
	
	if constexpr(debug){
	  //assert(fevals != nullptr);
	  fevals[blockIdx.x * pagani::CuhreFuncEvalsPerRegion<NDIM>() + pIndex].store(x, sBound, b);
	  fevals[blockIdx.x * pagani::CuhreFuncEvalsPerRegion<NDIM>() + pIndex].store(gpu::apply(*d_integrand, x), pIndex);
	  printf("feval:%e\n", gpu::apply(*d_integrand, x));
	}
			
	#pragma unroll 5
    for (int rul = 0; rul < NRULES; ++rul) {
      sum[rul] += fun * __ldg(&constMem._cRuleWt[gIndex * NRULES + rul]);
    }
  }



  template <typename IntegT, typename T, int NDIM>
  __device__ void
  verboseComputePermutation(IntegT* d_integrand,
                            int pIndex,
                            Bounds* b,
                            GlobalBounds sBound[],
                            // T* g,
                            gpu::cudaArray<T, NDIM>& x,
                            // T* sum,
                            // const Structures<double>& constMem,
                            T range[],
                            // T* jacobian,
                            double* generators,
                            int FEVAL,
                            // int iteration,
                            // T* sdata,
                            double* results,
                            double* funcEvals)
  {

    for (int dim = 0; dim < NDIM; ++dim) {
      x[dim] = 0;
    }

    // int gIndex = __ldg(&constMem._gpuGenPermGIndex[pIndex]);

    for (int dim = 0; dim < NDIM; ++dim) {
      T generator = __ldg(&generators[FEVAL * dim + pIndex]);
      x[dim] = sBound[dim].unScaledLower + ((.5 + generator) * b[dim].lower +
                                            (.5 - generator) * b[dim].upper) *
                                             range[dim];
    }

    T fun = gpu::apply(*d_integrand, x) /** (*jacobian)*/;
    results[pIndex] = fun; // target for reduction

    size_t index = pIndex * NDIM;
    for (int i = 0; i < NDIM; i++) {
      funcEvals[index + i] = x[i];
    }
    // we only care about func evaluations and results
    /*for (int rul = 0; rul < NRULES; ++rul) {
      sum[rul] += fun * __ldg(&constMem._cRuleWt[gIndex * NRULES + rul]);
    }*/
  }

  template <typename IntegT, typename T, int NDIM, int blockdim>
  __device__ void
  verboseSampleRegionBlock(IntegT* d_integrand,
                           int sIndex,
                           const Structures<double>& constMem,
                           int FEVAL,
                           int NSETS,
                           Region<NDIM> sRegionPool[],
                           GlobalBounds sBound[],
                           T* vol,
                           int* maxdim,
                           T range[],
                           T* jacobian,
                           double* generators,
                           double* results,
                           double* funcEvals)
  {
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[sIndex];
    __shared__ T sdata[blockdim];
    // T g[NDIM];
    gpu::cudaArray<T, NDIM> x;
    int perm = 0;

    T ratio =
      Sq(__ldg(&constMem._gpuG[2 * NDIM]) / __ldg(&constMem._gpuG[1 * NDIM]));
    int offset = 2 * NDIM;

    T sum[NRULES];
    Zap(sum);

    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * blockdim + threadIdx.x;

    if (pIndex < FEVAL) {
      verboseComputePermutation<IntegT, T, NDIM>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 // g,
                                                 x,
                                                 // sum,
                                                 // constMem,
                                                 range,
                                                 // jacobian,
                                                 generators,
                                                 FEVAL,
                                                 // iteration,
                                                 // sdata,
                                                 results,
                                                 funcEvals);
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
      verboseComputePermutation<IntegT, T, NDIM>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 // g,
                                                 x,
                                                 // sum,
                                                 // constMem,
                                                 range,
                                                 // jacobian,
                                                 generators,
                                                 FEVAL,
                                                 // iteration,
                                                 // sdata,
                                                 results,
                                                 funcEvals);
    }
    //__syncthreads();
    // Balance permutations
    pIndex = perm * blockdim + threadIdx.x;
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + threadIdx.x;
      verboseComputePermutation<IntegT, T, NDIM>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 // g,
                                                 x,
                                                 // sum,
                                                 // constMem,
                                                 range,
                                                 // jacobian,
                                                 generators,
                                                 FEVAL,
                                                 // iteration,
                                                 // sdata,
                                                 results,
                                                 funcEvals);
    }
    // __syncthreads();

    // for (int i = 0; i < NRULES; ++i) {
    // sum[i] = blockReduceSum /*computeReduce*/ (sum[i] /*, sdata*/);
    //__syncthreads();
    //}

    /*if (threadIdx.x == 0) {
      Result* r = &region->result;
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0;
        for (int s = 0; s < NSETS; ++s) {
          maxerr =
            max(maxerr,
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
    }*/
  }

  // BLOCK SIZE has to be atleast 4*DIM+1 for the first IF
  template <typename IntegT, typename T, int NDIM, int blockdim, bool debug>
  __device__ void
  SampleRegionBlock(IntegT* d_integrand,
                    //int sIndex,
                    const Structures<double>& constMem,
                    //int FEVAL,
                    //int NSETS,
                    Region<NDIM> sRegionPool[],
                    GlobalBounds sBound[],
                    T* vol,
                    int* maxdim,
                    T range[],
                    T* jacobian,
                    double* generators,
					quad::Func_Evals<NDIM>& fevals)
  {
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[0];
    __shared__ T sdata[blockdim];
    //T g[NDIM];
    //gpu::cudaArray<T, NDIM> x;
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
                                          //g,
                                          //x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          //FEVAL,
                                          sdata,
										  fevals);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
	  const T ratio = Sq(__ldg(&constMem._gpuG[2 * NDIM]) / __ldg(&constMem._gpuG[1 * NDIM]));
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
                                          //g,
                                          //x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          //FEVAL,
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
                                          //g,
                                          //x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          //FEVAL,
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


  __device__
  double scale_point(const double val, double low, double high){
	  return low + (high - low) * val;
  }
  
  __device__
  void
rebin(double rc, int nd, double r[], double xin[], double xi[])
{
    int i, k = 0;
    double dr = 0.0, xn = 0.0, xo = 0.0;

	//dr is the cummulative contribution
	
    for (i = 1; i < nd; i++) {
	  //rc is the average bin contribution
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
					 pagani::Curand_generator& rand_num_generator,
					 double& sum,
					 double& sq_sum,
					 double vol)
  {
	
	//random number generation for bin selection
	gpu::cudaArray<T, NDIM> x_random;
	constexpr size_t nbins = 10;
	int ndmx_p1 = nbins+1;
	__shared__ double xi[(nbins+1)*(NDIM+1)];
	__shared__ double d[(nbins+1) * (NDIM+1)];
	double dt[NDIM+1];   
	const int mxdim_p1 = NDIM + 1;
	double r[nbins+1];
	double xin[nbins+1];
	
	//size_t FEVAL = CuhreFuncEvalsPerRegion<NDIM>();
	//size_t num_passes = num_samples/FEVAL;
	
	if(threadIdx.x == 0){
		for (int j = 0; j <= NDIM; j++) {
			xi[j * ndmx_p1 + 1] = 1.0; // this index is the first for each bin for each dimension
			for(int bin = 0; bin <= nbins; ++bin){
				d[bin * (NDIM + 1) + j] = 0.;
				//printf("d[%i]:%f\n",bin * (NDIM + 1) + j, d[bin * (NDIM + 1) + j]);
			}
		}
		
		for (int i = 1; i <= nbins; i++)
		  r[i] = 1.0;
		for (int j = 1; j <= NDIM; j++) {
		  rebin(1. / nbins, nbins, r, xin, &xi[j * ndmx_p1]);
		}
		
		/*for(int i=0; i < (nbins+1)*(NDIM+1); ++i){
			if(threadIdx.x == 0 && blockIdx.x == 1)
				printf("xi[%i]:%f\n", i, xi[i]);
		}*/
	}
		
	__syncthreads();
		
	
	for(size_t pass = 0; pass < num_passes; ++pass){
			
		for(size_t sample = 0; sample < num_samples; ++sample){
			//use one thread to see what contribution gets marked
			int bins[NDIM+1];
			
			for (int dim = 1; dim <= NDIM; ++dim) {
				//draw random number
				const double random = (rand_num_generator)();
				
				//select bin with that random number
				const double probability = 1./static_cast<double>(nbins);
				const int bin = static_cast<int>(random/probability) + 1;
				bins[dim] = bin;

				
				const double bin_high = xi[(dim)*(nbins+1) + bin];
				const double bin_low = xi[(dim)*(nbins+1) + bin - 1];
				
				//if(threadIdx.x == 0 && blockIdx.x == 0)
				//	printf("dim %i bin %i bounds:%f,%f\n", dim, bin, bin_low, bin_high);
				
				//get the true bounds
				const double region_scaled_b_high = scale_point(bin_high, b[dim-1].lower, b[dim-1].upper);
				const double region_scaled_b_low = scale_point(bin_low, b[dim-1].lower, b[dim-1].upper);
				
				
				//scale a random point at those bounds
				
				double rand_point_in_bin = scale_point((rand_num_generator)(), region_scaled_b_low, region_scaled_b_high);
				
				if(threadIdx.x == 0 && blockIdx.x == 15)
					printf("sample %lu bin %i dim %i scaled bin bounds:%f,%f point:%e\n", 
						sample, bin, dim, region_scaled_b_low, region_scaled_b_high, rand_point_in_bin);
				x_random[dim-1] = rand_point_in_bin;
			}
			
			double f =  gpu::apply(*d_integrand, x_random) * (*jacobian);
			
			/*if(f != 0.  && blockIdx.x == 0){
				printf("\t\t feval tid:%i f:%e, bins:(%i,%i) contr indices:(%i,%i) %f, %f\n", 
				threadIdx.x, f, bins[1], bins[2], bins[1] * (NDIM+1) + 1, bins[2] * (NDIM+1) + 2, 
					x_random[0], x_random[1]);
			}*/
				
			sum += f;
			sq_sum += f*f;
			
			for(int dim = 1; dim <= NDIM; ++dim){
				atomicAdd(&d[bins[dim] * (NDIM+1) + dim], f * f);
				/*
				if(f != 0.)
					printf("tid %i dim:%i bin:%i adding %e (f:%e) to index d[%i]\n", 
						threadIdx.x, dim, bins[dim], pass_sq_sum, pass_sum, bins[dim] * (NDIM+1) + dim);*/
			}
		}
		
		
		
		double xo = 0.;
		double xn = 0.;
		double rc = 0.;
		
		__syncthreads();
		
		if(threadIdx.x == 0){
							
			if(blockIdx.x == 15){
				printf("---------------------------------\n");
				for(int dim=1; dim <= NDIM; ++dim){
					for(int bin = 1; bin <= nbins; ++bin){
					printf("dim :%i bin %i contribution:%e bounds:%f-%f\n", dim, bin, 
						d[(bin)*(NDIM+1) + dim], 
						scale_point(xi[(dim)*(nbins+1) + bin -1], b[dim-1].lower, b[dim-1].upper), 
						scale_point(xi[(dim)*(nbins+1) + bin], b[dim-1].lower, b[dim-1].upper)); 
					}
					printf("---------------------------------\n");
				}
				printf("========================\n");
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
							 (log(dt[j]) - log(d[i * mxdim_p1 + j])), 1.5);
						rc += r[i];
					}
					rebin( rc / nbins, nbins, r, xin, &xi[j * ndmx_p1]); // first bin of each dimension is at a diff index
				}
			}
			
			for (int j = 1; j <= NDIM; j++) {
				for (int i = 1; i <= nbins; i++)
					d[i * mxdim_p1 + j] = 0.0;
			}
			
			/*
			for(int i=0; i < (nbins+1)*(NDIM+1); ++i){
				if(threadIdx.x == 0 && blockIdx.x == 1)
				printf("xi[%i]:%f\n", i, xi[i]);
			}*/
			
		}
		__syncthreads();
		
	}
  }
	
	template <typename IntegT, typename T, int NDIM, int blockdim, bool debug = false>
	__device__ void
	Vegas_assisted_SampleRegionBlock(IntegT* d_integrand,
                    const Structures<double>& constMem,
                    Region<NDIM> sRegionPool[],
                    GlobalBounds sBound[],
                    T* vol,
                    int* maxdim,
                    T range[],
                    T* jacobian,
                    double* generators,
					quad::Func_Evals<NDIM>& fevals,
					unsigned int seed_init)
  {
	pagani::Curand_generator rand_num_generator(seed_init);  
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[0];
    __shared__ T sdata[blockdim];
	
	/*if(threadIdx.x == 0)
		printf("region %i bounds: (%f - %f) (%f - %f)  (%f - %f)  (%f - %f)  (%f - %f)\n", 
				blockIdx.x, 
				region->bounds[0].lower, region->bounds[0].upper,
				region->bounds[1].lower, region->bounds[1].upper,
				region->bounds[2].lower, region->bounds[2].upper,
				region->bounds[3].lower, region->bounds[3].upper,
				region->bounds[4].lower, region->bounds[4].upper);*/
	
	
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
                                          //g,
                                          //x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          //FEVAL,
                                          sdata,
										  fevals);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
	  const T ratio = Sq(__ldg(&constMem._gpuG[2 * NDIM]) / __ldg(&constMem._gpuG[1 * NDIM]));
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
                                          //g,
                                          //x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          //FEVAL,
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
                                          //g,
                                          //x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          //FEVAL,
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
	__syncthreads();
	
	size_t num_samples = 1000000;
	size_t num_passes = 2;
	double ran_sum = 0.;
	double sq_sum = 0.;
	
	
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
	
	sq_sum = sqrt(sq_sum * num_samples);
	sq_sum = (sq_sum - ran_sum) * (sq_sum + ran_sum);
	
	if(sq_sum <= 0.)
		sq_sum = 1.e-100;
	
	__syncthreads();
	ran_sum = blockReduceSum(ran_sum);
	sq_sum = blockReduceSum(sq_sum);
	__syncthreads();
	
	if(threadIdx.x == 0){
		Result* r = &region->result;	
		double mean = ran_sum / static_cast<double>(64* num_samples * num_passes);
		double var = sq_sum / static_cast<double>(64*num_passes * num_samples) - mean* mean;
		//printf("sum:%e (squared:%e) sq_sum:%e\n", ran_sum, ran_sum*ran_sum, sq_sum);
		printf("region %i mcubes:%e +- %e (sum:%e) nsamples:%i\n", blockIdx.x, vol[0]*mean, var, ran_sum, num_samples*num_passes*64);
		//printf("region %i pagani:%e +- %e mcubes:%e +- %e (ran_sum:%e)\n", blockIdx.x, r->avg, r->err, vol[0]*mean, var, ran_sum);
		r->avg = vol[0]*mean;
		r->err = sqrt(var);
	}
	
  }	

}

#endif
