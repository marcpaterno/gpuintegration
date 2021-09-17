#ifndef VEGAS_VEGAS_T_CUH
#define VEGAS_VEGAS_T_CUH

/*

code works for gaussian and sin using switch statement. device pointerr/template
slow down the code by 2x

chunksize needs to be tuned based on the ncalls. For now hardwired using a
switch statement


nvcc -O2 -DCUSTOM -o vegas vegasT.cu -arch=sm_70
OR
nvcc -O2 -DCURAND -o vegas vegasT.cu -arch=sm_70

example run command

nvprof ./vegas 0 6 0.0  10.0  1.0E+09  10 0 0 0

nvprof  ./vegas 1 9 -1.0  1.0  1.0E+07 15 10 10

nvprof ./vegas 2 2 -1.0 1.0  1.0E+09 1 0 0

Last three arguments are: total iterations, iteration

*/

#define OUTFILEVAR 0

#include "cudaPagani/quad/util/Volume.cuh"
#include "cudaPagani/quad/util/cudaApply.cuh"
#include "cudaPagani/quad/util/cudaArray.cuh"
#include "cudaPagani/quad/quad.h"
#include "vegas/util/func.cuh"
#include "vegas/util/util.cuh"
#include "vegas/util/vegas_utils.cuh"
#include "vegas/util/verbose_utils.cuh"
#include <chrono>
#include <ctime>
#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>

#define TINY 1.0e-30
#define WARP_SIZE 32
#define BLOCK_DIM_X 128
//#define ALPH 1.5 // commented out by Ioannis in order to match python vegas default of .5
//#define ALPH 0.5
//#define NDMX 500
//#define MXDIM 20

//#define NDMX1 NDMX + 1
//#define MXDIM1 MXDIM + 1
//#define PI 3.14159265358979323846



class Internal_Vegas_Params{
        static constexpr int NDMX = 500;
        static constexpr int MXDIM = 20;
        static constexpr double ALPH = 0.5;
    
    public:
        
        __host__ __device__ static constexpr int get_NDMX(){return NDMX;}
   
        __host__ __device__ static constexpr int get_NDMX_p1(){return NDMX+1;}
       
        __host__ __device__ static constexpr  double get_ALPH(){return ALPH;}
       
        __host__ __device__ static constexpr  int get_MXDIM(){return MXDIM;}
        
        constexpr __host__ __device__ static int get_MXDIM_p1(){return MXDIM+1;}
};

#define IMAX(a, b)                                                             \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

#define IMIN(a, b)                                                             \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a < _b ? _a : _b;                                                         \
  })

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n",                                     \
             __FILE__,                                                         \
             __LINE__,                                                         \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

//int verbosity = 0;

namespace cuda_mcubes{

using MilliSeconds =
  std::chrono::duration<double, std::chrono::milliseconds::period>;

__inline__ __device__ double
warpReduceSum(double val)
{
  val += __shfl_down_sync(0xffffffff, val, 16, WARP_SIZE);
  val += __shfl_down_sync(0xffffffff, val, 8, WARP_SIZE);
  val += __shfl_down_sync(0xffffffff, val, 4, WARP_SIZE);
  val += __shfl_down_sync(0xffffffff, val, 2, WARP_SIZE);
  val += __shfl_down_sync(0xffffffff, val, 1, WARP_SIZE);
  return val;
}

__inline__ __device__ double
blockReduceSum(double val)
{

  static __shared__ double shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val); // Each warp performs partial reduction

  if (lane == 0)
    shared[wid] = val; // Write reduced value to shared memory

  __syncthreads(); // Wait for all partial reductions

  // read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid == 0)
    val = warpReduceSum(val); // Final reduce within first warp
  __syncthreads(); //added by Ioannis due to cuda-memcheck racecheck reporting race between read/write
  return val;
}

__inline__ __device__ void
get_indx(uint32_t ms, uint32_t* da, int ND, int NINTV)
{
    //called like :    get_indx(m * chunkSize, &kg[1], ndim, ng);
  uint32_t dp[Internal_Vegas_Params::get_MXDIM()];
  uint32_t j, t0, t1;
  uint32_t m = ms;
  dp[0] = 1;
  dp[1] = NINTV;

  for (j = 0; j < ND - 2; j++) {
    dp[j + 2] = dp[j + 1] * NINTV;
  }
  //
  
  //if(threadIdx.x == 0 && blockIdx.x == 0)
  //    printf("%u, %u, %u, %u, %u, %u\n", dp[1], dp[2], dp[3], dp[4], dp[5], dp[6]);
  
  for (j = 0; j < ND; j++) {
    t0 = dp[ND - j - 1];
    t1 = m / t0;
    
    //if(threadIdx.x == 0 && blockIdx.x == 0)
    //    printf("t0:%u t1:%u\n", t0, t1);
    da[j] = 1 + t1;
    //if(threadIdx.x == 0 && blockIdx.x == 0)
    //    printf("dp[%i]:%u\n", j, t1+1);
   // if(threadIdx.x == 0 && blockIdx.x == 0)
    //    printf("m goes from %u to %u\n", m, m-t1*t0);
    m = m - t1 * t0;
    
  }
  
  /*if(blockIdx.x > 8554)
    printf("Block %i thread %u get_indx ms:%u found index\n", blockIdx.x, threadIdx.x, ms);*/
}


template <int ndim>
__inline__ __device__
void Setup_Integrand_Eval(curandState* localState, 
                            double xnd, double dxg, 
                            const double* const xi, const double* const regn, const double* const dx, 
                            const uint32_t* const kg, 
                            int* const ia, 
                            double* const x, 
                            double& wgt)
{
    constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
    /*
        input: 
            dim: dimension being processed
            integer kg array that holds the interval being processed
            xi: right boundary coordinate of each bin
        output:    
            integer ia array, that holds the bin being processed
            double x array that holds the points at which the integrand will be evaluated at each dimension
            double weight
    */
    
    for (int j = 1; j <= ndim; j++) {
            
#ifdef CURAND
          const double ran00 = curand_uniform_double(localState);
#endif

          const double xn = (kg[j] - ran00) * dxg + 1.0;
          double rc = 0., xo = 0.;
          ia[j] = IMAX(IMIN((int)(xn), ndmx), 1);
	  
          if (ia[j] > 1) {
            xo = xi[j * ndmx + 1 + ia[j]] - xi[j * ndmx + 1 + ia[j] - 1];
            rc = xi[j * ndmx + 1 + ia[j] - 1] + (xn - ia[j]) * xo;
          } else {
            xo = xi[j * ndmx + 1 + ia[j]];
            rc = (xn - ia[j]) * xo;
          }

          x[j] = regn[j] + rc * dx[j];
          wgt *= xo * xnd; //xnd is number of bins, xo is the length of the bin, xjac is 1/num_calls
        }
}

template <typename IntegT, int ndim>
__device__ void
Process_npg_samples(IntegT* d_integrand, 
                        int npg, 
                        double xnd, double xjac, 
                        curandState* localState, 
                        double dxg, 
                        const double* const regn, const double* const dx, const double* const xi, 
                        const uint32_t* const kg, 
                        int* const ia, 
                        double* const x, 
                        double& wgt, 
                        double* d, 
                        double& fb, 
                        double& f2b){
    
      for (int k = 1; k <= npg; k++) {
          
        double wgt = xjac;
        Setup_Integrand_Eval<ndim>(localState, xnd, dxg, xi, regn, dx, kg,  ia, x, wgt);
        
        gpu::cudaArray<double, ndim> xx;             
        for (int i = 0; i < ndim; i++) {
          xx[i] = x[i + 1];                       
        }

        double tmp = gpu::apply(*d_integrand, xx);
        double f = wgt * tmp;     
        double f2 = f * f;

        fb += f;
        f2b += f2;

#pragma unroll 2
        for (int j = 1; j <= ndim; j++) {
          atomicAdd(&d[ia[j] * Internal_Vegas_Params::get_MXDIM() + 1 + j], fabs(f));
        }

      }
}    

template <typename IntegT, int ndim>
__inline__ __device__
void Process_chunks(IntegT* d_integrand, 
                int chunkSize, int ng, int npg, 
                curandState* localState, 
                double dxg, double xnd, double xjac, 
                const double* const regn, const double* const dx, const double* const xi, 
                uint32_t* const kg, 
                int* const ia,  
                double* const x, 
                double& wgt, 
                double* d, 
                double& fbg, 
                double& f2bg){
    
    for (int t = 0; t < chunkSize; t++) {
      double fb = 0., f2b = 0.0;
      
      Process_npg_samples<IntegT, ndim>(d_integrand, npg, xnd, xjac, localState, dxg, regn, dx, xi, kg, ia, x, wgt, d, fb, f2b);

      f2b = sqrt(f2b * npg); //some times f2b becomes exactly zero, other times its equal to fb
      f2b = (f2b - fb) * (f2b + fb);
      
      if (f2b <= 0.0){ 
        f2b=TINY;
      }

      fbg += fb;
      f2bg += f2b;
      
      for (int k = ndim; k >= 1; k--) {
        kg[k] %= ng;
        if (++kg[k] != 1)
          break;
      }
    } 
    
}

template <typename IntegT, int ndim>
__global__ void
vegas_kernel(IntegT* d_integrand,
             int ng,
             int npg,
             double xjac,
             double dxg,
             double* result_dev,
             double xnd,
             double* xi,
             double* d,
             double* dx,
             double* regn,
             int ncubes,
             int iter,
             double sc,
             double sci,
             double ing,
             int chunkSize,
             uint32_t totalNumThreads,
             int LastChunk,
             int fcode,
             unsigned int seed_init/*,
             double* evals,
             double* evalPoints,
			 int* intervals*/)
{

  constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
  //constexpr int ndmx_p1 = Internal_Vegas_Params::get_NDMX_p1();
  constexpr int mxdim = Internal_Vegas_Params::get_MXDIM();
  constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();

  unsigned long long seed;
   //seed_init *= (iter) * ncubes;
  // seed_init = clock64();
  
  uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  double /*fb, f2b,*/ wgt/*, xn, xo, rc, f, f2*/;
  uint32_t kg[mxdim_p1];
  int ia[mxdim_p1];
  double x[mxdim_p1];
  //int k, j;
  double fbg = 0., f2bg = 0.;
  
  if (m < totalNumThreads) {
      
    //int orig_chunkSize = chunkSize;
    if (m == totalNumThreads - 1)
      chunkSize = LastChunk + 1;
     
    curandState localState;
    curand_init(seed_init, blockIdx.x, threadIdx.x, &localState);

    get_indx(m * chunkSize, &kg[1], ndim, ng);
    
    Process_chunks<IntegT, ndim>(d_integrand, chunkSize, ng, npg, &localState, dxg, xnd, xjac, regn, dx, xi, kg, ia, x, wgt, d, fbg, f2bg);
    /*for (int t = 0; t < chunkSize; t++) {
      fb = f2b = 0.0;
      
      for (k = 1; k <= npg; k++) {
        wgt = xjac;
	
        for (j = 1; j <= ndim; j++) {
            
#ifdef CUSTOM
          temp = a * seed + c;
          seed = temp & (p - 1);
          ran00 = (double)seed / (double)p;
#endif
#ifdef CURAND
          const double ran00 = curand_uniform_double(&localState);
#endif

          xn = (kg[j] - ran00) * dxg + 1.0;
          ia[j] = IMAX(IMIN((int)(xn), ndmx), 1);
	  
          if (ia[j] > 1) {
            xo = xi[j * ndmx + 1 + ia[j]] - xi[j * ndmx + 1 + ia[j] - 1];
            rc = xi[j * ndmx + 1 + ia[j] - 1] + (xn - ia[j]) * xo;
          } else {
            xo = xi[j * ndmx + 1 + ia[j]];
            rc = (xn - ia[j]) * xo;
          }

          x[j] = regn[j] + rc * dx[j];
          wgt *= xo * xnd; //xnd is number of bins, xo is the length of the bin, xjac is 1/num_calls
        }

        gpu::cudaArray<double, ndim> xx;//don't need to create it each time
        //size_t index = m*orig_chunkSize*npg + t*npg + (k-1);
        
        for (int i = 0; i < ndim; i++) {
          xx[i] = x[i + 1];
                    
          //evalPoints[index*ndim + i] = xx[i];
	  //intervals[index*ndim+ i] = kg[i+1];      
        }

	double tmp;
        tmp = gpu::apply(*d_integrand, xx);

        f = wgt * tmp; 
        
        
        //evals[index] = f;
        //computing S^(1)
        f2 = f * f;    //computing S^(2) //square temp*xo*xnd? (we leave xjac which is 1/num_calls outside, it won't get square so we don't have to make up for it)

        fb += f;
        f2b += f2;

#pragma unroll 2

        for (j = 1; j <= ndim; j++) {
          atomicAdd(&d[ia[j] * mxdim + 1 + j], fabs(f));
        }

      } // end of npg loop
      
      f2b = sqrt(f2b * npg); //some times f2b becomes exactly zero, other times its equal to fb
      f2b = (f2b - fb) * (f2b + fb);
      
      if (f2b <= 0.0){ 
        f2b=TINY;
      }

      fbg += fb;
      f2bg += f2b;
      
      for (int k = ndim; k >= 1; k--) {
        kg[k] %= ng;
        if (++kg[k] != 1)
          break;
      }
      //if(blockIdx.x > 9000)
      //  printf("thread %i Done with chunk %i\n", m, t); 
    } // end of chunk for loop
    */

    fbg = blockReduceSum(fbg);
    f2bg = blockReduceSum(f2bg);
      
    if (tx == 0) {
      //printf("Block %i done\n", blockIdx.x);
      atomicAdd(&result_dev[0], fbg);
      atomicAdd(&result_dev[1], f2bg);
    }
  } // end of subcube if
}

template <typename IntegT, int ndim>
__global__ void
vegas_kernelF(IntegT* d_integrand,
              int ng,
              int npg,
              double xjac,
              double dxg,
              double* result_dev,
              double xnd,
              double* xi,
              double* d,
              double* dx,
              double* regn,
              int ncubes,
              int iter,
              double sc,
              double sci,
              double ing,   //not needed?
              int chunkSize,
              uint32_t totalNumThreads,
              int LastChunk,
              int fcode,    //not needed
              unsigned int seed_init)
{

  constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
  //constexpr int ndmx_p1 = Internal_Vegas_Params::get_NDMX_p1();
  constexpr int mxdim = Internal_Vegas_Params::get_MXDIM();
  constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();

#ifdef CUSTOM
  uint64_t temp;
  uint32_t a = 1103515245;
  uint32_t c = 12345;
  uint32_t one, expi;
  one = 1;
  expi = 31;
  uint32_t p = one << expi;
#endif

  unsigned long long seed;
  // seed_init = (iter) * ncubes;

  uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  double fb, f2b, wgt, xn, xo, rc, f, f2, ran00;
  uint32_t kg[mxdim_p1];
  int iaj;
  double x[mxdim_p1];
  int k, j;
  double fbg, f2bg;


  if (m < totalNumThreads) {
      
    
    if (m == totalNumThreads - 1)
      chunkSize = LastChunk + 1;
  
    //seed = seed_init + m * chunkSize;
#ifdef CURAND
    
    curandState localState;
    curand_init(seed_init, blockIdx.x, threadIdx.x, &localState);
    //if(m == 0)
    //  printf("vegas_kernelF seed_init:%u\n", seed_init);
#endif

    fbg = f2bg = 0.0;
    get_indx(m * chunkSize, &kg[1], ndim, ng);
       
    for (int t = 0; t < chunkSize; t++) {
      fb = f2b = 0.0;

      for (k = 1; k <= npg; k++) {
        wgt = xjac;

        for (j = 1; j <= ndim; j++) {
#ifdef CUSTOM
          temp = a * seed + c;
          seed = temp & (p - 1);
          ran00 = (double)seed / (double)p;
#endif
#ifdef CURAND
          ran00 = curand_uniform_double(&localState);
	  
#endif
        
          xn = (kg[j] - ran00) * dxg + 1.0;
          iaj = IMAX(IMIN((int)(xn), ndmx), 1);
	  

	  
	  if (iaj > 1) {
            xo = xi[j * ndmx + 1 + iaj] - xi[j * ndmx + 1 + iaj - 1];
            rc = xi[j * ndmx + 1 + iaj - 1] + (xn - iaj) * xo;
          } else {
            xo = xi[j * ndmx + 1 + iaj];
            rc = (xn - iaj) * xo;
          }

          x[j] = regn[j] + rc * dx[j];

          wgt *= xo * xnd;
        }
        
        double tmp;
        gpu::cudaArray<double, ndim> xx;
        for (int i = 0; i < ndim; i++) {
          xx[i] = x[i + 1];
        }
           
        tmp = gpu::apply(*d_integrand, xx);
                
        f = wgt * tmp; // is this f(x)/p(x)?
        f2 = f * f;    // this is (f(x)/p(x))^2 in equation 2.
        
        fb += f;
        f2b += f2;

      } // end of npg loop

      f2b = sqrt(f2b * npg);
      f2b = (f2b - fb) * (f2b + fb); // this is equivalent to s^(2) - (s^(1))^2
      
      if (f2b <= 0.0) 
          f2b=TINY;
      
      fbg += fb;
      f2bg += f2b;

      for (int k = ndim; k >= 1; k--) {
        kg[k] %= ng;
        if (++kg[k] != 1)
          break;
      }

    } // end of chunk for loop

    fbg = blockReduceSum(fbg);
    f2bg = blockReduceSum(f2bg);

    if (tx == 0) {
      //printf("Block %i done\n", blockIdx.x);
      atomicAdd(&result_dev[0], fbg);
      atomicAdd(&result_dev[1], f2bg);
    }

  } // end of subcube if
}

__inline__
void
rebin(double rc, int nd, double r[], double xin[], double xi[])
{

  //--------------------------------
  // Assumptions
  // dr is the remaining distance to cover in the axis that still needs to be
  // assigned to bins xin is the length we have already assigned what is r?
  //--------------------------------

  int i, k = 0;
  double dr = 0.0, xn = 0.0, xo = 0.0;

  for (i = 1; i < nd; i++) {
    //printf("rebininning bin %i dr:%.15e\n", i, dr);
    while (rc > dr) {
      dr += r[++k];
      //printf("dr:%.15e\n", dr);
    }
    //printf("dr is set to %.15e k:%i\n", dr, k);
    if (k > 1)
      xo = xi[k - 1];
    //printf("xo:%.15e\n", xo);
    xn = xi[k];//printf("xn:%.15e\n", xn);
    dr -= rc;//printf("setting dr to %.15e by subtracting rc:%.15e\n", dr, rc);

    xin[i] = xn - (xn - xo) * dr / r[k];
  }

  for (i = 1; i < nd; i++)
    xi[i] = xin[i];
  xi[nd] = 1.0;
}

template <typename IntegT, int ndim>
void
vegas(IntegT integrand,
      double epsrel,
      double epsabs,
      int fcode,
      double ncall,
      double* tgral,
      double* sd,
      double* chi2a,
      int* status,
      int titer,
      int itmax,
      int skip,
      quad::Volume<double, ndim> const* vol)
{
  //printf("EXECUTING VEGAS\n");  
  //std::ofstream outfile_fevals = GetOutFileVar("pmcubes_eval.csv");
  //std::ofstream outfile_intervals = GetOutFileVar("pmcubes_intervals.csv");


  //outfile_fevals <<"iter, threadID, chunk , sampleID, dim1, dim2, dim3, dim4, dim5, dim6, f\n";
  //outfile_intervals << "iter, dim1_int, dim2_int, dim3_int, dim4_int, dim5_int, dim6_int, f\n";
  
  constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
  constexpr int ndmx_p1 = Internal_Vegas_Params::get_NDMX_p1();
  constexpr int mxdim = Internal_Vegas_Params::get_MXDIM();
  constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();
  
  IntegT* d_integrand = cuda_copy_to_managed(integrand);
  double regn[2 * mxdim + 1];
  
  for (int j = 1; j <= ndim; j++) {
    regn[j] = vol->lows[j - 1];
    regn[j + ndim] = vol->highs[j - 1];
  }

  int i, it, j, nd, ndo, ng, npg;
  double calls, dv2g, dxg, rc, ti, tsi, wgt, xjac, xn, xnd, xo;
  double k, ncubes;
  double schi, si, swgt;
  double result[2];
  double *d, *dt, *dx, *r, *x, *xi, *xin;
  int* ia;
    
  d = (double*)malloc(sizeof(double) * (ndmx_p1) * (mxdim_p1));
  dt = (double*)malloc(sizeof(double) * (mxdim_p1));
  dx = (double*)malloc(sizeof(double) * (mxdim_p1));
  r = (double*)malloc(sizeof(double) * (ndmx_p1));
  x = (double*)malloc(sizeof(double) * (mxdim_p1));
  xi = (double*)malloc(sizeof(double) * (mxdim_p1) * (ndmx_p1));
  xin = (double*)malloc(sizeof(double) * (ndmx_p1));
  ia = (int*)malloc(sizeof(int) * (mxdim_p1));

  // code works only  for (2 * ng - NDMX) >= 0)


  ndo = 1;
  for (j = 1; j <= ndim; j++){
    //xi[j * NDMX1 + 1] = 1.0;    //this is weird I think it translates to xi[j*NDMX + 1 + 1] not xi[j*(NDMX+1) + 1], could we have bugs because of it?
    xi[j * ndmx + 1 + 1] = 1.0; 
  }
    si = swgt = schi = 0.0;
  nd = ndmx;
  ng = 1;
  ng = (int)pow(ncall / 2.0 + 0.25, 1.0 / ndim); // why do we add .25?
  for (k = 1, i = 1; i < ndim; i++) {

    k *= ng;
  }
  double sci = 1.0 / k;
  double sc = k;
  k *= ng;
  ncubes = k;
  npg = IMAX(ncall / k, 2);
  calls = (double)npg * (double)k;
  dxg = 1.0 / ng;
  
  double ing = dxg;
  for (dv2g = 1, i = 1; i <= ndim; i++)
    dv2g *= dxg;
  dv2g = (calls * dv2g * calls * dv2g) / npg / npg / (npg - 1.0);
  xnd = nd;
  dxg *= xnd;
  xjac = 1.0 / calls;
  for (j = 1; j <= ndim; j++) {
    dx[j] = regn[j + ndim] - regn[j];
    xjac *= dx[j];
  }

  for (i = 1; i <= IMAX(nd, ndo); i++)
    r[i] = 1.0;
  for (j = 1; j <= ndim; j++){
    rebin(ndo / xnd, nd, r, xin, &xi[j * ndmx + 1]);
  }
  ndo = nd;

  double *d_dev, *dx_dev, *x_dev, *xi_dev, *regn_dev, *result_dev;
  int* ia_dev;

  cudaMalloc((void**)&result_dev, sizeof(double) * 2);
  cudaCheckError();
  cudaMalloc((void**)&d_dev, sizeof(double) * (ndmx_p1) * (mxdim_p1));
  cudaCheckError();
  cudaMalloc((void**)&dx_dev, sizeof(double) * (mxdim_p1));
  cudaCheckError();
  cudaMalloc((void**)&x_dev, sizeof(double) * (mxdim_p1));
  cudaCheckError();
  cudaMalloc((void**)&xi_dev, sizeof(double) * (mxdim_p1) * (ndmx_p1));
  cudaCheckError();
  cudaMalloc((void**)&regn_dev, sizeof(double) * ((ndim * 2) + 1));
  cudaCheckError();
  cudaMalloc((void**)&ia_dev, sizeof(int) * (mxdim_p1));
  cudaCheckError();

  cudaMemcpy(dx_dev, dx, sizeof(double) * (mxdim_p1), cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(x_dev, x, sizeof(double) * (mxdim_p1), cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(
    regn_dev, regn, sizeof(double) * ((ndim * 2) + 1), cudaMemcpyHostToDevice);
  cudaCheckError();

  cudaMemset(ia_dev, 0, sizeof(int) * (mxdim_p1));

  int chunkSize = GetChunkSize(ncall);

  /*switch (fcode) {
    case 0:
      chunkSize = 2048;
      break;
    case 1:
      chunkSize = 32;
      break;
    case 2:
      chunkSize = 2048;
      break;
    case 4:
      chunkSize = 2048;
      break;
    default:
      chunkSize = 32;
      break;
  }*/

  uint32_t totalNumThreads = (uint32_t)((ncubes + chunkSize - 1) / chunkSize);
  uint32_t totalCubes = totalNumThreads * chunkSize;
  int extra = totalCubes - ncubes;
  int LastChunk = chunkSize - extra;
  uint32_t nBlocks =
    ((uint32_t)(((ncubes + BLOCK_DIM_X - 1) / BLOCK_DIM_X)) / chunkSize) + 1;
  uint32_t nThreads = BLOCK_DIM_X;

  /*std::cout<<"ng:"<<ng<<"\n";
  std::cout<<"ncubes:"<<ncubes<<"\n";
  std::cout<<"ncall:"<<ncall<<"\n";
  std::cout<<"k:"<<k<<"\n";
  std::cout<<"npg:"<<npg<<"\n";
  std::cout<<"totalNumThreads:"<<totalNumThreads<<"\n";
  std::cout<<"totalCubes:"<<totalCubes<<"\n";
  std::cout<<"chunkSize:"<<chunkSize<<"\n";*/
   
  /*size_t numFunctionEvaluations =   chunkSize*npg*(totalNumThreads-1) + (LastChunk+1)*npg;
  size_t numPoints = numFunctionEvaluations * (size_t)ndim;
	
  int* intervals = quad::cuda_malloc_managed<int>(numPoints);	
  double* evals = quad::cuda_malloc_managed<double>(numFunctionEvaluations);
  double* eval_points = quad::cuda_malloc_managed<double>(numPoints);
  
  cudaCheckError();
  for(int i=0; i< numFunctionEvaluations; ++i)
	  evals[i] = -1.;
  for(int i=0; i< numPoints; ++i)
	  eval_points[i]= -1.;
  for(int i=0; i< numPoints; ++i)
	  intervals[i]= -1.;*/
  //printf("itmax:%i\n", itmax);
  
  for (it = 1; it <= itmax && (*status) == 1; it++) {
    ti = tsi = 0.0;
    for (j = 1; j <= ndim; j++) {
      for (i = 1; i <= nd; i++)
        d[i * mxdim + 1 + j] = 0.0;
    }

    cudaMemcpy(xi_dev,
               xi,
               sizeof(double) * (mxdim_p1) * (ndmx_p1),
               cudaMemcpyHostToDevice);
    cudaCheckError(); // bin bounds
    cudaMemset(
      d_dev, 0, sizeof(double) * (ndmx_p1) * (mxdim_p1)); // bin contributions
    cudaMemset(result_dev, 0, 2 * sizeof(double));
    //printf("executing vegas_kernel\n");
    //std::cout<<"seed_init "<<time(0) << "\n";
    //std::cout<<"nBlocks:"<<nBlocks<<", nThreads:"<<nThreads<<", totalNumThreads:"<<totalNumThreads<<"\n";
    vegas_kernel<IntegT, ndim><<<nBlocks, nThreads>>>(d_integrand,
                                                      ng,
                                                      npg,
                                                      xjac,
                                                      dxg,
                                                      result_dev,
                                                      xnd,
                                                      xi_dev,
                                                      d_dev,
                                                      dx_dev,
                                                      regn_dev,
                                                      ncubes,
                                                      it,
                                                      sc,
                                                      sci,
                                                      ing,
                                                      chunkSize,
                                                      totalNumThreads,
                                                      LastChunk,
                                                      fcode,
                                                      time(0)/it/*,
                                                      evals,
                                                      eval_points,
													  intervals*/);
    
    
    
    cudaMemcpy(xi,
               xi_dev,
               sizeof(double) * (mxdim_p1) * (ndmx_p1),
               cudaMemcpyDeviceToHost);
    cudaCheckError(); 

    cudaMemcpy(d,
               d_dev,
               sizeof(double) * (ndmx_p1) * (mxdim_p1),
               cudaMemcpyDeviceToHost);
    cudaCheckError(); // we do need to the contributions for the rebinning
    cudaMemcpy(result, result_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);
    
    
	/*if(OUTFILEVAR == 2 || OUTFILEVAR == 3){
	
		for(int threadID = 0; threadID < totalNumThreads; threadID++){        
        
			int ThreadChunkSize = threadID == (totalNumThreads -1 )? LastChunk+1 : chunkSize;
            
			for(int chunk = 0; chunk < ThreadChunkSize; chunk ++){
				for(int sampleID = 1; sampleID <= npg; sampleID++){
                    
					size_t func_eval_index = threadID*chunkSize*npg + chunk*npg + (sampleID-1);
					outfile_fevals.precision(10);
					outfile_fevals << it <<","
                                << threadID << ","
                                << chunk << ","
                                << sampleID << ",";
                    
					for(int dim = 0; dim < ndim; dim++){
                        
						size_t dim_sample_point_index = func_eval_index*ndim + dim;
						outfile_fevals << std::scientific 
                            << eval_points[dim_sample_point_index] << ",";            
					}
					outfile_fevals << std::scientific << evals[func_eval_index] <<"\n";
				}
			}
			//printf("Done with thread %i/%i\n", threadID, totalNumThreads);
		}
    }*/
	
    /*if(OUTFILEVAR == 3){
        
		for(int threadID = 0; threadID < totalNumThreads; threadID++){
            
			int ThreadChunkSize = threadID == (totalNumThreads -1 )? LastChunk+1 : chunkSize;
            
			for(int chunk = 0; chunk < ThreadChunkSize; chunk ++){
				for(int sampleID = 1; sampleID <= npg; sampleID++){
                    
					size_t func_eval_index = threadID*chunkSize*npg + chunk*npg + (sampleID-1);
                    outfile_intervals << it << ",";                
					for(int dim = 0; dim < ndim; dim++){
						size_t dim_sample_point_index = func_eval_index*ndim + dim;
						outfile_intervals << intervals[dim_sample_point_index] << ",";            
					}
					outfile_intervals.precision(10);
					outfile_intervals << std::scientific << evals[func_eval_index] <<"\n";
				}
			}
		}	
	}
    //printf("Done with iter output\n");
    
    //reset
    for(int i=0; i< numFunctionEvaluations; ++i){
        evals[i] = -1.;
    }
      
    for(int i=0; i< numFunctionEvaluations*ndim; ++i){
        eval_points[i]= -1.;
    }*/
    
    ti = result[0];
    tsi = result[1];

    tsi *= dv2g;
    //printf("iter %d  integ = %.15e   std = %.15e var:%.15e dv2g:%f\n", it, ti, sqrt(tsi), tsi, dv2g);
    
    if (it > skip) {
      wgt = 1.0 / tsi;
      si += wgt * ti;
      schi += wgt * ti * ti;
      swgt += wgt;
      *tgral = si / swgt;
      *chi2a = (schi - si * (*tgral)) / (it - 0.9999);
      if (*chi2a < 0.0)
        *chi2a = 0.0;
      *sd = sqrt(1.0 / swgt);
      tsi = sqrt(tsi);
      *status = GetStatus(*tgral, *sd, it, epsrel, epsabs);
      printf("%5d,%.4e,%.4e,%9.2g\n", it, *tgral, *sd, *chi2a);
    }
     
    for (j = 1; j <= ndim; j++) {

      xo = d[1 * mxdim+1 + j]; // bin 1 of dim j, and bin 2 just below
      xn = d[2 * mxdim+1 + j];

      d[1 * mxdim+1 + j] = (xo + xn) / 2.0;
      dt[j] = d[1 * mxdim+1 + j]; // set dt sum to contribution of bin 1

      for (i = 2; i < nd; i++) {
        // rc is the contribution of the first and last bin? why?
        rc = xo + xn;
        xo = xn;
        xn = d[(i + 1) * mxdim+1 + j];
        d[i * mxdim+1 + j] = (rc + xn) / 3.0;
        dt[j] += d[i * mxdim+1 + j];
      }

      // do bin nd last
      d[nd * mxdim+1 + j] = (xo + xn) / 2.0;

      dt[j] += d[nd * mxdim+1 + j];
    }

    for (j = 1; j <= ndim; j++) {
      if (dt[j] > 0.0) { // enter if there is any contribution only
        rc = 0.0;
        for (i = 1; i <= nd; i++) {
			if(d[i*mxdim+1+j]<TINY)d[i*mxdim+1+j]=TINY;
			//added by Ioannis based on vegasBook.c
          r[i] = pow((1.0 - d[i * mxdim+1 + j] / dt[j]) /
                       (log(dt[j]) - log(d[i * mxdim+1 + j])),
                       Internal_Vegas_Params::get_ALPH());
          rc += r[i]; // rc is it the total number of sub-increments
        }
        rebin(rc / xnd, nd, r, xin, &xi[j * ndmx + 1]);
      }
    }

    

  } // end of iterations

  //  Start of iterations without adjustment

  cudaMemcpy(xi_dev,
             xi,
             sizeof(double) * (mxdim_p1) * (ndmx_p1),
             cudaMemcpyHostToDevice);
  cudaCheckError();

  for (it = itmax + 1; it <= titer && (*status); it++) {
    //printf("starting iteration\n");
    
    ti = tsi = 0.0;
    
    cudaMemset(result_dev, 0, 2 * sizeof(double));
    //std::cout<<"from host vegas_kernelF total num threads:"<< totalNumThreads << "\n";
    //printf("executing vegas_kernelF\n");
    //std::cout<<"seed_init "<<time(0) << "\n";

    vegas_kernelF<IntegT, ndim><<<nBlocks, nThreads>>>(d_integrand,
                                                       ng,
                                                       npg,
                                                       xjac,
                                                       dxg,
                                                       result_dev,
                                                       xnd,
                                                       xi_dev,
                                                       d_dev,
                                                       dx_dev,
                                                       regn_dev,
                                                       ncubes,
                                                       it,
                                                       sc,
                                                       sci,
                                                       ing,
                                                       chunkSize,
                                                       totalNumThreads,
                                                       LastChunk,
                                                       fcode,
                                                       time(0)/it);

    cudaMemcpy(result, result_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);
    //printf("Done with vegas_kernelF\n");
    
    ti = result[0];
    tsi = result[1];
    tsi *= dv2g; // is dv2g 1/(M-1)?
    printf("iter %d  integ = %.15e   std = %.15e\n", it, ti, sqrt(tsi));

    wgt = 1.0 / tsi;
    si += wgt * ti;
    schi += wgt * ti * ti;
    swgt += wgt;
    *tgral = si / swgt;
    *chi2a = (schi - si * (*tgral)) / (it - 0.9999);
    
    if (*chi2a < 0.0)
      *chi2a = 0.0;
  
    *sd = sqrt(1.0 / swgt);
    tsi = sqrt(tsi);
    *status = GetStatus(*tgral, *sd, it, epsrel, epsabs);
    // printf("it %d\n", it);
    //if (verbosity == 1)
    //  printf("%5d,%14.7g,%.8e,%9.2g\n", it, *tgral, *sd, *chi2a);
    // printf("%3d   %e  %e\n", it, ti, tsi);
    //printf("done with iteration post processing\n");
  } // end of iterations
    
  //std::cout<<"vegas:"<<*status<<"\n";
  free(d);
  free(dt);
  free(dx);
  free(ia);
  free(x);
  free(xi);

  cudaFree(d_dev);
  cudaFree(dx_dev);
  cudaFree(ia_dev);
  cudaFree(x_dev);
  cudaFree(xi_dev);
  cudaFree(regn_dev);
  
  //outfile_fevals.close();
  //outfile_intervals.close();
}

template <typename IntegT, int NDIM>
cuhreResult<double>
integrate(IntegT ig,
          int ndim,
          double epsrel,
          double epsabs,
          double ncall,
          quad::Volume<double, NDIM> const* volume,
          int totalIters = 15,
          int adjustIters = 15,
          int skipIters = 5)
{
  cuhreResult<double> result;
  result.status = 1;
  int fcode = -1; // test that it's really not being used anywhere
  vegas<IntegT, NDIM>(ig,
                      epsrel,
                      epsabs,
                      fcode,
                      ncall,
                      &result.estimate,
                      &result.errorest,
                      &result.chi_sq,
                      &result.status,
                      totalIters,
                      adjustIters,
                      skipIters,
                      volume);
  //std::cout<<"status:"<<result.status<<"\n";
  return result;
}

template <typename IntegT, int NDIM>
cuhreResult<double>
simple_integrate(IntegT integrand,
                 int ndim,
                 double epsrel,
                 double epsabs,
                 double ncall,
                 quad::Volume<double, NDIM> const* volume,
                 int totalIters = 15,
                 int adjustIters = 15,
                 int skipIters = 5)
{
  cuhreResult<double> result;
  result.status = 1;
  int fcode = -1; // test that it's really not being used anywhere

  //for(int i=0; i< NDIM; ++i)
   // printf("vol[%i]:(%f,%f)\n", i, volume->lows[i], volume->highs[i]);
  
  //printf("called simple_integrationw ith epsrel:%f, epsabs:%f, ncall:%f, totalIters:%i, adjustIters:%i, skipIters:%i\n", epsrel, epsabs, ncall, totalIters, adjustIters, skipIters);
  do {
    vegas<IntegT, NDIM>(integrand,
                        epsrel,
                        epsabs,
                        fcode,
                        ncall,
                        &result.estimate,
                        &result.errorest,
                        &result.chi_sq,
                        &result.status,
                        totalIters,
                        adjustIters,
                        skipIters,
                        volume);
    /*std::cout << std::scientific << result.estimate << ","
        << std::scientific << result.errorest << ","
        << ncall << ","
        << result.status << "\n";*/
   // break;
   printf("done with %e for epsrel %e status:%i\n", ncall, epsrel, result.status);
  } while (result.status == 1 && AdjustParams(ncall, totalIters) == true);

  return result;
}
}
#endif
