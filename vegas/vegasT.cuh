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

__device__ long idum = -1;

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
#include <assert.h> 
#include <inttypes.h>

#define TINY 1.0e-30
#define WARP_SIZE 32
#define BLOCK_DIM_X 128

#define TRUE_VAL 
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
        static constexpr double ALPH = 1.5;
    
    public:
        
        __host__ __device__ static constexpr int get_NDMX(){return NDMX;}
   
        __host__ __device__ static constexpr int get_NDMX_p1(){return NDMX+1;}
       
        __host__ __device__ static constexpr  double get_ALPH(){return ALPH;}
       
        __host__ __device__ static constexpr  int get_MXDIM(){return MXDIM;}
        
        constexpr __host__ __device__ static int get_MXDIM_p1(){return MXDIM+1;}
};

//int verbosity = 0;

namespace cuda_mcubes{

//using MilliSeconds =
 // std::chrono::duration<double, std::chrono::milliseconds::period>;



#define NR_END 1
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
#define PI 3.14159265358979323846

__device__
double ran2(long *idum)
{
	int j;
	long k;
	static long idum2 = 123456789;
	static long iy = 0;
	static long iv[NTAB];
	double temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum = 1;
		else *idum = -(*idum);
		idum2 = (*idum);
		for (j = NTAB + 7; j >= 0; j--) {
			k = (*idum) / IQ1;
			*idum = IA1 * (*idum - k * IQ1) - k * IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB) iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ1;
	*idum = IA1 * (*idum - k * IQ1) - k * IR1;
	if (*idum < 0) *idum += IM1;
	k = idum2 / IQ2;
	idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
	if (idum2 < 0) idum2 += IM2;
	j = iy / NDIV;
	iy = iv[j] - idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp = AM * iy) > RNMX) return RNMX;
	else return temp;
}


__inline__ __device__ double
warpReduceSum(double val)
{
    
    //could there be an issue if block has fewer than 32 threads running?
    //at least with 1 thread and warpReduceSm commneted out, we still ahve chi-sq issues and 
    //worse absolute error
    
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

  if (wid == 0){
    val = warpReduceSum(val); // Final reduce within first warp
    // printf("[%i] final reduction for block warp %i reduction:%.15e\n", blockIdx.x, threadIdx.x, val);
  }
  __syncthreads(); //added by Ioannis due to cuda-memcheck racecheck reporting race between read/write
  return val;
}


__inline__ __device__ __host__ void
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


void Test_get_indx(int ndim, int ng, uint32_t totalNumThreads, int chunkSize, int it,  std::ofstream& interval_myfile){
    constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();
    
    if(it == 1)
		interval_myfile<<"m, kg[1], kg[2], kg[3], it\n";
	
    for(uint32_t m = 0; m < totalNumThreads; m++){
        uint32_t kg[mxdim_p1];
        get_indx(m , &kg[1], ndim, ng);
        
        interval_myfile<<m<<",";
            for(int ii = 1; ii<= ndim; ii++)
                interval_myfile<<kg[ii]<<",";
            interval_myfile<<it<<"\n";
    }
}


template <int ndim>
__inline__ __device__
void Setup_Integrand_Eval(curandState* localState, 
                            double xnd, double dxg, 
                            const double* const xi, const double* const regn, const double* const dx, 
                            const uint32_t* const kg, 
                            int* const ia, 
                            double* const x, 
                            double& wgt,
                            uint32_t& _seed,
                            uint64_t& temp,
                            int npg,
                            int chunkID,
                            int sampleID,
                            int chunkSize,
                            uint32_t cube_id,
                            double* randoms = nullptr)
{
    
#ifdef CUSTOM
	//uint64_t temp;
	uint32_t a = 1103515245;
	uint32_t c = 12345;
	uint32_t one, expi;
	one = 1;
	expi = 31;
	uint32_t p = one << expi;
    //uint32_t seed;
    //double ran00;
#endif
    
    double _ran00 = 0.;
    constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
    constexpr int ndmx1 = Internal_Vegas_Params::get_NDMX_p1();
    //long idum = (-1);
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
          const double ran00 = curand_uniform_double(localState);
          //const double ran00 = ran2(&idum);
 #ifdef CUSTOM
          	temp =  a * _seed + c;
            //if(threadIdx.x == 0 )
            //    printf("thread %i block%i seed:%u\n", threadIdx.x, blockIdx.x, _seed);
            _seed = temp & (p - 1);
            _ran00 = (double) _seed / (double) p ;    

#endif
          size_t m = blockIdx.x * blockDim.x + threadIdx.x;
          size_t nums_per_cube = npg*ndim;
          size_t nums_per_sample = ndim;
          //both chunkSize and sampleID should start at the same index to not be confusing
          //TODO: change indexing
          //uint32_t cube_id = (blockIdx.x * blockDim.x + threadIdx.x)*chunkSize + chunkID;
          size_t index = cube_id*nums_per_cube + nums_per_sample*(sampleID-1) + j-1;
          
          //if(threadIdx.x == 31)
          //    printf("attempting thread %i cube_id:%u index:%lu\n", threadIdx.x, cube_id, index); 
          
          randoms[index] = _ran00;

          //if(threadIdx.x == 31)
          //    printf("thread %i cube_id:%u index:%lu\n", threadIdx.x, cube_id, index); 

          const double xn = (kg[j] - _ran00) * dxg + 1.0;
          double rc = 0., xo = 0.;
          ia[j] = IMAX(IMIN((int)(xn), ndmx), 1);
                
          if (ia[j] > 1) {
            xo = (xi[j * ndmx1 + ia[j]]) - (xi[j * ndmx1 + ia[j] - 1]); //bin length
            rc = (xi[j * ndmx1 + ia[j] - 1]) + (xn - ia[j]) * xo; //scaling ran00 to bin bounds
          } else {
            xo = (xi[j * ndmx1 + ia[j]]);
            rc = (xn - ia[j]) * xo;
          }

          x[j] = regn[j] + rc * (dx[j]);
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
                        double& f2b,
                        uint32_t& _seed,
                        uint64_t& temp,
                        int chunkID,
                        int chunkSize,
                        uint32_t cube_id, 
                        double* randoms = nullptr,
                        double* funcevals = nullptr){
      constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();

      for (int k = 1; k <= npg; k++) {
          
        double wgt = xjac;
        Setup_Integrand_Eval<ndim>(localState, xnd, dxg, xi, regn, dx, kg,  ia, x, wgt, _seed, temp, npg, chunkID, k, chunkSize, cube_id, randoms);
        
        gpu::cudaArray<double, ndim> xx;             
        for (int i = 0; i < ndim; i++) {
          xx[i] = x[i + 1];                       
        }
        
        double tmp = gpu::apply(*d_integrand, xx);
        double f = wgt * tmp;     
        
        size_t m = blockIdx.x * blockDim.x + threadIdx.x;
        size_t nums_evals_per_cube = npg;

        size_t index = cube_id*nums_evals_per_cube + (k-1);
        funcevals[index] = f;
        double f2 = f * f;
        fb += f;
        f2b += f2;

#pragma unroll 2
        for (int j = 1; j <= ndim; j++) {
          atomicAdd(&d[ia[j] * mxdim_p1 + j], /*fabs(f)*/f2);
        }

      }
}    

template <typename IntegT, int ndim>
__inline__ __device__
void Process_chunks(IntegT* d_integrand, 
                int chunkSize, int lastChunk, int ng, int npg, 
                curandState* localState, 
                double dxg, double xnd, double xjac, 
                const double* const regn, const double* const dx, const double* const xi, 
                uint32_t* const kg, 
                int* const ia,  
                double* const x, 
                double& wgt, 
                double* d, 
                double& fbg, 
                double& f2bg,
                size_t cube_id_offset,
                double* randoms = nullptr,
                double* funcevals = nullptr){

    
    for (int t = 0; t < chunkSize; t++) {
      double fb = 0., f2b = 0.0;    //init to zero for each interval processed by thread
      //uint32_t _seed = t; //if doign single threaded, because one thread has all the ncubes, thus chunkSize equals ncubes
      uint32_t cube_id = cube_id_offset + t;
          
      uint32_t _seed = cube_id;
      uint64_t temp;
            
      Process_npg_samples<IntegT, ndim>(d_integrand, npg, xnd, xjac, localState, dxg, regn, dx, xi, kg, ia, x, wgt, d, fb, f2b, _seed, temp, t, chunkSize, cube_id, randoms, funcevals);
			
      f2b = sqrt(f2b * npg); 
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
             unsigned int seed_init,
             double* randoms = nullptr,
             double* funcevals = nullptr)
{

  //constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
  //constexpr int ndmx_p1 = Internal_Vegas_Params::get_NDMX_p1();
  //constexpr int mxdim = Internal_Vegas_Params::get_MXDIM();
  constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();
  
  uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t tx = threadIdx.x;

  double wgt;
  uint32_t kg[mxdim_p1];
  int ia[mxdim_p1];
  double x[mxdim_p1];
  double fbg = 0., f2bg = 0.;
  
  if (m < totalNumThreads) {
    
    size_t cube_id_offset = (blockIdx.x * blockDim.x + threadIdx.x)*chunkSize;
 
    if (m == totalNumThreads - 1)
      chunkSize = LastChunk;
    
    curandState localState;
    curand_init(seed_init, blockIdx.x, threadIdx.x, &localState);
    get_indx(/*m * chunkSize*/cube_id_offset, &kg[1], ndim, ng);
    
    if(threadIdx.x == 31)
        printf("m:%u/%u cube_id_offset:%i, ng:%i kg:%i, %i, %i\n", 
            m, totalNumThreads, cube_id_offset, ng, kg[1], kg[2], kg[3]);
            
    Process_chunks<IntegT, ndim>(d_integrand, chunkSize, LastChunk, ng, npg, &localState, dxg, xnd, xjac, regn, dx, xi, kg, ia, x, wgt, d, fbg, f2bg, cube_id_offset, randoms, funcevals);
  }
  
  //testing if synch is needed
  __syncthreads();
  fbg = blockReduceSum(fbg);
  f2bg = blockReduceSum(f2bg);
            
  if (tx == 0) {
      atomicAdd(&result_dev[0], fbg);
      atomicAdd(&result_dev[1], f2bg);
  }
   // end of subcube if
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
  constexpr int ndmx_p1 = Internal_Vegas_Params::get_NDMX_p1();
  //constexpr int mxdim = Internal_Vegas_Params::get_MXDIM();
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

  //unsigned long long seed;
  // seed_init = (iter) * ncubes;

  uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  double fb, f2b, wgt, xn, xo, rc, f, f2, ran00;
  uint32_t kg[mxdim_p1];
  int iaj;
  double x[mxdim_p1];
  int k, j;
  double fbg = 0., f2bg = 0.;
  long idum = (-1);
  if (m < totalNumThreads) {
          
    if (m == totalNumThreads - 1)
      chunkSize = LastChunk + 1;
  
    //seed = seed_init + m * chunkSize;
    
    curandState localState;
    curand_init(seed_init, blockIdx.x, threadIdx.x, &localState);
    //if(m == 0)
    //  printf("vegas_kernelF seed_init:%u\n", seed_init);

    fbg = f2bg = 0.0;
    get_indx(m * chunkSize, &kg[1], ndim, ng);
       
    for (int t = 0; t < chunkSize; t++) {
      fb = f2b = 0.0;

      for (k = 1; k <= npg; k++) {
        wgt = xjac;

        for (j = 1; j <= ndim; j++) {

          //ran00 = curand_uniform_double(&localState);
          ran00 = ran2(&idum);
          xn = (kg[j] - ran00) * dxg + 1.0;
          iaj = IMAX(IMIN((int)(xn), ndmx), 1);
	  

	  
	  if (iaj > 1) {          
            xo = xi[j * ndmx_p1 + iaj] - xi[j * ndmx_p1 + iaj - 1];
            rc = xi[j * ndmx_p1 + iaj - 1] + (xn - iaj) * xo;
          } else {
            xo = xi[j * ndmx_p1 + iaj];
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
                
        f = wgt * tmp; 
        f2 = f * f;    
        
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
    
  }
  
    //fbg = blockReduceSum(fbg);
    //f2bg = blockReduceSum(f2bg);

    if (tx == 0) {
      //printf("Block %i done\n", blockIdx.x);
      atomicAdd(&result_dev[0], fbg);
      atomicAdd(&result_dev[1], f2bg);
    }

   // end of subcube if
}

__inline__
void
rebin(double rc, int nd, double r[], double xin[], double xi[])
{
  int i, k = 0;
  double dr = 0.0, xn = 0.0, xo = 0.0;

  for (i = 1; i < nd; i++) {
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
  
	// for (i=1;i<=nd;i++) printf("xi[%i]: %.10e\n", i, xi[i]);
	// printf("---------------------\n");
}

void PrintBins(int iter, double* xi, double* d, int ndim, std::ofstream& outfile){
    int ndmx1 = Internal_Vegas_Params::get_NDMX_p1();
    int ndmx = Internal_Vegas_Params::get_NDMX();
    int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();
    
    if(iter == 1){
        outfile << "iter, dim, bin, bin_length, left, right, contribution\n";
    }
    
    if(iter <= 2){
        for(int dim = 1; dim <= ndim; dim++)
            for(int bin = 1; bin <= ndmx; bin++){
                
                double bin_length = xi[dim * ndmx1 + bin] - xi[dim * ndmx1 + bin -1];
                double left = xi[dim * ndmx1 + bin -1];
                double right = xi[dim * ndmx1 + bin];
                double contribution = d[bin * mxdim_p1 + dim];
                outfile << iter << "," 
                    << dim << ","
                    << bin << ","
                    << bin_length << "," 
                    << left << "," 
                    << right << "," 
                    << contribution << "\n";
            }
    }
}

void PrintRandomNums(double* randoms, int it, int ncubes, int npg, int ndim, std::ofstream& outfile){
    
    size_t nums_per_cube = npg*ndim;
    size_t nums_per_sample = ndim;
    
    if(it > 2)
        return;
    else{
        
        std::cout<<"expecting total random numbers:"<<ncubes*npg*ndim<<"\n";
        for(int cube = 0; cube < ncubes; cube++)
            for(int sample = 1; sample <= npg; sample++)
                for(int dim = 1; dim <= ndim; dim++){
                    
                    size_t index = cube*nums_per_cube + nums_per_sample*(sample-1) + dim-1;
     
                    
                    outfile << it << ","
                        << cube << ","
                        << cube << "," //same as chunk for single threaded
                        << sample << ","
                        << dim << ","
                        << randoms[index] << "\n";
				
				}
	} 
}

void PrintFuncEvals(double* funcevals, int it, int ncubes, int npg, int ndim, std::ofstream& outfile){
    
    size_t nums_per_cube = npg*ndim;
    size_t nums_per_sample = ndim;
    
    if(it > 2)
        return;
    else{
        
        std::cout<<"expecting total random numbers:"<<ncubes*npg*ndim<<"\n";
        for(int cube = 0; cube < ncubes; cube++)
            for(int sample = 1; sample <= npg; sample++){
                
                size_t nums_evals_per_chunk = npg;
                size_t index = cube*nums_evals_per_chunk + (sample-1);               
                outfile << it << ","
                        << cube << ","
                        << cube << "," //same as chunk for single threaded
                        << sample << ","
                        << funcevals[index] << "\n";
            }
            
            
    } 
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
    
  std::ofstream myfile_bin_bounds;
  myfile_bin_bounds.open ("pmcubes_bin_bounds_custom_1warp.csv");
  myfile_bin_bounds << "it, cube, chunk, sample, dim, ran00\n";   
    
  std::ofstream myfile_randoms;
  myfile_randoms.open ("pmcubes_random_nums_custom_1warp.csv");
  myfile_randoms << "it, cube, chunk, sample, dim, ran00\n";
  
  std::ofstream myfile_funcevals;
  myfile_funcevals.open ("pmcubes_funcevals_custom_1warp.csv");
  myfile_funcevals << "it, cube, chunk, sample, funceval\n";
  
  std::ofstream interval_myfile;
  interval_myfile.open ("pmcubes_intevals_custom_1warp.csv");
  
  std::ofstream iterations_myfile;
  iterations_myfile.open ("pmcubes_iters_custom_1warp.csv");
  iterations_myfile<<"iter, estimate, errorest, chi_sq, iter_estimate, iter_errorest\n";
  iterations_myfile.precision(15);
  
  auto t0 = std::chrono::high_resolution_clock::now();

  constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
  constexpr int ndmx_p1 = Internal_Vegas_Params::get_NDMX_p1();
  //constexpr int mxdim = Internal_Vegas_Params::get_MXDIM();
  constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();
  
  IntegT* d_integrand = cuda_copy_to_managed(integrand);
  double regn[2 * mxdim_p1];
  
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
    xi[j * ndmx_p1 + 1] = 1.0;  //this index is the first for each bin for each dimension
  }
    
  si = swgt = schi = 0.0;
  nd = ndmx;
  ng = 1;
  ng = (int)pow(ncall / 2.0 + 0.25, 1.0 / ndim); // why do we add .25?
  for (k = 1, i = 1; i < ndim; i++) {
    k *= ng;
  }
  
  double sci = 1.0 / k;  //I dont' think that's used anywhere
  double sc = k;         //I dont' think that's used either
  k *= ng;
  
  ncubes = k;
  
  npg = IMAX(ncall / k, 2);
  assert(npg == Compute_samples_per_cube(ncall, ncubes)); //to replace line directly above
  assert(ncubes == ComputeNcubes(ncall, ndim)); //to replace line directly above

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
    rebin(ndo / xnd, nd, r, xin, &xi[j * ndmx_p1]);
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

  //int chunkSize = GetChunkSize(ncall);
    //to be used in the future to simplify code
  
  int chunkSize = ncubes/32;
 
  uint32_t _totalNumThreads = (static_cast<uint32_t>(ncubes) % chunkSize) == 0 ? 
    (uint32_t)((ncubes) / chunkSize) : 
    (uint32_t)((ncubes) / chunkSize/* + 1*/);//there is no extra thread, there should be extra work on the last thread instead
    (uint32_t)((ncubes) / chunkSize/* + 1*/);//there is no extra thread, there should be extra work on the last thread instead
       
  uint32_t totalNumThreads = (uint32_t)((ncubes /*+ chunkSize - 1*/) / chunkSize);
  assert(_totalNumThreads == totalNumThreads);
  uint32_t totalCubes = totalNumThreads * chunkSize;
  int extra = ncubes - totalCubes;
  int LastChunk = /*chunkSize -*/ extra + chunkSize;
  /*uint32_t nBlocks =
    ((uint32_t)(((ncubes + BLOCK_DIM_X - 1) / BLOCK_DIM_X)) / chunkSize) + 1;
  uint32_t nThreads = BLOCK_DIM_X;*/
  uint32_t nBlocks = 1;
  uint32_t nThreads = 32;
  
  
  std::cout.precision(15);
  std::cout<<"ng:"<<ng<<"\n";
  std::cout<<"ncubes:"<<ncubes<<"\n";
  std::cout<<"ncall:"<<ncall<<"\n";
  std::cout<<"k:"<<k<<"\n";
  std::cout<<"npg:"<<npg<<"\n";
  std::cout<<"totalNumThreads:"<<totalNumThreads<<"\n";
  std::cout<<"_totalNumThreads:"<<_totalNumThreads<<"\n";
  std::cout<<"totalCubes:"<<totalCubes<<"\n";
  std::cout<<"chunkSize:"<<chunkSize<<"\n";
  std::cout<<"dv2g:"<<dv2g<<"\n";
  std::cout<<"extra:"<<extra<<"\n";
  std::cout<<"LastChunk:"<<LastChunk<<"\n";
  std::cout<<"-------------------\n";
  
  Kernel_Params kernel_params(ncall, chunkSize, ndim);   
  double* randoms = cuda_malloc_managed<double>((totalNumThreads*chunkSize+extra)*npg*ndim);
  double* funcevals = cuda_malloc_managed<double>((totalNumThreads*chunkSize+extra)*npg);
  printf("Allocating for %u cubes\n", totalNumThreads*chunkSize+extra);

  for (it = 1; it <= itmax /*&& (*status) == 1*/; it++) {
    ti = tsi = 0.0;
    for (j = 1; j <= ndim; j++) {
      for (i = 1; i <= nd; i++)
        d[i * mxdim_p1 + j] = 0.0;
    }

    cudaMemcpy(xi_dev,
               xi,
               sizeof(double) * (mxdim_p1) * (ndmx_p1),
               cudaMemcpyHostToDevice);
    cudaCheckError(); // bin bounds
    cudaMemset(
      d_dev, 0, sizeof(double) * (ndmx_p1) * (mxdim_p1)); // bin contributions
    cudaMemset(result_dev, 0, 2 * sizeof(double));
      
    using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>; 
    
    MilliSeconds time_diff = std::chrono::high_resolution_clock::now() - t0;
    unsigned int seed = static_cast<unsigned int>(time_diff.count()) + static_cast<unsigned int>(it);
    
    //Test_get_indx(ndim, ng, ncubes, chunkSize, it, interval_myfile);

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
                                                      /*time(NULL)/it*/
                                                      seed+it,
                                                      randoms,
                                                      funcevals);
    
    
    
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
    
    ti = result[0];
    tsi = result[1];

    tsi *= dv2g;
    //printf("-------------------------------------------\n");
    //printf("iter %d  integ = %.15e   std = %.15e var:%.15e dv2g:%f\n", it, ti, sqrt(tsi), tsi, dv2g);
    
    
    if (it > skip) {
      wgt = 1.0 / tsi;         
      si += wgt * ti;               
      schi += wgt * ti * ti;        
      swgt += wgt;                  
      *tgral = si / swgt;          
      *chi2a = (schi - si * (*tgral)) / (static_cast<double>(it) - 0.9999);
      if (*chi2a < 0.0)
        *chi2a = 0.0;
      *sd = sqrt(1.0 / swgt);
      tsi = sqrt(tsi);
      *status = GetStatus(*tgral, *sd, it, epsrel, epsabs);
      printf("cummulative ti:%5d, integral: %.15e, sd:%.4e,chi_sq:%9.2g\n", it, *tgral, *sd, *chi2a);
      //printf("-------------------------------------------\n");
       
    }
    
    iterations_myfile << it     << ","
                    << *tgral   << ","
                    << *sd      << ","
                    << *chi2a   << "," 
                    << ti       << ","
                    << tsi      << "\n";
                    
    PrintBins(it, xi, d, ndim, myfile_bin_bounds);
    PrintRandomNums(randoms, it, ncubes, npg, ndim, myfile_randoms);   
    PrintFuncEvals(funcevals, it, ncubes, npg, ndim, myfile_funcevals);
    // if(it > 1)  
       {   
    for (j = 1; j <= ndim; j++) {
      xo = d[1 * mxdim_p1 + j]; // bin 1 of dim j
      xn = d[2 * mxdim_p1 + j];  //bin 2 of dim j 
      d[1 * mxdim_p1 + j] = (xo + xn) / 2.0;
      dt[j] = d[1 * mxdim_p1 + j]; // set dt sum to contribution of bin 1

      for (i = 2; i < nd; i++) {
        rc = xo + xn;
        xo = xn;
        xn = d[(i + 1) * mxdim_p1 + j];
        d[i * mxdim_p1 + j] = (rc + xn) / 3.0;
        dt[j] += d[i * mxdim_p1 + j];
      }

     
      d[nd * mxdim_p1 + j] = (xo + xn) / 2.0;  // do bin nd last

      dt[j] += d[nd * mxdim_p1 + j];
    }

    for (j = 1; j <= ndim; j++) {
      if (dt[j] > 0.0) { // enter if there is any contribution only
        rc = 0.0;
        for (i = 1; i <= nd; i++) {
			if(d[i*mxdim_p1+j]<TINY) d[i*mxdim_p1+j]=TINY;    
          r[i] = pow((1.0 - d[i * mxdim_p1 + j] / dt[j]) /
                       (log(dt[j]) - log(d[i * mxdim_p1 + j])),
                       Internal_Vegas_Params::get_ALPH());
          rc += r[i]; // rc is it the total number of sub-increments
        }
        rebin(rc / xnd, nd, r, xin, &xi[j * ndmx_p1]);  //first bin of each dimension is at a diff index
      }
    }
    
  }
  } // end of iterations

  //  Start of iterations without adjustment

  cudaMemcpy(xi_dev,
             xi,
             sizeof(double) * (mxdim_p1) * (ndmx_p1),
             cudaMemcpyHostToDevice);
  cudaCheckError();

  for (it = itmax + 1; it <= titer /*&& (*status)*/; it++) {    
    ti = tsi = 0.0;
    cudaMemset(result_dev, 0, 2 * sizeof(double));
    
    using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>; 
    MilliSeconds time_diff = std::chrono::high_resolution_clock::now() - t0;
    unsigned int seed = static_cast<unsigned int>(time_diff.count()) + static_cast<unsigned int>(it);
    //std::cout<<"seed:"<<seed<<"\n";
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
                                                       /*time(0)/it,*/
                                                       /*seed*/it);

    cudaMemcpy(result, result_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);
    
    ti = result[0];
    tsi = result[1];
    tsi *= dv2g;
    //printf("iter %d  integ = %.15e   std = %.15e var:%.15e dv2g:%f\n", it, ti, sqrt(tsi), tsi, dv2g);
        
        
	wgt = 1.0 / tsi;       
	si += wgt * ti;         
	schi += wgt * ti * ti;  
	swgt += wgt;            
	*tgral = si / swgt;    
       
	*chi2a = (schi - si * (*tgral)) / (it - 0.9999); 
	if (*chi2a < 0.0) *chi2a = 0.0;
	*sd = sqrt(1.0 / swgt);
	tsi = sqrt(tsi);
        
    *status = GetStatus(*tgral, *sd, it, epsrel, epsabs);

    iterations_myfile << it     << ","
                    << *tgral   << ","
                    << *sd      << ","
                    << *chi2a   << "," 
                    << ti       << ","
                    << tsi      << "\n";    
    printf("cummulative ti:%5d, integral: %.15e, sd:%.4e,chi_sq:%9.2g\n", it, *tgral, *sd, *chi2a);
  } // end of iterations
      
  interval_myfile.close();
  myfile_bin_bounds.close();
  iterations_myfile.close();
  
  free(d);
  free(dt);
  free(dx);
  free(ia);
  free(x);
  free(xi);
  free(xin);
  free(r);

  cudaFree(d_dev);
  cudaFree(dx_dev);
  cudaFree(ia_dev);
  cudaFree(x_dev);
  cudaFree(xi_dev);
  cudaFree(regn_dev);
  cudaFree(result_dev);
  cudaFree(d_integrand);
  cudaFree(randoms);
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
