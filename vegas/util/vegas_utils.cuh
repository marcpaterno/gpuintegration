#ifndef VEGAS_UTILS_CUH
#define VEGAS_UTILS_CUH

#include "vegas/seqCodesDefs.hh"

#define BLOCK_DIM_X 128

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

#include <cmath>
#include <curand_kernel.h>


/*
class Parallel_params{
    int chunkSize;      //minimum cubes processed by each thread
    int LastChunk;      //number of cubes processed by the last thread, which gets assigned all leftover cubes
    uint32_t nBlocks;   //number of blocks launched by kernel
    uint32_t nThreads;  //number of threads per block, when launching kernel
    uint32_t totalNumThreads
    
    Vegas_Params* vegas_params = nullptr;
    friend class Mcubes_state;
    public:
        Set(double ncall){
            //requires that Vegas_Params have been properly set
            totalNumThreads = (uint32_t)((vegas_params->ncubes + chunkSize - 1) / chunkSize);
            uint32_t totalCubes = totalNumThreads * chunkSize;
            uint32_t extra = totalCubes - static_cast<uint32_t>(vegas_params->ncubes);
            LastChunk = chunkSize - extra;
            nBlocks =
                ((uint32_t)(((ncubes + BLOCK_DIM_X - 1) / BLOCK_DIM_X)) / chunkSize) + 1;
            nThreads = BLOCK_DIM_X;
        }
};

class Vegas_params{
    const int ncubes;
    const int ndim;
    const double ncall;
    const int npg;
    
    friend class Mcubes_state;
    public:
    
    Set(double NCALL, int NDIM){
        ncubes = ComputeNcubes(NCALL, NDIM);
        npg = Compute_samples_per_cube(NCALL, ncubes);
    }
    
};

class Mcubes_state{
    
    double* bin_right_coord = nullptr;
    double* bin_contributions = nullptr;
    
    int finished_iters = 0;
    
    const Parallel_params parallel_params;
    const Vegas_params vegas_params;
       
    public:
        Mcubes_state(double ncall, int ndim){
            constexpr int ndmx_p1 = Internal_Vegas_Params::get_NDMX_p1();
            constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();

            vegas_params.Set(ncall, ndim);
            parallel_params.Set(ncall);
            
            bin_right_coord = new double[ndmx_p1*mxdim_p1];  
            bin_contributions = new double[ndmx_p1*mxdim_p1];
        }
    
        ~Mcubes_state(){
            delete[] bin_right_coord;
            delete[] bin_contributions;  
        }
    
        void
        operator++(){
            finished_iters++;
        }
};


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
*/

class Custom_generator{  
      const uint32_t a = 1103515245;
      const uint32_t c = 12345;
      const uint32_t one = 1;
      const uint32_t expi = 31;
      uint32_t p = one << expi;
      uint32_t custom_seed = 0;
      uint64_t temp = 0; 
    
  public:
      __device__
      Custom_generator(uint32_t seed):custom_seed(seed){};
        
      __device__
      double operator()(){
        temp =  a * custom_seed + c;
        custom_seed = temp & (p - 1);    
        return (double) custom_seed / (double) p ; 
      }
      
      __device__
      void
      SetSeed(uint32_t seed){
        custom_seed = seed;
      }
};

class Curand_generator{
        curandState localState;
    public:
        __device__ 
        Curand_generator(){
            curand_init(0, blockIdx.x, threadIdx.x, &localState);
        }
        
        __device__ 
        Curand_generator(unsigned int seed){
            curand_init(seed, blockIdx.x, threadIdx.x, &localState);
        }
        
        __device__
        double operator()(){
            return curand_uniform_double(&localState);
        }
};

template<typename Generator>
class Random_num_generator{
        Generator generator;
    public:
        __device__
        Random_num_generator(unsigned int seed):generator(seed){}
        __device__
        double operator()(){
            return generator();
        }
};

namespace mcubes{
    
    template<typename T, typename U>
    __device__ __inline__ 
    constexpr bool 
    is_same(){
        return false;
    }
    
    template<typename Custom_generator>
    __device__ __inline__ 
    constexpr bool 
    is_same<Custom_generator, Custom_generator>(){
        return true;
    }
    
    //try the above to avoid class overhead
    
    template<typename T, typename U>
    struct TypeChecker 
    {
        //static const bool value = false; 
        __device__
        static constexpr bool 
        is_custom_generator(){
            return false;
        }
    };

    template<typename Custom_generator>
    struct TypeChecker<Custom_generator, Custom_generator>
    { 
        //static const bool value = true; 
        
        __device__
        static constexpr bool 
        is_custom_generator(){
            return true;
        }
    };
    
    /*template<typename Custom_generator>
    __device__
    constexpr bool 
    is_custom_generator(){
        return true;
    }
    
    template<typename Curand_generator>
    __device__ 
    constexpr bool 
    is_custom_generator(){
        return false;
    }*/
}

class Internal_Vegas_Params{
        static constexpr int ndmx = 500;
        static constexpr int mxdim = 20;
        static constexpr double alph = 1.5;
    
    public:
        
        __host__ __device__ static constexpr int get_NDMX(){return ndmx;}
   
        __host__ __device__ static constexpr int get_NDMX_p1(){return ndmx+1;}
       
        __host__ __device__ static constexpr  double get_ALPH(){return alph;}
       
        __host__ __device__ static constexpr  int get_MXDIM(){return mxdim;}
        
        constexpr __host__ __device__ static int get_MXDIM_p1(){return mxdim+1;}
};

__inline__
double ComputeNcubes(double ncall, int ndim){
    double ncubes = 1.;
    double intervals_per_dim = (int)pow(ncall / 2.0 + 0.25, 1.0 / ndim);
    for (int dim = 1; dim <= ndim; dim++) {
        ncubes *= intervals_per_dim;
    }
    
    return ncubes;
}

__inline__
int Compute_samples_per_cube(double ncall, double ncubes){
    int npg = IMAX(ncall / ncubes, 2);
    return npg; 
}

struct Kernel_Params{
    double ncubes = 0.;
    int npg = 0;
    uint32_t nBlocks = 0;
    uint32_t nThreads = 0;
    uint32_t totalNumThreads = 0;
    uint32_t totalCubes = 0;
    int extra = 0; 
    int LastChunk = 0; //how many chunks for the last thread
    
    Kernel_Params(double ncall, int chunkSize, int ndim){
        ncubes = ComputeNcubes(ncall, ndim);
        npg = Compute_samples_per_cube(ncall, ncubes);
    
        totalNumThreads = (uint32_t)((ncubes + chunkSize - 1) / chunkSize);
        totalCubes = totalNumThreads * chunkSize;
        extra = totalCubes - ncubes;
        LastChunk = chunkSize - extra;
        nBlocks =
            ((uint32_t)(((ncubes + BLOCK_DIM_X - 1) / BLOCK_DIM_X)) / chunkSize) + 1;
        nThreads = BLOCK_DIM_X;
    }
};

__inline__ bool
PrecisionAchieved(double estimate,
                  double errorest,
                  double epsrel,
                  double epsabs)
{
  if (std::abs(errorest / estimate) <= epsrel || errorest <= epsabs) {
    return true;
  } else
    return false;
}

__inline__ int
GetStatus(double estimate,
          double errorest,
          int iteration,
          double epsrel,
          double epsabs)
{
  if (PrecisionAchieved(estimate, errorest, epsrel, epsabs) && iteration >= 5) {
    return 0;
  } else
    return 1;
}

__inline__
int GetChunkSize(const double ncall){
    double small = 1.e7;
    double large = 8.e9;
    
    if(ncall <= small)
        return 32;
    else if(ncall <= large)
        return 2048; 
    else
        return 4096;  
}

/*
  returns true if it can update params for an extended run, updates two params
  returns false if it has increased both params to their maximum allowed values
  this maximum is not configurable by the user, placeholder values are currently placed
 */

bool
AdjustParams(double& ncall, int& totalIters)
{
  if (ncall >= 8.e9 && totalIters >= 100){
    // printf("Adjusting will return false\n");
    return false;
  }
  else if (ncall >= 8.e9) {
    //  printf("Adjusting will increase iters by 10 current value:%i\n", totalIters);
    totalIters += 10;
    return true;
  } else if (ncall >= 1.e9){
     // printf("Adjusting will increase ncall by 1e9 current value:%e\n", ncall);
        ncall += 1.e9;
    return true;
  }
  else{
    //  printf("Adjusting will multiply ncall by 10 current value:%e\n", ncall);
    ncall *= 10.;
    return true;  
  }
}

#endif
