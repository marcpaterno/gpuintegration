/*

code works for gaussian and sin using switch statement. device pointerr/template
slow down the code by 2x

chunksize needs to be tuned based on the ncalls. For now hardwired using a
switch statement

nvcc -O2 -DCUSTOM -o vegas vegas_mcubes.cu -arch=sm_70
OR
nvcc -O2 -DCURAND -o vegas vegas_mcubes.cu -arch=sm_70

example run command

kokkos test ./vegas 0 6 0.0  10.0  2.0E+09  58, 0, 0
nvprof ./vegas 0 6 0.0  10.0  2.0E+09  58, 0, 0
nvprof  ./vegas 1 9 -1.0  1.0  1.0E+07 15 10 10
nvprof ./vegas 2 2 -1.0 1.0  1.0E+09 1 0 0

Last three arguments are: total iterations, iteration

#if defined(KOKKOS_ENABLE_CUDA)
  s << "macro  KOKKOS_ENABLE_CUDA      : defined" << std::endl;
#endif

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <ctime>
#include <sys/time.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#define PI 3.14159265358979323846
#define DEBUG 0
#define CUSTOM
#define BLOCK_DIM_X 128

#include <chrono>
#include "common/kokkos/cudaMemoryUtil.h"
#include "common/kokkos/cudaApply.cuh"
#include "common/kokkos/Volume.cuh"
#include "common/integration_result.hh"

namespace kokkos_mcubes {

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;

  typedef Kokkos::TeamPolicy<> team_policy;
  typedef Kokkos::TeamPolicy<>::member_type member_type;

  typedef Kokkos::TeamPolicy<> team_policy;
  typedef Kokkos::TeamPolicy<>::member_type member_type;

  // int scratch_size = ScratchViewType::shmem_size(256);

  template <typename T, typename U>
  KOKKOS_INLINE_FUNCTION std::common_type_t<T, U>
  IMAX(T a, U b)
  {
    return (a > b) ? a : b;
  }

  template <typename T, typename U>
  KOKKOS_INLINE_FUNCTION std::common_type_t<T, U>
  IMIN(T a, U b)
  {
    return (a < b) ? a : b;
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION T
  blockReduceSum(T val, const member_type team_member)
  {
    double sum = 0.;
    team_member.team_barrier();
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, team_member.team_size()),
      [=](int& i, T& lsum) { lsum += val; },
      sum);
    return sum;
  }

  class Internal_Vegas_Params {
    static constexpr int ndmx = 500;
    static constexpr int mxdim = 20;
    static constexpr double alph = 1.5;
    static constexpr double tiny = 1.0e-30;

  public:
    static constexpr int
    get_NDMX()
    {
      return ndmx;
    }

    static constexpr double
    get_TINY()
    {
      return tiny;
    }

    static constexpr int
    get_NDMX_p1()
    {
      return ndmx + 1;
    }

    static constexpr double
    get_ALPH()
    {
      return alph;
    }

    static constexpr int
    get_MXDIM()
    {
      return mxdim;
    }

    static constexpr int
    get_MXDIM_p1()
    {
      return mxdim + 1;
    }
  };

  __inline__ double
  ComputeNcubes(double ncall, int ndim)
  {
    double ncubes = 1.;
    double intervals_per_dim = (int)pow(ncall / 2.0 + 0.25, 1.0 / ndim);
    for (int dim = 1; dim <= ndim; dim++) {
      ncubes *= intervals_per_dim;
    }

    return ncubes;
  }

  __inline__ int
  Compute_samples_per_cube(double ncall, double ncubes)
  {
    int npg = IMAX(ncall / ncubes, 2);
    return npg;
  }

  struct Kernel_Params {
    double ncubes = 0.;
    int npg = 0;
    uint32_t nBlocks = 0;
    uint32_t nThreads = 0;
    uint32_t totalNumThreads = 0;
    uint32_t totalCubes = 0;
    int extra = 0;
    int LastChunk = 0; // how many chunks for the last thread

    Kernel_Params(double ncall, int chunkSize, int ndim)
    {
      ncubes = ComputeNcubes(ncall, ndim);
      npg = Compute_samples_per_cube(ncall, ncubes);

      totalNumThreads = (uint32_t)((ncubes) / chunkSize);
      totalCubes = totalNumThreads * chunkSize;
      extra = totalCubes - ncubes;
      LastChunk = chunkSize - extra;
      nBlocks = totalNumThreads % BLOCK_DIM_X == 0 ?
                  totalNumThreads / BLOCK_DIM_X :
                  totalNumThreads / BLOCK_DIM_X + 1;
      nThreads = BLOCK_DIM_X;
    }
  };

  KOKKOS_INLINE_FUNCTION void
  get_indx(int ms, uint32_t* da, int ND, int NINTV)
  {

    int dp[Internal_Vegas_Params::get_MXDIM()];
    int j, t0, t1;
    int m = ms;
    dp[0] = 1;
    dp[1] = NINTV;

    for (j = 0; j < ND - 2; j++) {
      dp[j + 2] = dp[j + 1] * NINTV;
    }
    //
    for (j = 0; j < ND; j++) {
      t0 = dp[ND - j - 1];
      t1 = m / t0;
      da[j] = 1 + t1;
      m = m - t1 * t0;
    }
  }

  class Custom_generator {
    const uint32_t a = 1103515245;
    const uint32_t c = 12345;
    const uint32_t one = 1;
    const uint32_t expi = 31;
    uint32_t p = one << expi;
    uint32_t custom_seed = 0;
    uint64_t temp = 0;
    const int block_id;
    const int thread_id;

  public:
    KOKKOS_INLINE_FUNCTION
    Custom_generator(uint32_t seed, int blockId, int threadId)
      : custom_seed(seed), block_id(blockId), thread_id(threadId){};

    KOKKOS_INLINE_FUNCTION double
    operator()()
    {
      temp = a * custom_seed + c;
      custom_seed = temp & (p - 1);
      // printf("random %f a:%lu c:%lu temp:%lu p:%lu\n", (double)custom_seed /
      // (double)p, a, c, temp, p);
      return (double)custom_seed / (double)p;
    }

    KOKKOS_INLINE_FUNCTION void
    SetSeed(uint32_t seed)
    {
      custom_seed = seed;
    }
  };

  class Curand_generator {
#if defined(__CUDA_ARCH__)
    curandState localState;
#endif
  public:
    KOKKOS_INLINE_FUNCTION
    Curand_generator(int blockId, int threadId)
    {

#if defined(__CUDA_ARCH__)
      curand_init(0, blockId, threadId, &localState);
#endif
    }

    KOKKOS_INLINE_FUNCTION
    Curand_generator(unsigned int seed, int blockId, int threadId)
    {
#if defined(__CUDA_ARCH__)
      curand_init(seed, blockId, threadId, &localState);
#endif
    }

    KOKKOS_INLINE_FUNCTION double
    operator()()
    {
#if defined(__CUDA_ARCH__)
      return curand_uniform_double(&localState);
#else
      return -1.;
#endif
    }
  };

  template <typename Generator>
  class Random_num_generator {
    Generator generator;

  public:
    KOKKOS_INLINE_FUNCTION
    Random_num_generator(unsigned int seed, int blockId, int threadId)
      : generator(seed, blockId, threadId)
    {}

    KOKKOS_INLINE_FUNCTION double
    operator()()
    {
      return generator();
    }

    KOKKOS_INLINE_FUNCTION void
    SetSeed(uint32_t seed)
    {
      generator.SetSeed(seed);
    }
  };

  template <typename T, typename U>
  KOKKOS_INLINE_FUNCTION constexpr bool
  is_same()
  {
    return false;
  }

  template <typename Custom_generator>
  KOKKOS_INLINE_FUNCTION constexpr bool
  is_same<Custom_generator, Custom_generator>()
  {
    return true;
  }

  template <typename T, typename U>
  struct TypeChecker {
    KOKKOS_INLINE_FUNCTION static constexpr bool
    is_custom_generator()
    {
      return false;
    }
  };

  template <class Custom_generator>
  struct TypeChecker<Custom_generator, Custom_generator> {

    KOKKOS_INLINE_FUNCTION static constexpr bool
    is_custom_generator()
    {
      return true;
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
    if (PrecisionAchieved(estimate, errorest, epsrel, epsabs) &&
        iteration >= 5) {
      return 0;
    } else
      return 1;
  }

__inline__ int
GetChunkSize(const double ncall)
{
  double small = 1.e7;
  double large = 8.e9;
  if(ncall < 1e6)
    return 4;
  if (ncall <= small)
    return 32;
  else if (ncall <= large)
    return 2048;
  else
    return 4096;
}

  template <int ndim, typename GeneratorType = kokkos_mcubes::Custom_generator>
  KOKKOS_INLINE_FUNCTION void
  Setup_Integrand_Eval(Random_num_generator<GeneratorType>* rand_num_generator,
                       double xnd,
                       double dxg,
                       ViewVectorDouble xi,
                       ViewVectorDouble regn,
                       ViewVectorDouble dx,
                       const uint32_t* const kg,
                       int* const ia,
                       double* const x,
                       double& wgt)
  {
    constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
    constexpr int ndmx1 = Internal_Vegas_Params::get_NDMX_p1();

    for (int j = 1; j <= ndim; j++) {
      const double ran00 = (*rand_num_generator)();
      const double xn = (kg[j] - ran00) * dxg + 1.0;
      double rc = 0., xo = 0.;
      ia[j] = IMAX(IMIN((int)(xn), ndmx), 1);

      if (ia[j] > 1) {
        xo =
          (xi[j * ndmx1 + ia[j]]) - (xi[j * ndmx1 + ia[j] - 1]); // bin length
        rc = (xi[j * ndmx1 + ia[j] - 1]) +
             (xn - ia[j]) * xo; // scaling ran00 to bin bounds
      } else {
        xo = (xi[j * ndmx1 + ia[j]]);
        rc = (xn - ia[j]) * xo;
      }

      x[j] = regn[j] + rc * (dx[j]);
      wgt *= xo * xnd; // xnd is number of bins, xo is the length of the bin,
                       // xjac is 1/num_calls
    }
  }

  template <typename IntegT,
            int ndim,
            typename GeneratorType = kokkos_mcubes::Custom_generator>
  KOKKOS_INLINE_FUNCTION void
  Process_npg_samples(Kokkos::View<IntegT*, Kokkos::CudaUVMSpace> integrand,
                      int npg,
                      double xnd,
                      double xjac,
                      Random_num_generator<GeneratorType>* rand_num_generator,
                      double dxg,
                      ViewVectorDouble regn,
                      ViewVectorDouble dx,
                      ViewVectorDouble xi,
                      const uint32_t* kg,
                      int* const ia,
                      double* const x,
                      double& wgt,
                      ViewVectorDouble d,
                      double& fb,
                      double& f2b,
                      uint32_t cube_id)
  {
    constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();

    for (int k = 1; k <= npg; k++) {

      double wgt = xjac;
      Setup_Integrand_Eval<ndim, GeneratorType>(
        rand_num_generator, xnd, dxg, xi, regn, dx, kg, ia, x, wgt);

      gpu::cudaArray<double, ndim> xx;
      for (int i = 0; i < ndim; i++) {
        xx[i] = x[i + 1];
      }

      const double tmp = gpu::apply(integrand(0), xx);
      const double f = wgt * tmp;

      double f2 = f * f;
      fb += f;
      f2b += f2;

      for (int j = 1; j <= ndim; j++) {
        Kokkos::atomic_add(&d(ia[j] * (mxdim_p1) + j), fabs(f2));
      }
    }
  }

  template <typename IntegT,
            int ndim,
            typename GeneratorType = kokkos_mcubes::Custom_generator>
  KOKKOS_INLINE_FUNCTION void
  Process_chunks(Kokkos::View<IntegT*, Kokkos::CudaUVMSpace> integrand,
                 int chunkSize,
                 int lastChunk,
                 int ng,
                 int npg,
                 Random_num_generator<GeneratorType>* rand_num_generator,
                 double dxg,
                 double xnd,
                 double xjac,
                 ViewVectorDouble regn,
                 ViewVectorDouble dx,
                 ViewVectorDouble xi,
                 uint32_t* const kg,
                 int* const ia,
                 double* const x,
                 double& wgt,
                 ViewVectorDouble d,
                 double& fbg,
                 double& f2bg,
                 size_t cube_id_offset)
  {

    for (int t = 0; t < chunkSize; t++) {
      double fb = 0.,
             f2b = 0.0; // init to zero for each interval processed by thread
      uint32_t cube_id = cube_id_offset + t;

      if constexpr (kokkos_mcubes::TypeChecker<
                      GeneratorType,
                      Custom_generator>::is_custom_generator()) {
        rand_num_generator->SetSeed(cube_id);
      }

      Process_npg_samples<IntegT, ndim, GeneratorType>(integrand,
                                                       npg,
                                                       xnd,
                                                       xjac,
                                                       rand_num_generator,
                                                       dxg,
                                                       regn,
                                                       dx,
                                                       xi,
                                                       kg,
                                                       ia,
                                                       x,
                                                       wgt,
                                                       d,
                                                       fb,
                                                       f2b,
                                                       cube_id);

      f2b = sqrt(f2b * npg);
      f2b = (f2b - fb) * (f2b + fb);

      if (f2b <= 0.0) {
        f2b = Internal_Vegas_Params::get_TINY();
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

  template <typename IntegT,
            int ndim,
            typename GeneratorType = kokkos_mcubes::Custom_generator>
  void
  vegas_kernel_kokkos(Kokkos::View<IntegT*, Kokkos::CudaUVMSpace> integrand,
                      uint32_t nBlocks,
                      uint32_t nThreads,
                      int ng,
                      int npg,
                      double xjac,
                      double dxg,
                      ViewVectorDouble result_dev,
                      double xnd,
                      ViewVectorDouble xi,
                      ViewVectorDouble d,
                      ViewVectorDouble dx,
                      ViewVectorDouble regn,
                      int _chunkSize,
                      uint32_t totalNumThreads,
                      int LastChunk,
                      unsigned int seed_init)
  {
    Kokkos::parallel_for(
      "kokkos_vegas_kernel",
      team_policy(nBlocks, nThreads).set_scratch_size(0, Kokkos::PerTeam(2 * nThreads * sizeof(double))),
      KOKKOS_LAMBDA(const member_type team_member) {
        int chunkSize = _chunkSize;
        //ScratchViewDouble sh_buff(team_member.team_scratch(0), 2 * nThreads);
        constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();
        uint32_t tx = team_member.team_rank(); // local id
        uint32_t m = team_member.league_rank() * team_member.team_size() +
                     tx; // global thread id
        double wgt;
        uint32_t kg[mxdim_p1];
        int ia[mxdim_p1];
        double x[mxdim_p1];
        double fbg = 0., f2bg = 0.;

        if (m < totalNumThreads) {

          size_t cube_id_offset =
            (team_member.league_rank() * team_member.team_size() + tx) *
            chunkSize;

          if (m == totalNumThreads - 1)
            chunkSize = LastChunk;

          Random_num_generator<GeneratorType> rand_num_generator(
            seed_init, team_member.league_rank(), team_member.team_rank());
          get_indx(cube_id_offset, &kg[1], ndim, ng);

          Process_chunks<IntegT, ndim, GeneratorType>(integrand,
                                                      chunkSize,
                                                      LastChunk,
                                                      ng,
                                                      npg,
                                                      &rand_num_generator,
                                                      dxg,
                                                      xnd,
                                                      xjac,
                                                      regn,
                                                      dx,
                                                      xi,
                                                      kg,
                                                      ia,
                                                      x,
                                                      wgt,
                                                      d,
                                                      fbg,
                                                      f2bg,
                                                      cube_id_offset);
        }

         //sh_buff(tx) = fbg;
         //sh_buff(tx + team_member.team_size()) = f2bg;
        // printf("pre-reductin vals:%f +- %f\n", fbg, f2bg);
        team_member.team_barrier();

        fbg = blockReduceSum(fbg, team_member);
        f2bg = blockReduceSum(f2bg, team_member);
        if (tx == 0) {
           //double fbgs = 0.0;
           //double f2bgs = 0.0;
          /*for (int ii = 0; ii < (m + ii) < totalNumThreads && ii < team_member.team_size(); ++ii) { 
            fbgs += sh_buff(ii); 
            f2bgs += sh_buff(ii + team_member.team_size());
          }*/
          // printf("block %i storing %f +- %f\n", team_member.league_rank(),
          // fbg, f2bg);
          Kokkos::atomic_add(&result_dev(0), fbg);
          Kokkos::atomic_add(&result_dev(1), f2bg);
        }
      });
  }

  template <typename IntegT,
            int ndim,
            typename GeneratorType = kokkos_mcubes::Custom_generator>
  void
  vegas_kernel_kokkosF(Kokkos::View<IntegT*, Kokkos::CudaUVMSpace> integrand,
                       uint32_t nBlocks,
                       uint32_t nThreads,
                       int ng,
                       int npg,
                       double xjac,
                       double dxg,
                       ViewVectorDouble result_dev,
                       double xnd,
                       ViewVectorDouble xi,
                       ViewVectorDouble dx,
                       ViewVectorDouble regn,
                       int _chunkSize,
                       uint32_t totalNumThreads,
                       int LastChunk,
                       unsigned int seed_init)
  {

    constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
    constexpr int ndmx_p1 = Internal_Vegas_Params::get_NDMX_p1();
    constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();

    Kokkos::parallel_for(
      "vegas_kernelF",
      team_policy(nBlocks, nThreads).set_scratch_size(0, Kokkos::PerTeam(2 * nThreads * sizeof(double))),
      KOKKOS_LAMBDA(const member_type team_member) {
        int chunkSize = _chunkSize;
       // ScratchViewDouble sh_buff(team_member.team_scratch(0), 2 * nThreads);

        uint32_t tx = team_member.team_rank();
        uint32_t m =
          team_member.league_rank() * team_member.team_size() + tx; // global thread id

        double fb, f2b, wgt, xn, xo, rc, f, f2, ran00;
        uint32_t kg[mxdim_p1];
        int iaj;
        double x[mxdim_p1];
        int k;
        double fbg = 0., f2bg = 0.;

        if (m < totalNumThreads) {

          size_t cube_id_offset = (team_member.league_rank() * team_member.team_size() +  tx) * chunkSize;

          if (m == totalNumThreads - 1)
            chunkSize = LastChunk;
          Random_num_generator<GeneratorType> rand_num_generator(
            seed_init, team_member.league_rank(), team_member.team_rank());

          fbg = f2bg = 0.0;
          get_indx(cube_id_offset, &kg[1], ndim, ng);

          for (int t = 0; t < chunkSize; t++) {
            fb = f2b = 0.0;
            if constexpr (kokkos_mcubes::TypeChecker<
                            GeneratorType,
                            Custom_generator>::is_custom_generator()) {
              rand_num_generator.SetSeed(cube_id_offset);
            }

            for (k = 1; k <= npg; k++) {
              wgt = xjac;

              for (int j = 1; j <= ndim; j++) {

                ran00 = rand_num_generator();
                xn = (kg[j] - ran00) * dxg + 1.0;
                iaj = IMAX(IMIN((int)(xn), ndmx),1); 

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

              tmp = gpu::apply((integrand(0)), xx);
              f = wgt * tmp;
              f2 = f * f;

              fb += f;
              f2b += f2;
            } // end of npg loop

            f2b = sqrt(f2b * npg);
            f2b = (f2b - fb) * (f2b + fb);
            if (f2b <= 0.0)
              f2b = Internal_Vegas_Params::get_TINY();

            fbg += fb;
            f2bg += f2b;

            for (int k = ndim; k >= 1; k--) {
              kg[k] %= ng;
              if (++kg[k] != 1)
                break;
            }
          } // end of chunk for loop

        } // end of subcube if

        //sh_buff(tx) = fbg;
        //sh_buff(tx + team_member.team_size()) = f2bg;
        team_member.team_barrier();
        fbg = blockReduceSum(fbg, team_member);
        f2bg = blockReduceSum(f2bg, team_member);
        

        if (tx == 0) {
          Kokkos::atomic_add(&result_dev(0), fbg);
          Kokkos::atomic_add(&result_dev(1), f2bg);
        }
      });
  }

  void
  rebin(double rc, int nd, double* r, double* xin, double* xi)
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
  }

  template <typename IntegT,
            int ndim,
            typename GeneratorType = typename kokkos_mcubes::Custom_generator>
  void
  vegas(IntegT integrand,
        double epsrel,
        double epsabs,
        double ncall,
        double* tgral,
        double* sd,
        double* chi2a,
        int* status,
        size_t* iters,
        int titer,
        int itmax,
        int skip,
        quad::Volume<double, ndim> const* vol)
  {

    auto t0 = std::chrono::high_resolution_clock::now();

    constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
    constexpr int ndmx_p1 = Internal_Vegas_Params::get_NDMX_p1();
    constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();

    Kokkos::View<IntegT*, Kokkos::CudaUVMSpace> d_integrand("d_integrand", 1);
    d_integrand(0) = integrand;

    int i, it, j, k, nd, ndo, ng, npg, ncubes;
    double calls, dv2g, dxg, rc, ti, tsi, wgt, xjac, xn, xnd, xo;
    double schi, si, swgt;

    ViewVectorDouble d_result("result", 2); // result_dev in the original
    ViewVectorDouble d_xi("xi",
                          ((ndmx_p1) * (mxdim_p1))); // xi_dev in the original
    ViewVectorDouble d_d("d", ((ndmx_p1) * (mxdim_p1))); // d_dev in the
                                                         // original
    ViewVectorDouble d_dx("dx", mxdim_p1);           // dx_dev in the original
    ViewVectorDouble d_regn("regn", 2 * (mxdim_p1)); // regn_dev in the original

    // create host mirrors of device views
    ViewVectorDouble::HostMirror result = Kokkos::create_mirror_view(d_result);
    ViewVectorDouble::HostMirror xi =
      Kokkos::create_mirror_view(d_xi); // left coordinate of bin
    ViewVectorDouble::HostMirror d = Kokkos::create_mirror_view(d_d);
    ViewVectorDouble::HostMirror dx = Kokkos::create_mirror_view(d_dx);
    ViewVectorDouble::HostMirror regn = Kokkos::create_mirror_view(d_regn);

    for (j = 1; j <= ndim; j++) {
      regn[j] = vol->lows[j - 1];
      regn[j + ndim] = vol->highs[j - 1];
    }

    // create arrays used only on host
    double *dt, *r, *xin;
    dt = (double*)malloc(sizeof(double) * (mxdim_p1));
    r = (double*)malloc(sizeof(double) * (ndmx_p1));
    xin = (double*)malloc(sizeof(double) * (ndmx_p1));

    // code works only  for (2 * ng - NDMX) >= 0)
    ndo = 1;
    for (j = 1; j <= ndim; j++)
      xi(j * (ndmx_p1) + 1) = 1.0;

    si = swgt = schi = 0.0;
    nd = ndmx;
    ng = 1;

    ng = (int)pow(ncall / 2.0 + 0.25, 1.0 / ndim);
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

    for (dv2g = 1, i = 1; i <= ndim; i++)
      dv2g *= dxg;

    dv2g = (calls * dv2g * calls * dv2g) / npg / npg / (npg - 1.0);

    xnd = nd;
    dxg *= xnd;
    xjac = 1.0 / calls;

    for (j = 1; j <= ndim; j++) {
      dx(j) = regn[j + ndim] - regn[j];
      xjac *= dx(j);
    }

    for (i = 1; i <= IMAX(nd, ndo); i++)
      r[i] = 1.0;

    for (j = 1; j <= ndim; j++) {
      rebin(ndo / xnd, nd, r, xin, xi.data() + (j * (ndmx_p1)));
    }

    ndo = nd;
    Kokkos::deep_copy(d_dx, dx);
    Kokkos::deep_copy(d_regn, regn);

    int chunkSize = GetChunkSize(ncall);

    uint32_t totalNumThreads =
      (uint32_t)((ncubes /*+ chunkSize - 1*/) / chunkSize);
    uint32_t totalCubes = totalNumThreads * chunkSize; // even-split cubes
    int extra = ncubes - totalCubes;                   // left-over cubes
    int LastChunk = extra + chunkSize; // last chunk of last thread
    Kernel_Params params(ncall, chunkSize, ndim);

    for (it = 1; it <= itmax && (*status) == 1; (*iters)++, it++) {

      ti = tsi = 0.0;
      for (j = 1; j <= ndim; j++) {
        for (i = 1; i <= nd; i++)
          d(i * mxdim_p1 + j) = 0.0;
      }

      deep_copy(d_xi, xi);
      Kokkos::deep_copy(d_result, 0.0);
      // std::cout<<"npg:"<<npg<<", dxg:"<<dxg<<", ng:"<< ng << std::endl;
      MilliSeconds time_diff = std::chrono::high_resolution_clock::now() - t0;
      unsigned int seed = /*static_cast<unsigned int>(time_diff.count()) +
                          */static_cast<unsigned int>(it);
      // seed = 0;
      vegas_kernel_kokkos<IntegT, ndim, GeneratorType>(d_integrand,
                                                       params.nBlocks,
                                                       params.nThreads,
                                                       ng,
                                                       npg,
                                                       xjac,
                                                       dxg,
                                                       d_result,
                                                       xnd,
                                                       d_xi,
                                                       d_d,
                                                       d_dx,
                                                       d_regn,
                                                       chunkSize,
                                                       totalNumThreads,
                                                       LastChunk,
                                                       seed + it);
      Kokkos::fence();

      deep_copy(xi, d_xi);
      deep_copy(d, d_d);
      deep_copy(result, d_result);

      ti = result(0);
      tsi = result(1);
      tsi *= dv2g;
      // printf("iter %d  integ = %.15e   std = %.15e var:%.15e dv2g:%f\n", it,
      // ti, sqrt(tsi), tsi, dv2g);

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
        //printf("%i %e +- %e iteration: %e +- %e chi:%.15f\n",
        //    it, *tgral, *sd, ti, sqrt(tsi), *chi2a);
        tsi = sqrt(tsi);
        *status = GetStatus(*tgral, *sd, it, epsrel, epsabs);
      }

      for (j = 1; j <= ndim; j++) {
        xo = d[1 * mxdim_p1 + j]; // bin 1 of dim j
        xn = d[2 * mxdim_p1 + j]; // bin 2 of dim j
        d[1 * mxdim_p1 + j] = (xo + xn) / 2.0;
        dt[j] = d[1 * mxdim_p1 + j]; // set dt sum to contribution of bin 1

        for (i = 2; i < nd; i++) {
          rc = xo + xn;
          xo = xn;
          xn = d[(i + 1) * mxdim_p1 + j];
          d[i * mxdim_p1 + j] = (rc + xn) / 3.0;
          dt[j] += d[i * mxdim_p1 + j];
        }

        d[nd * mxdim_p1 + j] = (xo + xn) / 2.0; // do bin nd last
        dt[j] += d[nd * mxdim_p1 + j];
      }

      for (j = 1; j <= ndim; j++) {
        if (dt[j] > 0.0) { // enter if there is any contribution only
          rc = 0.0;
          for (i = 1; i <= nd; i++) {
            // if(d[i*mxdim_p1+j]<TINY) d[i*mxdim_p1+j]=TINY;
            // if(d[i*mxdim_p1+j]<TINY) printf("d[%i]:%.15e\n", i*mxdim_p1+j,
            // d[i*mxdim_p1+j]); printf("d[%i]:%.15e\n", i*mxdim_p1+j,
            // d[i*mxdim_p1+j]);
            r[i] = pow((1.0 - d[i * mxdim_p1 + j] / dt[j]) /
                         (log(dt[j]) - log(d[i * mxdim_p1 + j])),
                       Internal_Vegas_Params::get_ALPH());
            rc += r[i]; // rc is it the total number of sub-increments
          }
          rebin(
            rc / xnd,
            nd,
            r,
            xin,
            xi.data() +
              (j * ndmx_p1)); // first bin of each dimension is at a diff index
        }
      }
    }

    Kokkos::deep_copy(d_xi, xi);
    for (it = itmax + 1; it <= titer && (*status) == 1; (*iters)++, it++) {

      ti = tsi = 0.0;
      Kokkos::deep_copy(d_result, 0.0);
      MilliSeconds time_diff = std::chrono::high_resolution_clock::now() - t0;
      unsigned int seed = /*static_cast<unsigned int>(time_diff.count()) +*/
                          static_cast<unsigned int>(it);
      
      vegas_kernel_kokkosF<IntegT, ndim, GeneratorType>(d_integrand,
                                                        params.nBlocks,
                                                        params.nThreads,
                                                        ng,
                                                        npg,
                                                        xjac,
                                                        dxg,
                                                        d_result,
                                                        xnd,
                                                        d_xi,
                                                        d_dx,
                                                        d_regn,
                                                        chunkSize,
                                                        totalNumThreads,
                                                        LastChunk,
                                                        seed + it);
      Kokkos::fence();
      Kokkos::deep_copy(result, d_result);

      ti = result(0);
      tsi = result(1);
      tsi *= dv2g;
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
      //printf("%i %e +- %e iteration: %e +- %e chi:%.15f\n",
      //       it, *tgral, *sd, ti, sqrt(tsi), *chi2a);
    } // end of iterations

    free(dt);
    free(r);
    free(xin);
    // TOTO check if we forget to free anything
  }

  template <typename IntegT,
            int NDIM,
            typename GeneratorType = typename kokkos_mcubes::Custom_generator>
  numint::integration_result
  integrate(IntegT ig,
            double epsrel,
            double epsabs,
            double ncall,
            quad::Volume<double, NDIM> const* volume,
            int totalIters = 15,
            int adjustIters = 15,
            int skipIters = 5)
  {

    numint::integration_result result;
    result.status = 1;
    vegas<IntegT, NDIM, GeneratorType>(ig,
                                       epsrel,
                                       epsabs,
                                       ncall,
                                       &result.estimate,
                                       &result.errorest,
                                       &result.chi_sq,
                                       &result.status,
                                       &result.iters,
                                       totalIters,
                                       adjustIters,
                                       skipIters,
                                       volume);
    return result;
  }

}
