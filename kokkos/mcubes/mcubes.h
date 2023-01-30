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

#define WARP_SIZE 32
#define BLOCK_DIM_X 128
#define ALPH 1.5
#define NDMX 500
#define MXDIM 20

#define NDMX1 NDMX + 1
#define MXDIM1 MXDIM + 1
#define TINY 1.0e-30

#define PI 3.14159265358979323846
#define DEBUG 0
#define CUSTOM
// #define KOKKOS_ENABLE_CUDA_LAMBDA
// #define KOKKOS_ENABLE_CUDA
#include <chrono>

using MilliSeconds =
  std::chrono::duration<double, std::chrono::milliseconds::period>;
typedef Kokkos::View<int*> ViewVectorInt;
typedef Kokkos::View<float*> ViewVectorFloat;
typedef Kokkos::View<double*> ViewVectorDouble;

typedef Kokkos::TeamPolicy<> team_policy;
typedef Kokkos::TeamPolicy<>::member_type member_type;

typedef Kokkos::TeamPolicy<> team_policy;
typedef Kokkos::TeamPolicy<>::member_type member_type;

typedef Kokkos::View<double*,
                     Kokkos::DefaultExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  ScratchViewDouble;

// int scratch_size = ScratchViewType::shmem_size(256);

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

template <typename T, int NDIM>
struct Volume {

  T lows[NDIM] = {0.0};
  T highs[NDIM];

  Volume()
  {
    for (T& x : highs)
      x = 1.0;
  }

  Volume(std::array<T, NDIM> l, std::array<T, NDIM> h)
  {
    std::memcpy(lows, l.data(), NDIM * sizeof(T));
    std::memcpy(highs, h.data(), NDIM * sizeof(T));
  }

  Volume(T const* l, T const* h)
  {
    std::memcpy(lows, l, NDIM * sizeof(T));
    std::memcpy(highs, h, NDIM * sizeof(T));
  }
};

namespace gpu {
  template <typename T, std::size_t s>
  class cudaArray {
  public:
    const __device__ T*
    begin() const
    {
      return &data[0];
    }

    const __device__ T*
    end() const
    {
      return (&data[0] + s);
    }

    constexpr __device__ std::size_t
    size() const
    {
      return s;
    }

    __device__ T&
    operator[](std::size_t i)
    {
      return data[i];
    }

    __device__ T const&
    operator[](std::size_t i) const
    {
      return data[i];
    }

    T data[s];
  };

  namespace detail {
    template <class F, size_t N, std::size_t... I>
    __device__ double
    apply_impl(F&& f,
               gpu::cudaArray<double, N> const& data,
               std::index_sequence<I...>)
    {
      return f(data[I]...);
    };
  }

  template <class F, size_t N>
  __device__ double
  // Unsure if we need to pass 'f' by value, for GPU execution
  apply(F&& f, gpu::cudaArray<double, N> const& data)
  {
    return detail::apply_impl(
      std::forward<F>(f), data, std::make_index_sequence<N>());
  }
}

class Internal_Vegas_Params {
  static constexpr int ndmx = 500;
  static constexpr int mxdim = 20;
  static constexpr double alph = 1.5;

public:
  static constexpr int
  get_NDMX()
  {
    return ndmx;
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

  constexpr static int
  get_MXDIM_p1()
  {
    return mxdim + 1;
  }
};

__inline__ __host__ __device__ void
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
  Custom_generator(uint32_t seed, int blockId, int threadId)
    : custom_seed(seed), block_id(blockId), thread_id(threadId){};

  __device__ double
  operator()()
  {
    temp = a * custom_seed + c;
    custom_seed = temp & (p - 1);
    return (double)custom_seed / (double)p;
  }

  __device__ void
  SetSeed(uint32_t seed)
  {
    custom_seed = seed;
  }
};

class Curand_generator {
  curandState localState;

public:
  __device__
  Curand_generator(int blockId, int threadId)
  {
#if defined(__CUDA_ARCH__)
    curand_init(0, blockId, threadId, &localState);
#endif
  }

  __device__
  Curand_generator(unsigned int seed, int blockId, int threadId)
  {
#if defined(__CUDA_ARCH__)
    curand_init(seed, blockId, threadId, &localState);
#endif
  }

  __device__ double
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
  __device__
  Random_num_generator(unsigned int seed, int blockId, int threadId)
    : generator(seed, blockId, threadId)
  {}

  __device__ double
  operator()()
  {
    return generator();
  }
};

namespace mcubes {

  template <typename T, typename U>
  __inline__ __device__ constexpr bool
  is_same()
  {
    return false;
  }

  template <typename Custom_generator>
  __inline__ __device__ constexpr bool
  is_same<Custom_generator, Custom_generator>()
  {
    return true;
  }

  template <typename T, typename U>
  struct TypeChecker {
    static constexpr __device__ bool
    is_custom_generator()
    {
      return false;
    }
  };

  template <typename Custom_generator>
  struct TypeChecker<Custom_generator, Custom_generator> {
    static constexpr __device__ bool
    is_custom_generator()
    {
      return true;
    }
  };
}

template <typename T>
struct Result {
  T estimate = 0.;
  T errorest = 0.;
  int status = -1;
  double chi_sq = 0.;
  size_t iters = 0;
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

__host__ __device__ int
GetChunkSize(const double ncall)
{
  double small = 1.e7;
  double large = 8.e9;

  if (ncall <= small)
    return 32;
  else if (ncall <= large)
    return 2048;
  else
    return 4096;
}

template <int ndim, typename GeneratorType = Curand_generator>
__device__ void
Setup_Integrand_Eval(Random_num_generator<GeneratorType>* rand_num_generator,
                     double xnd,
                     double dxg,
                     ViewVectorDouble xi,
                     ViewVectorDouble regn,
                     ViewVectorDouble dx,
                     const uint32_t* const kg,
                     int* const ia,
                     double* const x,
                     double& wgt,
                     int npg,
                     int sampleID,
                     uint32_t cube_id)
{
  constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
  constexpr int ndmx1 = Internal_Vegas_Params::get_NDMX_p1();

  for (int j = 1; j <= ndim; j++) {
    const double ran00 = (*rand_num_generator)();
    const double xn = (kg[j] - ran00) * dxg + 1.0;
    double rc = 0., xo = 0.;
    ia[j] = IMAX(IMIN((int)(xn), ndmx), 1);

    if (ia[j] > 1) {
      xo = (xi[j * ndmx1 + ia[j]]) - (xi[j * ndmx1 + ia[j] - 1]); // bin length
      rc = (xi[j * ndmx1 + ia[j] - 1]) +
           (xn - ia[j]) * xo; // scaling ran00 to bin bounds
    } else {
      xo = (xi[j * ndmx1 + ia[j]]);
      rc = (xn - ia[j]) * xo;
    }

    x[j] = regn[j] + rc * (dx[j]);
    wgt *= xo * xnd; // xnd is number of bins, xo is the length of the bin, xjac
                     // is 1/num_calls
  }
}

template <typename IntegT, int ndim, typename GeneratorType = Curand_generator>
__device__ void
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
    Setup_Integrand_Eval<ndim, GeneratorType>(rand_num_generator,
                                              xnd,
                                              dxg,
                                              xi,
                                              regn,
                                              dx,
                                              kg,
                                              ia,
                                              x,
                                              wgt,
                                              npg,
                                              k,
                                              cube_id);

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

template <typename IntegT, int ndim, typename GeneratorType = Curand_generator>
__device__ void
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

    if constexpr (mcubes::TypeChecker<GeneratorType, Custom_generator>::
                    is_custom_generator()) {
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
      f2b = TINY;
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

template <typename IntegT, int ndim, typename GeneratorType = Curand_generator>
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
    "vegas_kernel",
    team_policy(nBlocks, nThreads).set_scratch_size(0, Kokkos::PerTeam(2048)),
    KOKKOS_LAMBDA(const member_type team_member) {
      int chunkSize = _chunkSize;
      ScratchViewDouble sh_buff(team_member.team_scratch(0),
                                2 * BLOCK_DIM_X * sizeof(double));

      constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();
      uint32_t tx = team_member.team_rank(); // local id
      uint32_t m =
        team_member.league_rank() * BLOCK_DIM_X + tx; // global thread id
      double wgt;
      uint32_t kg[mxdim_p1];
      int ia[mxdim_p1];
      double x[mxdim_p1];
      double fbg = 0., f2bg = 0.;

      if (m < totalNumThreads) {

        size_t cube_id_offset =
          (team_member.league_rank() * BLOCK_DIM_X + tx) * chunkSize;

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

      sh_buff(tx) = fbg;
      sh_buff(tx + BLOCK_DIM_X) = f2bg;

      team_member.team_barrier();

      if (tx == 0) {
        double fbgs = 0.0;
        double f2bgs = 0.0;
        for (int ii = 0; ii < BLOCK_DIM_X && (m + ii) < totalNumThreads; ++ii) {
          fbgs += sh_buff(ii);
          f2bgs += sh_buff(ii + BLOCK_DIM_X);
        }
        Kokkos::atomic_add(&result_dev(0), fbgs);
        Kokkos::atomic_add(&result_dev(1), f2bgs);
      }
    });
}

template <typename IntegT, int ndim, typename GeneratorType = Curand_generator>
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
    team_policy(nBlocks, nThreads).set_scratch_size(0, Kokkos::PerTeam(2048)),
    KOKKOS_LAMBDA(const member_type team_member) {
      int chunkSize = _chunkSize;
      ScratchViewDouble sh_buff(team_member.team_scratch(0), 256);

      uint32_t tx = team_member.team_rank();
      uint32_t m =
        team_member.league_rank() * BLOCK_DIM_X + tx; // global thread id

      double fb, f2b, wgt, xn, xo, rc, f, f2, ran00;
      uint32_t kg[mxdim_p1];
      int iaj;
      double x[mxdim_p1];
      int k;
      double fbg = 0., f2bg = 0.;

      if (m < totalNumThreads) {

        size_t cube_id_offset =
          (team_member.league_rank() * BLOCK_DIM_X + tx) * chunkSize;

        if (m == totalNumThreads - 1)
          chunkSize = LastChunk;
        Random_num_generator<GeneratorType> rand_num_generator(
          seed_init, team_member.league_rank(), team_member.team_rank());

        fbg = f2bg = 0.0;
        get_indx(cube_id_offset, &kg[1], ndim, ng);

        for (int t = 0; t < chunkSize; t++) {
          fb = f2b = 0.0;
          if constexpr (mcubes::TypeChecker<GeneratorType, Custom_generator>::
                          is_custom_generator()) {
            rand_num_generator->SetSeed(cube_id_offset + t);
          }
          for (k = 1; k <= npg; k++) {
            wgt = xjac;

            for (int j = 1; j <= ndim; j++) {

              ran00 = rand_num_generator();
              xn = (kg[j] - ran00) * dxg + 1.0;
              iaj = IMAX(IMIN((int)(xn), ndmx),
                         1); // this is the bin, different on each dim

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
            f2b = TINY;

          fbg += fb;
          f2bg += f2b;

          for (int k = ndim; k >= 1; k--) {

            kg[k] %= ng;

            if (++kg[k] != 1)
              break;
          }
        } // end of chunk for loop

      } // end of subcube if

      sh_buff(tx) = fbg;
      sh_buff(tx + BLOCK_DIM_X) = f2bg;

      team_member.team_barrier();

      if (tx == 0) {
        double fbgs = 0.0;
        double f2bgs = 0.0;
        for (int ii = 0; ii < BLOCK_DIM_X && m + ii < totalNumThreads; ++ii) {
          fbgs += sh_buff(ii);
          f2bgs += sh_buff(ii + BLOCK_DIM_X);
        }

        Kokkos::atomic_add(&result_dev(0), fbgs);
        Kokkos::atomic_add(&result_dev(1), f2bgs);
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
          typename GeneratorType = typename ::Curand_generator>
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
      Volume<double, ndim> const* vol)
{

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
    uint32_t nBlocks =
      ((uint32_t)(((ncubes + BLOCK_DIM_X - 1) / BLOCK_DIM_X)) / chunkSize) + 1;
    uint32_t nThreads = BLOCK_DIM_X;

    for (it = 1; it <= itmax && (*status) == 1; (*iters)++, it++) {

      ti = tsi = 0.0;
      for (j = 1; j <= ndim; j++) {
        for (i = 1; i <= nd; i++)
          d(i * mxdim_p1 + j) = 0.0;
      }

      deep_copy(d_xi, xi);
      uint32_t nBlocks =
        ((uint32_t)(((ncubes + BLOCK_DIM_X - 1) / BLOCK_DIM_X)) / chunkSize) +
        1;
      uint32_t nThreads = BLOCK_DIM_X;
      Kokkos::deep_copy(d_result, 0.0);
      // std::cout<<"npg:"<<npg<<", dxg:"<<dxg<<", ng:"<< ng << std::endl;
      MilliSeconds time_diff = std::chrono::high_resolution_clock::now() - t0;
      unsigned int seed = static_cast<unsigned int>(time_diff.count()) +
                          static_cast<unsigned int>(it);
      // seed = 0;
      vegas_kernel_kokkos<IntegT, ndim, GeneratorType>(d_integrand,
                                                       nBlocks,
                                                       nThreads,
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
        // printf("%i %.15f +- %.15f iteration: %.15f +- %.15f chi:%.15f\n",
        //     it, *tgral, *sd, ti, sqrt(tsi), *chi2a);
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
      unsigned int seed = static_cast<unsigned int>(time_diff.count()) +
                          static_cast<unsigned int>(it);
      seed = 0;
      vegas_kernel_kokkosF<IntegT, ndim, GeneratorType>(d_integrand,
                                                        nBlocks,
                                                        nThreads,
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

      Kokkos::deep_copy(result, d_result);

      ti = result(0);
      tsi = result(1);
      tsi *= dv2g;
      // printf("iter = %d  integ = %e   std = %e\n", it, ti, sqrt(tsi));

      wgt = 1.0 / tsi;
      si += wgt * ti;
      schi += wgt * ti * ti;
      swgt += wgt;
      *tgral = si / swgt;
      *chi2a = (schi - si * (*tgral)) / (it - 0.9999);

      if (*chi2a < 0.0)
        *chi2a = 0.0;

      *sd = sqrt(1.0 / swgt);
      // printf("%i, %.15f,  %.15f, %.15f, %.15f, %.15f\n", it, *tgral, *sd, ti,
      // sqrt(tsi), *chi2a);

      tsi = sqrt(tsi);
      *status = GetStatus(*tgral, *sd, it, epsrel, epsabs);
      // printf("it %d\n", it);
      // printf("z cummulative:%5d   %14.7g+/-%9.4g  %9.2g\n", it, *tgral, *sd,
      // *chi2a); printf("%3d   %e  %e\n", it, ti, tsi);

    } // end of iterations

    free(dt);
    free(r);
    free(xin);
    // TOTO check if we forget to free anything
  }
}

template <typename IntegT,
          int NDIM,
          typename GeneratorType = typename ::Curand_generator>
Result<double>
integrate(IntegT ig,
          double epsrel,
          double epsabs,
          double ncall,
          Volume<double, NDIM> const* volume,
          int totalIters = 15,
          int adjustIters = 15,
          int skipIters = 5)
{

  Result<double> result;
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
