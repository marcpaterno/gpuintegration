#ifndef INTEGRAND_CUH
#define INTEGRAND_CUH

#include <array>
#include <vector>

#include "cuda/pagani/quad/util/Volume.cuh"
#include "cuda/pagani/quad/util/cudaApply.cuh"
#include "cuda/pagani/quad/util/cudaArray.cuh"
#include "cuda/pagani/quad/quad.h"
#include "cuda/mcubes/util/func.cuh"
#include "cuda/mcubes/util/util.cuh"
#include "cuda/mcubes/util/vegas_utils.cuh"
#include "cuda/mcubes/util/verbose_utils.cuh"
#include "cuda/mcubes/seqCodesDefs.hh"

// no stratified samples, we don't care about intervals and sub-cubes, just have
// each thread draw randomly from a bin
template <typename IntegT,
          int ndim,
          typename GeneratorType = Curand_generator,
          size_t num_bins>
__global__
cuda_generate_random_points(unsigned int seed_init,
                            double* funcevals,
                            double* random_points,
                            const double* const dx,
                            const double* const regn,
                            size_t num_random_points,
                            double xjac)
{

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_random_points) {

    constexpr int ndmx1 = Internal_Vegas_Params::get_NDMX_p1();
    constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();

    size_t bin_ids[ndim + 1] = {
      0.}; //+1 because all mcubes code ignores index 0
    gpu::cudaArray<double, ndim> x;
    int ia[mxdim_p1];

    Random_num_generator<GeneratorType> rand_num_generator(seed_init);

    double wgt = xjac;
    for (int j = 1; j <= ndim; j++) {
      const double ran00 = (*rand_num_generator)();
      const size_t random_bin = static_cast<size_t>(ran00 * num_bins) +
                                1; //+1 because there is no zero bin
      bin_ids[j] = IMAX(IMIN(static_cast<int>(random_bin), ndmx), 1);
      double rc = 0., xo = 0.;

      if (ia[j] > 1) {
        xo = (xi[j * ndmx1 + ia[j]]) - (xi[j * ndmx1 + ia[j] - 1]); // bin
                                                                    // length
        rc = (xi[j * ndmx1 + ia[j] - 1]) +
             (xn - ia[j]) * xo; // scaling ran00 to bin bounds
      } else {
        xo = (xi[j * ndmx1 + ia[j]]);
        rc = (xn - ia[j]) * xo;
      }

      x[j - 1] = regn[j] + rc * (dx[j]);
      wgt *= xo * xnd;
      // random_points[offset] = x[j-1];
    }
    const double tmp = gpu::apply(*d_integrand, xx);
    const double f = wgt * tmp;
    // funcevals[other_offset] = f;
  }
}

template <int ndim>
struct Eval_Point {
  std::array<double, ndim> values = {0.};
  double
  operator()(size_t index)
  {
    return values[index];
  }
};

template <int dim>
class Vegas_state {
  const size_t num_bins_per_dim = 500;
  const size_t max_dim = 20;

  std::array<double, (num_bins_per_dim + 1) * (max_dim + 1)> bin_right_coord;
  std::vector<Eval_Point<dim>> random_points;

public:
  Integrator(double* bins) : bin_right_coord.data()(bins);
  std::vector<Eval_Point<ndim>> generate_random_points();
  generate_points_points_and_values();

  template <class IntegT>
  void Setup(IntegT integrand);
};

std::vector<Eval_Point<ndim>>
Vegas_state::generate_random_points(size_t num_points)
{

  std::vector<Eval_Point<ndim>> random_points;
  random_points.reserve(num_points);

  Eval_Point<ndim>* d_points;
  cudaMalloc((void**)&d_points, sizeof(Eval_Point<ndim>) * num_points);
}

/*
   This function will complete some mcubes iterations and generate a state
   consisting of the bin coordinates Based on those bin coordinates, we can then
   execute another function to return a set of random points and/or function
   evaluations
*/
template <typename IntegT,
          int ndim,
          bool DEBUG_MCUBES = false,
          typename GeneratorType = typename ::Curand_generator>
Vegas_state<ndim>
Generate_state(IntegT integrand,
               double ncall,
               int titer,
               quad::Volume<double, ndim> const* vol)
{
  // all of the ofstreams below will be removed, replaced by DataLogger
  auto t0 = std::chrono::high_resolution_clock::now();

  constexpr int ndmx = Internal_Vegas_Params::get_NDMX();
  constexpr int ndmx_p1 = Internal_Vegas_Params::get_NDMX_p1();
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

  d = (double*)malloc(sizeof(double) * (ndmx_p1) * (mxdim_p1)); // contributions
  dt = (double*)malloc(sizeof(double) * (mxdim_p1));            // for cpu-only
  dx = (double*)malloc(sizeof(double) *
                       (mxdim_p1)); // length of integ-space at each dim
  r = (double*)malloc(sizeof(double) * (ndmx_p1));
  x = (double*)malloc(sizeof(double) * (mxdim_p1));
  xi =
    (double*)malloc(sizeof(double) * (mxdim_p1) * (ndmx_p1)); // right bin coord
  xin = (double*)malloc(sizeof(double) * (ndmx_p1));
  ia = (int*)malloc(sizeof(int) * (mxdim_p1));

  // code works only  for (2 * ng - NDMX) >= 0)

  ndo = 1;
  for (j = 1; j <= ndim; j++) {
    xi[j * ndmx_p1 + 1] =
      1.0; // this index is the first for each bin for each dimension
  }

  si = swgt = schi = 0.0;
  nd = ndmx;
  ng = 1;
  ng = (int)pow(ncall / 2.0 + 0.25, 1.0 / ndim); // why do we add .25?
  for (k = 1, i = 1; i < ndim; i++) {
    k *= ng;
  }

  double sci = 1.0 / k; // I dont' think that's used anywhere
  double sc = k;        // I dont' think that's used either
  k *= ng;
  ncubes = k;

  npg = IMAX(ncall / k, 2);
  // assert(npg == Compute_samples_per_cube(ncall, ncubes)); //to replace line
  // directly above assert(ncubes == ComputeNcubes(ncall, ndim)); //to replace
  // line directly above

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
  for (j = 1; j <= ndim; j++) {
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

  int chunkSize = GetChunkSize(ncall);

  uint32_t totalNumThreads =
    (uint32_t)((ncubes /*+ chunkSize - 1*/) / chunkSize);

  uint32_t totalCubes = totalNumThreads * chunkSize; // even-split cubes
  int extra = ncubes - totalCubes;                   // left-over cubes
  int LastChunk = extra + chunkSize; // last chunk of last thread

  uint32_t nBlocks =
    ((uint32_t)(((ncubes + BLOCK_DIM_X - 1) / BLOCK_DIM_X)) / chunkSize) +
    1; // compute blocks based on chunk_size, ncubes, and block_dim_x
  uint32_t nThreads = BLOCK_DIM_X;

  IterDataLogger<DEBUG_MCUBES> data_collector(
    totalNumThreads, chunkSize, extra, npg, ndim);

  for (it = 1; it <= itmax && (*status) == 1; (*iters)++, it++) {
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

    using MilliSeconds =
      std::chrono::duration<double, std::chrono::milliseconds::period>;

    MilliSeconds time_diff = std::chrono::high_resolution_clock::now() - t0;
    unsigned int seed = static_cast<unsigned int>(time_diff.count()) +
                        static_cast<unsigned int>(it);
    vegas_kernel<IntegT, ndim, DEBUG_MCUBES, GeneratorType>
      <<<nBlocks, nThreads>>>(d_integrand,
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
                              seed + it,
                              data_collector.randoms,
                              data_collector.funcevals);

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

    if constexpr (DEBUG_MCUBES == true) {
      data_collector.PrintBins(it, xi, d, ndim);
      data_collector.PrintIterResults(it, *tgral, *sd, *chi2a, ti, tsi);
    }

    // replace above with datalogger.print();
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
          &xi[j * ndmx_p1]); // first bin of each dimension is at a diff index
      }
    }

  } // end of iterations

  Vegas_state<ndim> state(xi);
}

#endif