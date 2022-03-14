#include "cuda/pagani/quad/GPUquad/Pagani.cuh"
#include "cuda/pagani/quad/quad.h"
#include "cuda/pagani/quad/util/cudaArray.cuh"
#include "cuda/pagani/quad/util/cudaMemoryUtil.h"
#include "cuda/pagani/tests/model.cuh"

namespace gpu {

  class EZ_sq_simplified {
  public:
    EZ_sq_simplified() = default;

    EZ_sq_simplified(double omega_m, double omega_l, double omega_k)
      : _omega_m(omega_m), _omega_l(omega_l), _omega_k(omega_k)
    {}

    __device__ double
    operator()(double z) const
    {
      // NOTE: this is valid only for \Lambda CDM cosmology, not wCDM
      double const zplus1 = 1.0 + z;
      return (_omega_m * zplus1 * zplus1 * zplus1 + _omega_k * zplus1 * zplus1 +
              _omega_l);
    }

  private:
    double _omega_m = 0.0;
    double _omega_l = 0.0;
    double _omega_k = 0.0;
  };

  class EZ_simplified {
  public:
    EZ_simplified() = default;

    EZ_simplified(double omega_m, double omega_l, double omega_k)
      : _ezsq(omega_m, omega_l, omega_k)
    {}

    __device__ double
    operator()(double z) const
    {
      auto const sqr = _ezsq(z);
      return sqrt(sqr);
    }

  private:
    EZ_sq_simplified _ezsq;
  };

  class DV_DO_DZ_t_simplified {
  public:
    DV_DO_DZ_t_simplified() = default;

    DV_DO_DZ_t_simplified(EZ_simplified ezt, double h)
      : /*_da(da),*/ _ezt(ezt), _h(h)
    {}

    __device__ double
    operator()(double zt) const
    {
      // double const da_z = _da(zt); // da_z needs to be in Mpc
      // Units: (Mpc/h)^3
      // 2997.92 is Hubble distance, c/H_0
      // return 2997.92 * (1.0 + zt) * (1.0 + zt)  * sqrt(_h)  * erf(_h) /
      // _ezt(zt);
      return 2997.92 * (1.0 + zt) * (1.0 + zt) * (_h) * (_h) / _ezt(zt);
    }

  private:
    // quad::Interp1D _da;
    EZ_simplified _ezt;
    double _h = 0.0;
  };
}

template <typename Model>
__global__ void
Evaluate(const Model* model,
         const gpu::cudaDynamicArray<double> input,
         const size_t size,
         double* results)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t n_threads = blockDim.x * gridDim.x;

  // Loop over the part of the array this is handled in 
  // conviently-sized blocks.
  size_t i = 0;
  for ( /*already initialized*/ ; i*n_threads < size; ++i) {
    auto idx = tid + i * n_threads;
    results[idx] = (*model)(input[idx]);
  }
 
  // Clean up the rest of the range.
  auto idx = tid +i * n_threads;
  if (idx < size) results[idx] = (*model)(input[idx]);
}

template <typename Model>
std::vector<double>
Compute_GPU_model(const Model& model, const std::vector<double>& input)
{
  // must change name of cudaDynamicArray, it's really cudaUnifiedArray
  Model* ptr_to_thing_in_unified_memory = quad::cuda_copy_to_managed(model);
  gpu::cudaDynamicArray<double> unified_input(input.data(), input.size());

  double* unified_output = quad::cuda_malloc_managed<double>(input.size());

  // TODO: How do we pick a good number of threads to run? Good number of
  // blocks? What would make the testing most robust?
  Evaluate<Model><<<2, 3>>>(ptr_to_thing_in_unified_memory,
                            unified_input,
                            input.size(),
                            unified_output);
  cudaDeviceSynchronize();

  std::vector<double> results;
  results.reserve(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    results.push_back(unified_output[i]);
  }

  cudaFree(ptr_to_thing_in_unified_memory);
  cudaFree(unified_output);
  return results;
}

std::vector<double>
gpuExecute()
{
  std::vector<double> zt_poitns = {0.156614,
                                   0.239091,
                                   0.3,
                                   0.360909,
                                   0.443386,
                                   0.456614,
                                   0.539091,
                                   0.6,
                                   0.660909,
                                   0.743386};
  std::vector<double> results;

  gpu::EZ_simplified ezt(4.15, 3.13, 9.9);
  gpu::DV_DO_DZ_t_simplified dv_do_dz_t(ezt, 7.4);

  results =
    Compute_GPU_model<gpu::DV_DO_DZ_t_simplified>(dv_do_dz_t, zt_poitns);
  return results;
}
