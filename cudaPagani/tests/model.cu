#include "cudaPagani/quad/GPUquad/Pagani.cuh"
#include "cudaPagani/quad/quad.h"
#include "cudaPagani/quad/util/cudaArray.cuh"
#include "cudaPagani/quad/util/cudaMemoryUtil.h"
#include "cudaPagani/tests/model.cuh"

namespace gpu{
    
    class EZ_sq {
    public:
      EZ_sq() = default;

      EZ_sq(double omega_m, double omega_l, double omega_k)
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

    class EZ {
    public:
      EZ() = default;

      EZ(double omega_m, double omega_l, double omega_k)
        : _ezsq(omega_m, omega_l, omega_k)
      {}

      __device__ double
      operator()(double z) const
      {
        auto const sqr = _ezsq(z);
        return sqrt(sqr);
      }

    private:
      EZ_sq _ezsq;
    };

    class DV_DO_DZ_t {
    public:
      DV_DO_DZ_t() = default;

      DV_DO_DZ_t(EZ ezt, double h) : /*_da(da),*/ _ezt(ezt), _h(h)
      {}
        
      __device__ double
      operator()(double zt) const
      {
        //double const da_z = _da(zt); // da_z needs to be in Mpc
        // Units: (Mpc/h)^3
        // 2997.92 is Hubble distance, c/H_0
        return 2997.92 * (1.0 + zt) * (1.0 + zt)  * _h  * _h / _ezt(zt);
      }

    private:
      //quad::Interp1D _da;
      EZ _ezt;
      double _h = 0.0;
    };
}



template<typename Model>
__global__ 
void
Evaluate(const Model* model,
         const gpu::cudaDynamicArray<double>& input, //make const ref
         const size_t size,
         gpu::cudaDynamicArray<double> results)
{
  
  for (size_t i = 0; i < size; i++) {
    results[i] = (*model)(input[i]);
  }
}

template<typename Model, size_t arraySize>
std::array<double, arraySize>
Compute_GPU_model(const Model& model, const std::array<double, arraySize>& input){
    //must change name of cudaDynamicArray, it's really cudaUnifiedArray
    
    Model* ptr_to_thing_in_unified_memory = quad::cuda_copy_to_managed(model);
    gpu::cudaDynamicArray<double> unified_input(input.data(), arraySize);
    gpu::cudaDynamicArray<double> unified_output(arraySize);
    
    Evaluate<Model><<<1,1>>>(ptr_to_thing_in_unified_memory, unified_input, arraySize, unified_output);
    cudaDeviceSynchronize();
    
    std::array<double, arraySize> results;
    for(size_t i = 0; i < arraySize; ++i)
        results[i] = unified_output[i];
    
    cudaFree(ptr_to_thing_in_unified_memory);
    return results;
}

double* gpuExecute(){
    std::array<double, 10> zt_poitns = {0.156614, 0.239091, 0.3, 0.360909, 0.443386, 0.456614, 0.539091, 0.6, 0.660909, 0.743386};
    constexpr size_t arraySize = zt_poitns.size();
    std::array<double, 10> results;
    
    gpu::EZ ezt(4.15, 3.13, 9.9);
    gpu::DV_DO_DZ_t dv_do_dz_t(ezt, 7.4);
        
    results = Compute_GPU_model<gpu::DV_DO_DZ_t, arraySize>(dv_do_dz_t, zt_poitns);
    for(auto i : results)
        std::cout<<i<<"\n";
    return results.data();
}
