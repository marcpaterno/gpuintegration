#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "cuda/pagani/quad/GPUquad/Func_Eval.cuh"

class GENZ_4_2D {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    // double alpha = 25.;
    double beta = .5;
    return exp(-1.0 *
               (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2)));
  }
};

int
main()
{
  double epsrel = 1.0e-3;
  double epsabs = 1.e-20;
  constexpr int ndim = 2;
  GENZ_4_2D integrand;
  double true_value = 0.00502655;

  using Reg_estimates = Region_estimates<double, ndim>;
  using Sub_regs = Sub_regions<double, ndim>;
  using Regs_characteristics = Region_characteristics<ndim>;

  Workspace<double, ndim> workspace;
  quad::Volume<double, ndim> vol;

  size_t partitions_per_axis = 2;
  if (ndim < 5)
    partitions_per_axis = 4;
  else if (ndim <= 10)
    partitions_per_axis = 2;
  else
    partitions_per_axis = 1;

  Sub_regions<double, ndim> sub_regions(partitions_per_axis);
  sub_regions.uniform_split(partitions_per_axis);

  constexpr size_t fEvalPerRegion = pagani::CuhreFuncEvalsPerRegion<ndim>();
  quad::Rule<double> rule;
  const int key = 0;
  const int verbose = 0;
  Structures<double> constMem;
  rule.Init(ndim, fEvalPerRegion, key, verbose, &constMem);
  double* generators =
    cuda_malloc<double>(sizeof(double) * ndim * fEvalPerRegion);

  size_t block_size = 64;
  quad::ComputeGenerators<double, ndim>
    <<<1, block_size>>>(generators, fEvalPerRegion, constMem);

  double* integ_space_lows = cuda_malloc<double>(ndim);
  double* integ_space_highs = cuda_malloc<double>(ndim);

  cuda_memcpy_to_device<double>(integ_space_highs, &vol.highs[0], ndim);
  cuda_memcpy_to_device<double>(integ_space_lows, &vol.lows[0], ndim);

  GENZ_4_2D* d_integrand = make_gpu_integrand<GENZ_4_2D>(integrand);

  Reg_estimates estimates(sub_regions.size);
  Regs_characteristics region_characteristics(sub_regions.size);
  unsigned int seed = 4;
  std::cout << "Launching kernel with " << sub_regions.size << " regions"
            << std::endl;
  quad::Func_Evals<ndim> fevals;
  quad::VEGAS_ASSISTED_INTEGRATE_GPU_PHASE1<GENZ_4_2D, double, ndim, 64>
    <<<sub_regions.size, 64>>>(d_integrand,
                               sub_regions.dLeftCoord,
                               sub_regions.dLength,
                               sub_regions.size,
                               estimates.integral_estimates,
                               estimates.error_estimates,
                               region_characteristics.active_regions,
                               region_characteristics.sub_dividing_dim,
                               epsrel,
                               epsabs,
                               constMem,
                               integ_space_lows,
                               integ_space_highs,
                               generators,
                               fevals,
                               seed);
  cudaDeviceSynchronize();

  double mcubes_est =
    reduction<double>(estimates.integral_estimates, sub_regions.size);
  double mcubes_err =
    reduction<double>(estimates.error_estimates, sub_regions.size);

  double* h_regions_estimates =
    copy_to_host(estimates.integral_estimates, sub_regions.size);
  double* h_regions_errorests =
    copy_to_host(estimates.error_estimates, sub_regions.size);
  std::cout << "num_regions:" << sub_regions.size << std::endl;
  for (int i = 0; i < sub_regions.size; ++i) {
    std::cout << "region " << i << ":" << h_regions_estimates[i] << "+-"
              << h_regions_errorests[i] << std::endl;
  }

  quad::Func_Evals<ndim> fevals_vanilla;
  quad::INTEGRATE_GPU_PHASE1<GENZ_4_2D, double, ndim, 64>
    <<<sub_regions.size, 64>>>(d_integrand,
                               sub_regions.dLeftCoord,
                               sub_regions.dLength,
                               sub_regions.size,
                               estimates.integral_estimates,
                               estimates.error_estimates,
                               region_characteristics.active_regions,
                               region_characteristics.sub_dividing_dim,
                               epsrel,
                               epsabs,
                               constMem,
                               integ_space_lows,
                               integ_space_highs,
                               generators,
                               fevals_vanilla);
  cudaDeviceSynchronize();

  double pagani_est =
    reduction<double>(estimates.integral_estimates, sub_regions.size);
  double pagani_err =
    reduction<double>(estimates.error_estimates, sub_regions.size);

  std::cout << "true value:" << true_value << std::endl;
  std::cout << "pagani:" << pagani_est << "+-" << pagani_err << std::endl;
  std::cout << "mcubes:" << mcubes_est << "+-" << mcubes_err << std::endl;

  h_regions_estimates =
    copy_to_host(estimates.integral_estimates, sub_regions.size);
  h_regions_errorests =
    copy_to_host(estimates.error_estimates, sub_regions.size);
  std::cout << "num_regions:" << sub_regions.size << std::endl;
  for (int i = 0; i < sub_regions.size; ++i) {
    std::cout << "region " << i << ":" << h_regions_estimates[i] << "+-"
              << h_regions_errorests[i] << std::endl;
  }
  return 0;
}