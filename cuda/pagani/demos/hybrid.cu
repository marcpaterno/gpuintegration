#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "cuda/pagani/quad/GPUquad/Func_Eval.cuh"

class GENZ_4_5D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double w, double v)
  {
    // double alpha = 25.;
    double beta = .5;
    return exp(-1.0 *
               (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2) +
                pow(25, 2) * pow(z - beta, 2) + pow(25, 2) * pow(w - beta, 2) +
                pow(25, 2) * pow(v - beta, 2)));
  }
};

int
main()
{
  double epsrel = 1.0e-3;
  double epsabs = 1.e-20;
  GENZ_4_5D integrand;
  double true_value = 1.79132603674879e-06;

  std::cout << integrand(.5, 0., 0., 0., .0) << std::endl;
  std::cout << integrand(.5, 0.5, 0.5, 0.5, .50) << std::endl;
  /*std::cout<< integrand(.499, .4999, .4999, .4999, .1) << std::endl;
  std::cout<< integrand(.6, .6, .6, .6, .6) << std::endl;

  std::cout<< integrand(.4, .4, .4, .4, .4) << std::endl;

  std::cout<< integrand(.2, .2, .2, .2, .2) << std::endl;
  std::cout<< integrand(.7, .7, .7, .7, .7) << std::endl;
  std::cout<< integrand(.9, .9, .9, .9, .9) << std::endl;
  std::cout<<"end\n";
  */

  int const ndim = 5;
  using Reg_estimates = Region_estimates<ndim>;
  using Sub_regs = Sub_regions<ndim>;
  using Regs_characteristics = Region_characteristics<ndim>;

  Workspace<ndim> workspace;
  quad::Volume<double, ndim> vol;

  size_t partitions_per_axis = 2;
  if (ndim < 5)
    partitions_per_axis = 4;
  else if (ndim <= 10)
    partitions_per_axis = 2;
  else
    partitions_per_axis = 1;

  Sub_regions<ndim> sub_regions(partitions_per_axis);
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

  GENZ_4_5D* d_integrand = make_gpu_integrand<GENZ_4_5D>(integrand);

  Reg_estimates estimates(sub_regions.size);
  Regs_characteristics region_characteristics(sub_regions.size);
  unsigned int seed = 4;
  std::cout << "Launching kernel with " << sub_regions.size << " regions"
            << std::endl;
  quad::Func_Evals<ndim> fevals;
  quad::VEGAS_ASSISTED_INTEGRATE_GPU_PHASE1<GENZ_4_5D, double, ndim, 64>
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

  quad::Func_Evals<ndim> fevals_vanilla;
  quad::INTEGRATE_GPU_PHASE1<GENZ_4_5D, double, ndim, 64>
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
  return 0;
}
