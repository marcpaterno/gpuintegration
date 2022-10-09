#ifndef ALTERNATIVE_TIME_AND_CALL_CUH
#define ALTERNATIVE_TIME_AND_CALL_CUH

#include <chrono>
#include "cuda/pagani/quad/GPUquad/PaganiUtils.cuh"
#include "cuda/pagani/quad/GPUquad/Workspace.cuh"
#include "cuda/pagani/quad/util/cuhreResult.cuh"
#include "cuda/pagani/quad/util/Volume.cuh"
#include "nvToolsExt.h"
#include <string>

template <typename F,
          typename T,
          int ndim,
          bool use_custom = false,
          int debug = 0>
bool
clean_time_and_call(std::string id,
                    F integrand,
                    T epsrel,
                    T true_value,
                    char const* algname,
                    std::ostream& outfile,
                    quad::Volume<T, ndim>& vol,
                    bool relerr_classification = true)
{

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  T constexpr epsabs = 1.0e-40;
  bool good = false;
  Workspace<T, ndim, use_custom> workspace;

  auto print_custom = [=](bool use_custom_flag) {
    std::string to_print = use_custom_flag == true ? "custom" : "library";
    return to_print;
  };

  for (int i = 0; i < 3; i++) {

    auto const t0 = std::chrono::high_resolution_clock::now();
    size_t partitions_per_axis = 2;
    if (ndim < 5)
      partitions_per_axis = 4;
    else if (ndim <= 10)
      partitions_per_axis = 2;
    else
      partitions_per_axis = 1;

    Sub_regions<T, ndim> sub_regions(partitions_per_axis);
    sub_regions.uniform_split(partitions_per_axis);

    constexpr bool predict_split = false;
    constexpr bool collect_iters = false;

    cuhreResult<T> result =
      workspace.template integrate<F, predict_split, collect_iters, debug>(
        integrand, sub_regions, epsrel, epsabs, vol, relerr_classification);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
    T const absolute_error = std::abs(result.estimate - true_value);

    if (result.status == 0) {
      good = true;
    }

    outfile.precision(17);
    if (i != 0)
      outfile << std::fixed << std::scientific << id << "," << ndim << ","
              << print_custom(use_custom) << "," << true_value << "," << epsrel
              << "," << epsabs << "," << result.estimate << ","
              << result.errorest << "," << result.nregions << ","
              << result.status << "," << dt.count() << std::endl;
  }
  return good;
}

void
print_header()
{
  std::cout << "id, ndim, use_custom, integral, epsrel, epsabs, estimate, "
               "errorest, nregions, status, time\n";
}

#endif