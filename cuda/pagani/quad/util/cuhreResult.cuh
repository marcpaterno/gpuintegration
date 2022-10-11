#error Include common/integration_result.hh instead
#ifndef CUDA_PAGANI_QUAD_UTIL_CUHRE_RESULT_CUH
#define CUDA_PAGANI_QUAD_UTIL_CUHRE_RESULT_CUH

#include <ostream>

struct cuhreResult {

  double estimate = 0.;
  double errorest = 0.;
  size_t neval = 0;
  size_t nregions = 0;
  size_t nFinishedRegions = 0;
  int status = -1;
  int lastPhase = -1;
  double chi_sq = 0.;
  size_t iters = 0;
};

inline
std::ostream&
operator<<(std::ostream& os, cuhreResult const& res)
{
  os << res.estimate << "," << res.errorest << "," << res.nregions << ","
     << res.chi_sq << "," << res.status;
  return os;
}

#endif
