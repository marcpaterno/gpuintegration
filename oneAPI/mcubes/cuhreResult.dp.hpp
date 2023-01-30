#ifndef CUHRE_RESULT_CUH
#define CUHRE_RESULT_CUH

#include <ostream>

template <typename T>
struct cuhreResult {

  T estimate = 0.;
  T errorest = 0.;
  size_t neval = 0;
  size_t nregions = 0;
  size_t nFinishedRegions = 0;
  int status = -1;
  int lastPhase = -1;
  size_t phase2_failedblocks = 0; // is not currently being set
  double chi_sq = 0.;
  size_t iters = 0;
};

template <typename T>
std::ostream&
operator<<(std::ostream& os, cuhreResult<T> const& res)
{
  os << res.estimate << "," << res.errorest << "," << res.nregions << ","
     << res.chi_sq << "," << res.status;
  return os;
}

#endif