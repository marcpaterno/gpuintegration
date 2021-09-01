#ifndef CUHRE_RESULT_CUH
#define CUHRE_RESULT_CUH

template <typename T>
struct cuhreResult {

  cuhreResult()
  {
    estimate = 0.;
    errorest = 0.;
    neval = 0.;
    nregions = 0.;
    status = 0.;
    // activeRegions = 0.;
    phase2_failedblocks = 0.;
    lastPhase = 0;
    nFinishedRegions = 0;
  };

  T estimate;
  T errorest;
  size_t neval;
  size_t nregions;
  size_t nFinishedRegions;
  int status;
  int lastPhase;
  // size_t activeRegions;    // is not currently being set
  size_t phase2_failedblocks; // is not currently being set
  double chi_sq = 0.;
};


#endif
