#ifndef ONE_API_TWO_LEVE_ERROR_ESTIMATE_H
#define ONE_API_TWO_LEVE_ERROR_ESTIMATE_H

template <typename T>
void
RefineError(sycl::queue& q,
            T* dRegionsIntegral,
            T* dRegionsError,
            T* dParentsIntegral,
            T* dParentsError,
            T* newErrs,
            double* activeRegions,
            size_t currIterRegions,
            T epsrel,
            int heuristicID)
{
  q.submit([&](auto& cgh) {
     cgh.parallel_for(sycl::range<1>(currIterRegions), [=](sycl::id<1> gtid) {
       size_t tid = static_cast<size_t>(gtid);
       T selfErr = dRegionsError[tid];
       T selfRes = dRegionsIntegral[tid];

       size_t inRightSide = (2 * tid >= currIterRegions);
       size_t inLeftSide = (0 >= inRightSide);
       size_t siblingIndex = tid + (inLeftSide * currIterRegions / 2) -
                             (inRightSide * currIterRegions / 2);
       size_t parIndex = tid - inRightSide * (currIterRegions * .5);

       T siblErr = dRegionsError[siblingIndex];
       T siblRes = dRegionsIntegral[siblingIndex];

       T parRes = dParentsIntegral[parIndex];
       // T parErr = dParentsError[parIndex];

       T diff = siblRes + selfRes - parRes;
       diff = fabs(.25 * diff);

       T err = selfErr + siblErr;

       if (err > 0.0) {
         T c = 1 + 2 * diff / err;
         selfErr *= c;
       }

       selfErr += diff;

       newErrs[tid] = selfErr;
       int PassRatioTest =
         heuristicID != 1 && selfErr < MaxErr(selfRes, epsrel, 1e-200);
       activeRegions[tid] = static_cast<double>(!(PassRatioTest));
     });
   })
    .wait();
}

template <size_t ndim>
void
two_level_errorest_and_relerr_classify(
  sycl::queue& q,
  Region_estimates<ndim>& current_iter_raw_estimates,
  const Region_estimates<ndim>& prev_iter_two_level_estimates,
  const Region_characteristics<ndim>& reg_classifiers,
  double epsrel,
  bool relerr_classification = true)
{

  bool forbid_relerr_classification = !relerr_classification;
  if (prev_iter_two_level_estimates.size == 0) {
    return;
  }

  double* new_two_level_errorestimates =
    sycl::malloc_device<double>(reg_classifiers.size, q);
  RefineError<double>(q,
                      current_iter_raw_estimates.integral_estimates,
                      current_iter_raw_estimates.error_estimates,
                      prev_iter_two_level_estimates.integral_estimates,
                      prev_iter_two_level_estimates.error_estimates,
                      new_two_level_errorestimates,
                      reg_classifiers.active_regions,
                      reg_classifiers.size,
                      epsrel,
                      forbid_relerr_classification);

  sycl::free(current_iter_raw_estimates.error_estimates, q);
  current_iter_raw_estimates.error_estimates = new_two_level_errorestimates;
}

#endif
