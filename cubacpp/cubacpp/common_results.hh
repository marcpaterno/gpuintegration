#ifndef CUBACPP_ABS_RESULTS_HH
#define CUBACPP_ABS_RESULTS_HH

namespace cubacpp {

  // common_results contains the common part of all integration results. It is
  // intended to be used as a base for other results classes, so that they
  // will each contain the same set of data members.
  //
  //   neval is the number of function evaluations performed.
  //   nregions is tne number of regions into which the integration volume has
  //            been subdivided. Not all algorithms can report this.
  //   status indicates convergence.
  //             -1 (default) means no calculated was attempted.
  //              0           the integration has converged
  //              1           the integration terminated without convergence.
  //
  // NOTE: common_results has no virtual functions (including the destructor),
  //       and is *not* intended to be used polymorphically.
  //
  struct common_results {

    long long neval = 0;
    int nregions = -1;
    int status = -1;

    bool
    converged() const
    {
      return status == 0;
    }

    common_results(long long neval, int nregions, int status)
      : neval(neval), nregions(nregions), status(status)
    {}

    common_results() = default;
  };

}

#endif // CUBACPP_ABS_RESULTS_HH
