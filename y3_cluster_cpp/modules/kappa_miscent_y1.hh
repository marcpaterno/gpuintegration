#include "utils/make_integration_volumes.hh"

#include "cubacpp/integration_result.hh"
#include "cubacpp/integration_volume.hh"

#include "models/angle_to_dist_t.hh"
#include "models/average_sci_t.hh"
#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/int_lc_lt_des_t.hh"
#include "models/int_zo_zt_des_t.hh"
#include "models/lo_lc_t.hh"
#include "models/mor_des_t.hh"
#include "models/omega_z_des.hh"
#include "models/roffset_t.hh"
#include "models/sig_sum.hh"

#include "Optional/optional.hpp"
#include <vector>

namespace y3_cluster {
  using std::experimental::optional;
}
using namespace y3_cluster;

// kappa_miscent_y1 is a class that models the concept of
// "CosmoSISVectorIntegrand", and is thus suitable for use as the template
// parameter for the class template CosmosisIntegrationModule.
//
// Notes:
//    1) std::optional<T> is used for data members that are not
//    constructible from CosmoSIS configuration parameters.
//
//    2) The object as created by the only constructor does not need to be
//    in a callable state.
//
//    3) After a call to set_sample has been made, the object must be in a
//    callable state.
//
//    4) State that *can* be correctly set by the constructor *should* be set by
//    the constructor. Do not needlessly repeat initialization in calls to
//    set_sample.
//
//
class kappa_miscent_y1 {
private:
  // We define the type alias volume_t to be the right dimensionality
  // of integration volume for our integrand. If we were to change the
  // number of arguments required by the function call operator (below),
  // we would need to also modify this type alias to keep consistent.
  using volume_t = cubacpp::IntegrationVolume<7>;

  // State obtained from each sample.
  // If there were a type X that did not have a default constructor,
  // we would use std::optional<X> as our data member.
  optional<INT_LC_LT_DES_t> lc_lt;
  optional<MOR_DES_t> mor;
  optional<OMEGA_Z_DES> omega_z;
  optional<DV_DO_DZ_t> dv_do_dz;
  optional<HMF_t> hmf;
  optional<INT_ZO_ZT_DES_t> int_zo_zt;
  optional<ROFFSET_t> roffset;
  optional<LO_LC_t> lo_lc;
  optional<SIG_SUM> sigma;
  optional<AVERAGE_SCI_t> sci;
  optional<ANGLE_TO_DIST_t> arc2dist;
  std::vector<double> zo_low_;
  std::vector<double> zo_high_;
  std::vector<double> radii_;

public:
  // Initialize my integrand object from the parameters read
  // from the relevant block in the CosmoSIS ini file.
  explicit kappa_miscent_y1(cosmosis::DataBlock& config);

  // Set any data members from values read from the current sample.
  // Do not attempt to copy the sample!.
  void set_sample(cosmosis::DataBlock& sample);

  // The function to be integrated. All arguments to this function must be of
  // type double, and there must be at least two of them (because our
  // integration routine does not work for functions of one variable). The
  // function is const because calling it does not change the state of the
  // object.
  std::vector<double> operator()(double lo,
                                 double lc,
                                 double lt,
                                 double zt,
                                 double lnM,
                                 double rmis,
                                 double theta) const;

  // finalize_sample() is where products can be put into the cosmosis::DataBlock
  // representing the current sample. The object 'sample' passed to this
  // function will be the exact same object as was passed to the most recent
  // call to set_sample(). The object 'results' will be the results of the
  // integration that has just been done for that sample. This is generally the
  // item which should be put into the sample.
  void finalize_sample(
    cosmosis::DataBlock& sample,
    std::vector<cubacpp::integration_results_v> const& results) const;

  // module_label() should return the label for this module. The name this
  // returns is the name that must be used in the 'ini file' for configuring the
  // module made with this class. We return char const* rather than std::string
  // to avoid some needless memory allocations.
  static char const* module_label();

  // The following non-member (static) function creates a vector of integration
  // volumes (the type alias defined above) based on the parameters read from
  // the configuration block for the module.
  static std::vector<volume_t> make_integration_volumes(
    cosmosis::DataBlock& cfg);
};

inline std::vector<double>
kappa_miscent_y1::operator()(double lo,
                             double lc,
                             double lt,
                             double zt,
                             double lnM,
                             double rmis,
                             double theta) const
{
  // For any data members of type std::optional<X>, we have to use operator*
  // to access the X object (as if we were dereferencing a pointer).
  std::vector<double> results((1 + radii_.size()) * zo_low_.size());
  double scaled_Rmis;
  double common_term = (*roffset)(rmis) * (*lo_lc)(lo, lc, rmis) *
                       (*lc_lt)(lc, lt, zt) * (*mor)(lt, lnM, zt) *
                       (*dv_do_dz)(zt) * (*hmf)(lnM, zt) * (*omega_z)(zt) /
                       2.0 / 3.1415926535897;
  // Number counts followed by the radius bins, repeating over zo bins
  double val;
  double dist;
  for (std::size_t i = 0; i != zo_low_.size(); i++) {
    val = (*int_zo_zt)(zo_low_[i], zo_high_[i], zt) * common_term;
    for (std::size_t j = 0; j != radii_.size(); j++) {
      dist = (*arc2dist)(radii_[j], zt);
      scaled_Rmis = std::sqrt(dist * dist + rmis * rmis +
                              2 * rmis * dist * std::cos(theta));
      results[i * (radii_.size() + 1) + j + 1] =
        (*sci)(zt) * (*sigma)(scaled_Rmis, lnM, zt) * val;
    }
    results[i * (radii_.size() + 1)] = val; // results[i*(radii_.size()+1)+1];
  }
  return results;
}

inline char const*
kappa_miscent_y1::module_label()
{
  return "kappa_miscent_y1";
}

// The implementation of make_integration_volumes can be almost the same for
// any CosmoSISVectorIntegrand-type class. Only the names and number of the
// parameters provided need to be changed. It is critical that the names be
// given in the order that correspond to the order of arguments in the class's
// function call operator. While the compiler can verify the number of arguments
// provided is correct, it can not verify that their order matches the order of
// arguments in the function call operator.
std::vector<kappa_miscent_y1::volume_t>
kappa_miscent_y1::make_integration_volumes(cosmosis::DataBlock& cfg)
{
  try {
    return y3_cluster::make_integration_volumes_wall_of_numbers(
      cfg, module_label(), "lo", "lc", "lt", "zt", "lnm", "rmis", "theta");
  }
  catch (std::exception const& ex) {
    std::cerr << ex.what();
    throw;
  };
}
