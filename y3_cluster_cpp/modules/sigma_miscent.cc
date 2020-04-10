#include "utils/make_integration_volumes.hh"
#include "utils/module_macros.hh"

#include "cubacpp/integration_result.hh"
#include "cubacpp/integration_volume.hh"
#include "utils/datablock.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/int_zo_zt_t.hh"
#include "models/lc_lt_t.hh"
#include "models/lo_lc_t.hh"
#include "models/mor_sdss_t.hh"
#include "models/omega_z_sdss.hh"
#include "models/roffset_t.hh"
#include "models/sig_sum.hh"
#include <optional>
#include <vector>
using namespace y3_cluster;

// sigma_miscent is a class that models the concept of "CosmoSISIntegrand",
// and is thus suitable for use as the template parameter for the class template
// CosmosisIntegrationModule.
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
class sigma_miscent {
private:
  // We define the type alias volume_t to be the right dimensionality
  // of integration volume for our integrand. If we were to change the
  // number of arguments required by the function call operator (below),
  // we would need to also modify this type alias to keep consistent.
  using volume_t = cubacpp::IntegrationVolume<7>;

  // State obtained from each sample.
  // If there were a type X that did not have a default constructor,
  // we would use std::optional<X> as our data member.
  std::optional<LC_LT_t> lc_lt;
  std::optional<MOR_sdss> mor;
  std::optional<OMEGA_Z_SDSS> omega_z_sdss;
  std::optional<DV_DO_DZ_t> dv_do_dz;
  std::optional<HMF_t> hmf;
  std::optional<INT_ZO_ZT_t> int_zo_zt;
  std::optional<ROFFSET_t> roffset;
  std::optional<LO_LC_t> lo_lc;
  std::optional<SIG_SUM> sigma;
  std::vector<double> zo_low_;
  std::vector<double> zo_high_;
  std::vector<double> radii_;

public:
  // Initialize my integrand object from the parameters read
  // from the relevant block in the CosmoSIS ini file.
  explicit sigma_miscent(cosmosis::DataBlock& config);

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

// We write using declarations so that we don't have to type the namespace name
// each time we use these names
using cosmosis::DataBlock;
using cubacpp::integration_results_v;

sigma_miscent::sigma_miscent(DataBlock& config)
  : lc_lt()
  , mor()
  , omega_z_sdss()
  , dv_do_dz()
  , hmf()
  , int_zo_zt()
  , roffset()
  , lo_lc()
  , sigma()
  , zo_low_(
      config.view<std::vector<double>>(sigma_miscent::module_label(), "zo_low"))
  , zo_high_(config.view<std::vector<double>>(sigma_miscent::module_label(),
                                              "zo_high"))
  , radii_(
      config.view<std::vector<double>>(sigma_miscent::module_label(), "radii"))
{}

void
sigma_miscent::set_sample(DataBlock& sample)
{
  // If we had a data member of type std::optional<X>, we would set the
  // value using std::optional::emplace(...) here. emplace takes a set
  // of arguments that it passes to the constructor of X.
  lc_lt.emplace(sample);
  mor.emplace(sample);
  dv_do_dz.emplace(sample);
  hmf.emplace(sample);
  omega_z_sdss.emplace(sample);
  int_zo_zt.emplace(sample);
  roffset.emplace(sample);
  lo_lc.emplace(sample);
  sigma.emplace(sample);
}

std::vector<double>
sigma_miscent::operator()(double lo,
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
                       (*dv_do_dz)(zt) * (*hmf)(lnM, zt) * (*omega_z_sdss)(zt) /
                       2.0 / 3.1415926535897;
  // Number counts followed by the radius bins, repeating over zo bins
  double val;
  for (std::size_t i = 0; i != zo_low_.size(); i++) {
    val = (*int_zo_zt)(zo_low_[i], zo_high_[i], zt) * common_term;
    for (std::size_t j = 0; j != radii_.size(); j++) {
      scaled_Rmis = std::sqrt(radii_[j] * radii_[j] + rmis * rmis +
                              2 * rmis * radii_[j] * std::cos(theta));
      results[i * (radii_.size() + 1) + j + 1] =
        (*sigma)(scaled_Rmis, lnM, zt) * val;
    }
    results[i * (radii_.size() + 1)] = val; // results[i*(radii_.size()+1)+1];
  }
  return results;
}

//
void
sigma_miscent::finalize_sample(
  cosmosis::DataBlock& sample,
  std::vector<integration_results_v> const& results) const
{
  std::vector<int> NM_status;
  std::vector<int> NM_nregions;
  std::vector<int> NM_nevals;
  std::vector<double> N_vals;
  std::vector<double> N_errors;
  std::vector<double> N_probs;

  std::vector<double> totSigma_vals_temp;
  std::vector<double> totSigma_errors_temp;
  std::vector<double> totSigma_probs_temp;

  //
  // TODO: Try to refactor this code, to abstract away the manipulations done
  // for each physics quantity.
  std::size_t n_zo_bins = zo_low_.size();
  std::size_t n_radii_bins = radii_.size();
  for (auto const& result : results) {
    for (std::size_t i = 0; i != zo_low_.size(); i++) {
      NM_status.push_back(result.status);
      NM_nregions.push_back(result.nregions);
      NM_nevals.push_back(result.neval);

      N_vals.push_back(result.value[i * (n_radii_bins + 1)]);
      N_errors.push_back(result.error[i * (n_radii_bins + 1)]);
      N_probs.push_back(result.prob[i * (n_radii_bins + 1)]);

      totSigma_vals_temp.insert(
        totSigma_vals_temp.end(),
        result.value.begin() + i * (n_radii_bins + 1) + 1,
        result.value.begin() + (i + 1) * (n_radii_bins + 1));
      totSigma_errors_temp.insert(
        totSigma_errors_temp.end(),
        result.error.begin() + i * (n_radii_bins + 1) + 1,
        result.error.begin() + (i + 1) * (n_radii_bins + 1));
      totSigma_probs_temp.insert(
        totSigma_probs_temp.end(),
        result.prob.begin() + i * (n_radii_bins + 1) + 1,
        result.prob.begin() + (i + 1) * (n_radii_bins + 1));
    }
  }
  std::vector<std::size_t> extents{results.size() * n_zo_bins, n_radii_bins};
  cosmosis::ndarray<double> totSigma_vals(totSigma_vals_temp, extents);
  cosmosis::ndarray<double> totSigma_errors(totSigma_errors_temp, extents);
  cosmosis::ndarray<double> totSigma_probs(totSigma_probs_temp, extents);

  sample.put_val(module_label(), "N_vals", N_vals);
  sample.put_val(module_label(), "N_errors", N_errors);
  sample.put_val(module_label(), "N_probs", N_probs);
  sample.put_val(module_label(), "NM_status", NM_status);
  sample.put_val(module_label(), "NM_nregions", NM_nregions);
  sample.put_val(module_label(), "NM_nevals", NM_nevals);
  sample.put_val(module_label(), "Sigma_radius", radii_);

  sample.put_val(module_label(), "totSigma_vals", totSigma_vals);
  sample.put_val(module_label(), "totSigma_errors", totSigma_errors);
  sample.put_val(module_label(), "totSigma_probs", totSigma_probs);
}

char const*
sigma_miscent::module_label()
{
  return "sigma_miscent";
}

// The implementation of make_integration_volumes can be almost the same for
// any CosmoSISIntegrand-type class. Only the names and number of the parameters
// provided need to be changed. It is critical that the names be given in the
// order that correspond to the order of arguments in the class's function call
// operator. While the compiler can verify the number of arguments provided is
// correct, it can not verify that their order matches the order of arguments in
// the function call operator.
std::vector<sigma_miscent::volume_t>
sigma_miscent::make_integration_volumes(cosmosis::DataBlock& cfg)
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

DEFINE_COSMOSIS_VECTOR_INTEGRATION_MODULE(sigma_miscent)
