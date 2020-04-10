#ifndef Y3_CLUSTER_COSMOSIS_INTEGRATION_MODULE_HH
#define Y3_CLUSTER_COSMOSIS_INTEGRATION_MODULE_HH

#include "cubacpp/cuhre.hh"
#include "cubacpp/integrand_traits.hh"
#include "cubacpp/integration_result.hh"
#include "cubacpp/integration_volume.hh"
#include "utils/datablock.hh"
#include "utils/entry.hh"

#include <iostream>
#include <vector>

namespace y3_cluster {
  // CosmoSISVectorIntegrationModule is a class template used to create a class
  // that is a CosmoSIS physics module, and which, for each CosmoSIS sample,
  // calculates the definite integral of a vector-valued function, storing the
  // result in the cosmosis::DataBlock for that sample.
  //
  // Likelihood calculations based on the result of this integration must be
  // done by a separate CosmoSIS module.

  template <typename COSMOSISINTEGRAND, typename ALG = cubacpp::Cuhre>
  class CosmoSISVectorIntegrationModule {
  public:
    using IntegrandType = COSMOSISINTEGRAND;
    using Algorithm = ALG;

    explicit CosmoSISVectorIntegrationModule(cosmosis::DataBlock& cfg);

    void execute(cosmosis::DataBlock& sample);

  private:
    using volume_t = cubacpp::integration_volume_for_t<IntegrandType>;
    using integration_results_type = typename cubacpp::integrand_traits<
      IntegrandType>::integration_results_type;

    IntegrandType integrand_;
    Algorithm algorithm_;
    std::vector<volume_t> volumes_;
    double eps_rel_;
    double eps_abs_;
  };
} // namespace y3_cluster

// transform_all is a function template that simplifies use of std::transform
// for the very common case in which an entire input sequence is being
// transformed.
template <typename F, typename IN, typename OUT>
void
transform_all(IN const& in, OUT& out, F&& f)
{
  std::transform(cbegin(in), cend(in), back_inserter(out), std::forward<F>(f));
}

template <typename I, typename A>
y3_cluster::CosmoSISVectorIntegrationModule<I, A>::
  CosmoSISVectorIntegrationModule(cosmosis::DataBlock& cfg)
try : integrand_(cfg),
      algorithm_(),
      volumes_(IntegrandType::make_integration_volumes(cfg)),
      eps_rel_(cfg.view<double>(IntegrandType::module_label(), "eps_rel")),
      eps_abs_(cfg.view<double>(IntegrandType::module_label(), "eps_abs")) {
  algorithm_.maxeval = cfg.view<int>(IntegrandType::module_label(), "max_eval");
  cubacores(0, 0);
}
catch (cosmosis::Exception const&) {
  std::cerr
    << "\nDuring construction of a CosmoSISVectorIntegrationModule, the "
       "lookup of some parameter"
    << "\nfailed. It may be a wrong name, or a wrong type.\n";
}
catch (...) {
  std::cerr << "\nUnknown exception type thrown while constructing a "
               "CosmoSISVectorIntegrationModule.\n\n";
}

template <typename I, typename A>
void
y3_cluster::CosmoSISVectorIntegrationModule<I, A>::execute(
  cosmosis::DataBlock& sample)
{
  // Prepare the integrand for this sample.
  integrand_.set_sample(sample);

  // Perform the (vectorized) integration for each volume.
  std::vector<integration_results_type> results;
  results.reserve(volumes_.size());
  auto integrate_volume = [this](auto const& vol) {
    return algorithm_.integrate(integrand_, eps_rel_, eps_abs_, vol);
  };
  transform_all(volumes_, results, integrate_volume);

  // Put the result into the sample.
  integrand_.finalize_sample(sample, results);
}

#endif
