#ifndef Y3_CLUSTER_COSMOSIS_SCALAR_INTEGRATION_MODULE_HH
#define Y3_CLUSTER_COSMOSIS_SCALAR_INTEGRATION_MODULE_HH

#include "cubacpp/cubacpp.hh"
#include "utils/datablock.hh"
#include "utils/entry.hh"
#include "utils/ndarray.hh"

#include "utils/multi_dimensional_integrator.hh"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace y3_cluster {
  // CosmoSISScalarIntegrationModule is a class template used to create a class
  // that is a CosmoSIS physics module, and which, for each CosmoSIS sample,
  // calculates the definite integral of a scalar-valued function.
  //
  // A module instantiated from this class template can be configured to
  // integrate multiple distinct volumes (ranges of integration for each
  // variable of integration). In addition, it can be configured to do these
  // integrations on each point in a multidimensional grid.
  //
  // The results are put into the cosmosis::DataBlock for the sample
  //
  // Likelihood calculations based on the result of this integration must be
  // done by a separate CosmoSIS module.

  template <typename COSMOSISINTEGRAND>
  class CosmoSISScalarIntegrationModule {
  public:
    using IntegrandType = COSMOSISINTEGRAND;
    using volume_t = cubacpp::integration_volume_for_t<IntegrandType>;
    using grid_point_t = typename IntegrandType::grid_point_t;
    using integration_results_t = cubacpp::integration_result;

    explicit CosmoSISScalarIntegrationModule(cosmosis::DataBlock& cfg);

    // Evaluate all the integrals specified (all grid points, and all volumes),
    // and record the results in the sample.
    void execute(cosmosis::DataBlock& sample);

  private:
    // Return the number of results that will be entered into the DataBlock.
    // This is the total number of integrals computed in one call to execute().
    std::size_t num_results() const;

    std::vector<integration_results_t>
    integrate_cartesian_product_of_volumes_and_gridpoints();

    std::vector<integration_results_t>
    integrate_zipped_sequence_of_volumes_and_gridpoints();

    // finalize_sample() is where products can be put into the
    // cosmosis::DataBlock representing the current sample. The object 'sample'
    // passed to this function will be the exact same object as was passed to
    // the most recent call to set_sample(). The object 'results' will be the
    // results of the integration that has just been done for that sample. This
    // is generally the item which should be put into the sample.
    void finalize_sample(
      cosmosis::DataBlock& sample,
      std::vector<integration_results_t> const& results) const;

    void finalize_sample_cartesian_product_of_volumes_and_gridpoints(
      cosmosis::DataBlock& sample,
      std::vector<integration_results_t> const& results) const;

    void finalize_sample_zipped_sequence_of_volumes_and_gridpoints(
      cosmosis::DataBlock& sample,
      std::vector<integration_results_t> const& results) const;

    IntegrandType integrand_;
    MultiDimensionalIntegrator algorithm_;
    std::vector<volume_t> volumes_;
    std::vector<grid_point_t> grid_points_;
    double eps_rel_;
    double eps_abs_;
    bool use_cartesian_product_of_volumes_and_gridpoints_;
  };
} // namespace y3_cluster

template <typename I>
y3_cluster::CosmoSISScalarIntegrationModule<I>::CosmoSISScalarIntegrationModule(
  cosmosis::DataBlock& cfg)
try
  : integrand_(cfg),
    algorithm_(cfg.view<std::string>(IntegrandType::module_label(),
                                     "algorithm")),
    volumes_(IntegrandType::make_integration_volumes(cfg)),
    grid_points_(IntegrandType::make_grid_points(cfg)),
    eps_rel_(cfg.view<double>(IntegrandType::module_label(), "eps_rel")),
    eps_abs_(cfg.view<double>(IntegrandType::module_label(), "eps_abs")),
    use_cartesian_product_of_volumes_and_gridpoints_(
      cfg.view<bool>(IntegrandType::module_label(), "use_cartesian_product")) {
  if (not use_cartesian_product_of_volumes_and_gridpoints_) {
    if (volumes_.size() != grid_points_.size()) {
      throw std::runtime_error(
        "An integration module was configured to use a zipped sequence of "
        "volumes and gridpoints,\n"
        "but the number of volumes did not equal the number of gridpoints.\n");
    }
  }
  algorithm_.set_maxeval(
    cfg.view<int>(IntegrandType::module_label(), "max_eval"));
  cubacores(0, 0);
}
catch (cosmosis::Exception const&) {
  std::cerr
    << "\nDuring construction of a CosmoSISScalarIntegrationModule, the "
       "lookup of some parameter"
    << "\nfailed. It may be a wrong name, or a wrong type.\n";
}
catch (std::exception const& e) {
  std::cerr << "\nDuring construction of a CosmoSISScalarIntegrationModule, an "
               "std::exeption throw was encountered.\n"
               "The error message was:\n"
            << e.what();
}
catch (...) {
  std::cerr << "\nUnknown exception type thrown while constructing a "
               "CosmoSISScalarIntegrationModule.\n\n";
}

template <typename I>
void
y3_cluster::CosmoSISScalarIntegrationModule<I>::execute(
  cosmosis::DataBlock& sample)
{
  integrand_.set_sample(sample);
  auto results = use_cartesian_product_of_volumes_and_gridpoints_ ?
                   integrate_cartesian_product_of_volumes_and_gridpoints() :
                   integrate_zipped_sequence_of_volumes_and_gridpoints();
  finalize_sample(sample, results);
}

template <typename I>
std::size_t
y3_cluster::CosmoSISScalarIntegrationModule<I>::num_results() const
{
  return (use_cartesian_product_of_volumes_and_gridpoints_) ?
           volumes_.size() * grid_points_.size() :
           volumes_.size();
}

template <typename I>
std::vector<cubacpp::integration_result>
y3_cluster::CosmoSISScalarIntegrationModule<
  I>::integrate_cartesian_product_of_volumes_and_gridpoints()
{
  std::vector<integration_results_t> results;
  results.reserve(num_results());

  for (auto const& volume : volumes_) {
    for (auto const& grid_point : grid_points_) {
      integrand_.set_grid_point(grid_point);
      results.push_back(
        algorithm_.integrate(integrand_, eps_rel_, eps_abs_, volume));
    }
  }
  return results;
}

template <typename COSMOSISINTEGRAND>
std::vector<cubacpp::integration_result>
y3_cluster::CosmoSISScalarIntegrationModule<
  COSMOSISINTEGRAND>::integrate_zipped_sequence_of_volumes_and_gridpoints()
{
  std::vector<integration_results_t> results;
  results.reserve(num_results());
  for (std::size_t i = 0; i != num_results(); ++i) {
    integrand_.set_grid_point(grid_points_[i]);
    results.push_back(
      algorithm_.integrate(integrand_, eps_rel_, eps_abs_, volumes_[i]));
  }
  return results;
}

template <typename I>
void
y3_cluster::CosmoSISScalarIntegrationModule<I>::finalize_sample(
  cosmosis::DataBlock& sample,
  std::vector<cubacpp::integration_result> const& res) const
{
  if (use_cartesian_product_of_volumes_and_gridpoints_)
    finalize_sample_cartesian_product_of_volumes_and_gridpoints(sample, res);
  else
    finalize_sample_zipped_sequence_of_volumes_and_gridpoints(sample, res);
}

template <typename I>
void
y3_cluster::CosmoSISScalarIntegrationModule<I>::
  finalize_sample_zipped_sequence_of_volumes_and_gridpoints(
    cosmosis::DataBlock& sample,
    std::vector<cubacpp::integration_result> const& res) const
{
  using cosmosis::ndarray;
  using cubacpp::integration_result;

  auto const nresults = num_results();

  // Create an ndarray to give a view into the 'res' vector.
  ndarray<integration_result> results(res, {nresults});

  // Create storage buffers for ndarrays to be inserted into the sample, and
  // then create the ndarrays.
  std::vector<double> vals_buffer(nresults);
  ndarray<double> vals(vals_buffer, {nresults});

  std::vector<double> errors_buffer(nresults);
  ndarray<double> errors(errors_buffer, {nresults});

  std::vector<double> probs_buffer(nresults);
  ndarray<double> probs(probs_buffer, {nresults});

  std::vector<int> statuses_buffer(nresults);
  ndarray<int> statuses(statuses_buffer, {nresults});

  std::vector<int> nregions_buffer(nresults);
  ndarray<int> nregions(nregions_buffer, {nresults});

  for (std::size_t i = 0; i != nresults; ++i) {
    vals(i) = results(i).value;
    errors(i) = results(i).error;
    probs(i) = results(i).prob;
    statuses(i) = results(i).status;
    nregions(i) = results(i).nregions;
  }

  auto module_label = IntegrandType::module_label();
  sample.put_val(module_label, "vals", vals);
  sample.put_val(module_label, "errors", errors);
  sample.put_val(module_label, "probs", probs);
  sample.put_val(module_label, "status", statuses);
  sample.put_val(module_label, "nregions", nregions);
}
template <typename I>
void
y3_cluster::CosmoSISScalarIntegrationModule<I>::
  finalize_sample_cartesian_product_of_volumes_and_gridpoints(
    cosmosis::DataBlock& sample,
    std::vector<cubacpp::integration_result> const& res) const
{
  using cosmosis::ndarray;
  using cubacpp::integration_result;

  auto const nresults = num_results();

  auto const ngrid_points = grid_points_.size();
  auto const nvolumes = volumes_.size();

  // Create an ndarray to give a view into the 'res' vector.
  ndarray<integration_result> results(res, {nvolumes, ngrid_points});

  // Create storage buffers for ndarrays to be inserted into the sample, and
  // then create the ndarrays.
  std::vector<double> vals_buffer(nresults);
  ndarray<double> vals(vals_buffer, {nvolumes, ngrid_points});

  std::vector<double> errors_buffer(nresults);
  ndarray<double> errors(errors_buffer, {nvolumes, ngrid_points});

  std::vector<double> probs_buffer(nresults);
  ndarray<double> probs(probs_buffer, {nvolumes, ngrid_points});

  std::vector<int> statuses_buffer(nresults);
  ndarray<int> statuses(statuses_buffer, {nvolumes, ngrid_points});

  std::vector<int> nregions_buffer(nresults);
  ndarray<int> nregions(nregions_buffer, {nvolumes, ngrid_points});

  for (std::size_t ivol = 0; ivol != nvolumes; ++ivol) {
    for (std::size_t jgp = 0; jgp != ngrid_points; ++jgp) {
      vals(ivol, jgp) = results(ivol, jgp).value;
      errors(ivol, jgp) = results(ivol, jgp).error;
      probs(ivol, jgp) = results(ivol, jgp).prob;
      statuses(ivol, jgp) = results(ivol, jgp).status;
      nregions(ivol, jgp) = results(ivol, jgp).nregions;
    }
  }

  auto module_label = IntegrandType::module_label();
  sample.put_val(module_label, "vals", vals);
  sample.put_val(module_label, "errors", errors);
  sample.put_val(module_label, "probs", probs);
  sample.put_val(module_label, "status", statuses);
  sample.put_val(module_label, "nregions", nregions);
}

#endif
