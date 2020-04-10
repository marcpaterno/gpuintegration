#ifndef Y3_CLUSTER_COSMOSIS_ONED_INTEGRATION_MODULE_HH
#define Y3_CLUSTER_COSMOSIS_ONED_INTEGRATION_MODULE_HH

#include <tuple>
#include <vector>

#include <gsl/gsl_integration.h>

namespace y3_cluster {

  template <typename COSMOSISINTEGRAND>
  class OneDIntegrationModule {
  public:
    using IntegrandType = COSMOSISINTEGRAND;
    using volume_t = std::array<double, 2>;
    using grid_point_t = typename IntegrandType::grid_point_t;
    using integration_results_t = std::tuple<int, double, double>;

    explicit OneDIntegrationModule(cosmosis::DataBlock& cfg);
    ~OneDIntegrationModule();

    void execute(cosmosis::DataBlock& sample);

  private:
    // Number of entries into datablock aka integrals computed
    std::size_t num_results() const;

    // Wrapper for the integrand to adhere to GSL specs
    static double integrand_wrapper(double x, void* p);

    // Do the integral for each grid point and volume
    std::vector<integration_results_t> integrate_full_sequence();

    // Put the results into the datablock
    void finalize_sample(
      cosmosis::DataBlock& sample,
      std::vector<integration_results_t> const& results) const;

    IntegrandType integrand_;
    std::vector<volume_t> volumes_;
    std::vector<grid_point_t> grid_points_;
    double eps_rel_;
    double eps_abs_;
    std::size_t maxeval_;
    gsl_integration_workspace* workspace_;
  };
} // end of y3_cluster

template <typename I>
y3_cluster::OneDIntegrationModule<I>::OneDIntegrationModule(
  cosmosis::DataBlock& cfg)
try : integrand_(cfg),
      volumes_(IntegrandType::make_integration_volumes(cfg)),
      grid_points_(IntegrandType::make_grid_points(cfg)),
      eps_rel_(cfg.view<double>(IntegrandType::module_label(), "eps_rel")),
      eps_abs_(cfg.view<double>(IntegrandType::module_label(), "eps_abs")),
      maxeval_(cfg.view<int>(IntegrandType::module_label(), "max_eval")) {
  if (volumes_.size() != grid_points_.size()) {
    throw std::runtime_error(
      "An integration module was configured with unequal numbers "
      "of volumes and gridpoints\n");
  }
  workspace_ = gsl_integration_workspace_alloc(maxeval_);
}
catch (cosmosis::Exception const&) {
  std::cerr
    << "\nDuring construction of a OneDIntegrationModule, the lookup of some "
       "parameter\nfailed. It may be a wrong name, or a wrong type.\n";
}
catch (std::exception const& e) {
  std::cerr
    << "\nDuring construction of a OneDIntegrationModule, an std::exeption "
       "throw was encountered.\nThe error message was:\n"
    << e.what();
}
catch (...) {
  std::cerr << "\nUnknown exception type thrown while constructing a "
               "OneDIntegrationModule.\n\n";
}

template <typename I>
y3_cluster::OneDIntegrationModule<I>::~OneDIntegrationModule()
{
  gsl_integration_workspace_free(workspace_);
}

template <typename I>
void
y3_cluster::OneDIntegrationModule<I>::execute(cosmosis::DataBlock& sample)
{
  integrand_.set_sample(sample);
  auto results = integrate_full_sequence();
  finalize_sample(sample, results);
}

template <typename I>
std::size_t
y3_cluster::OneDIntegrationModule<I>::num_results() const
{
  return volumes_.size();
}

template <typename I>
double
y3_cluster::OneDIntegrationModule<I>::integrand_wrapper(double x, void* p)
{
  return (*(I*)p)(x);
}

template <typename I>
std::vector<std::tuple<int, double, double>>
y3_cluster::OneDIntegrationModule<I>::integrate_full_sequence()
{
  // Set up the
  std::vector<integration_results_t> results;
  results.reserve(num_results());

  // Set up GSL integration things
  int status;
  double result, error;
  gsl_function F;
  F.function = &integrand_wrapper;

  for (std::size_t i = 0; i != num_results(); ++i) {
    integrand_.set_grid_point(grid_points_[i]);
    F.params = &integrand_;
    status = gsl_integration_qag(&F,
                                 std::get<0>(volumes_[i]),
                                 std::get<1>(volumes_[i]),
                                 eps_abs_,
                                 eps_rel_,
                                 maxeval_,
                                 6,
                                 workspace_,
                                 &result,
                                 &error);
    results.push_back(integration_results_t(status, result, error));
  }
  return results;
}

template <typename I>
void
y3_cluster::OneDIntegrationModule<I>::finalize_sample(
  cosmosis::DataBlock& sample,
  std::vector<integration_results_t> const& results) const
{
  auto const nresults = num_results();

  // Make and fill buffers for the outputs
  std::vector<double> statuses, vals, errors;
  for (std::size_t i = 0; i != nresults; ++i) {
    statuses.push_back(std::get<0>(results[i]));
    vals.push_back(std::get<1>(results[i]));
    errors.push_back(std::get<2>(results[i]));
  }

  // Place into the datablock
  auto module_label = IntegrandType::module_label();
  sample.put_val(module_label, "status", statuses);
  sample.put_val(module_label, "vals", vals);
  sample.put_val(module_label, "errors", errors);
}

#endif
