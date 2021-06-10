#ifndef SIGMA_MISCENT_Y1_SCALARINTEGRAND_CUH
#define SIGMA_MISCENT_Y1_SCALARINTEGRAND_CUH

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudaCuhre/integrands/hmf_t.cuh"
#include "cudaCuhre/integrands/sig_sum.cuh"
#include "cudaCuhre/quad/util/cudaArray.cuh"
#include "utils/datablock.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_integration_volumes.hh"

#include "cubacpp/integration_result.hh"
#include "cubacpp/integration_volume.hh"

template <class M>
M
make_from_file(char const* filename)
{
  static_assert(std::is_default_constructible<M>::value,
                "Type must be default constructable");
  char const* basedir = std::getenv("Y3_CLUSTER_CPP_DIR");
  if (basedir == nullptr)
    throw std::runtime_error("Y3_CLUSTER_CPP_DIR was not defined\n");
  std::string fname(basedir);
  fname += '/';
  fname += filename;
  std::ifstream in(fname);
  if (!in) {
    std::string msg("Failed to open file: ");
    msg += fname;
    throw std::runtime_error(msg);
  }
  M result;
  in >> result;
  return result;
}

namespace quad {

  // We write using declarations so that we don't have to type the namespace
  // name each time we use these names
  using cosmosis::DataBlock;
  using cubacpp::integration_result;

  template <class T>
  class Snapshotsim_ScalarIntegrand_Sigma {
  public:
    // Define the data-type describing a grid point; this should be an
    // instance of std::array<double, N> with N set to the number
    // of different paramaters being varied in the grid.
    // The alias we define must be grid_point_t.
    using grid_point_t = std::array<double, 2>; // we only vary radius.

  private:
    // We define the type alias volume_t to be the right dimensionality
    // of integration volume for our integrand. If we were to change the
    // number of arguments required by the function call operator (below),
    // we would need to also modify this type alias to keep consistent.
    using volume_t = cubacpp::IntegrationVolume<2>;

    // State obtained from configuration. These things should be set in the
    // constructor.
    // <none in this example>

    // State obtained from each sample.
    // If there were a type X that did not have a default constructor,

    hmf_t<T> hmf;
    sig_sum<T> sigma;

    // State set for current 'bin' to be integrated.
    double radius_;
    double zt_;

  public:
    Snapshotsim_ScalarIntegrand_Sigma()
    {
      hmf = make_from_file<hmf_t<T>>("data/HMF_t.dump");
      sigma = make_from_file<sig_sum<T>>("data/SIG_SUM.dump");
    }

    // Initialize my integrand object from the parameters read
    // from the relevant block in the CosmoSIS ini file.
    Snapshotsim_ScalarIntegrand_Sigma(cosmosis::DataBlock& config)
      : hmf(), sigma(), radius_(), zt_()
    {}

    // Set any data members from values read from the current sample.
    // Do not attempt to copy the sample!.
    void
    set_sample(cosmosis::DataBlock& sample)
    {
      // If we had a data member of type std::optional<X>, we would set the
      // value using std::optional::emplace(...) here. emplace takes a set
      // of arguments that it passes to the constructor of X.
      hmf.emplace(sample);
      sigma.emplace(sample);
    }

    void
    set_sample(hmf_t<T> const& hmf_in, sig_sum<T> const& sig_sum_in)
    {
      hmf = std::move(hmf_in);
      sigma = std::move(sig_sum_in);
    }

    // Set the data for the current bin.
    void
    set_grid_point(grid_point_t const& pt)
    {
      radius_ = pt[1];
      zt_ = pt[0];
    }

    // The function to be integrated. All arguments to this function must be of
    // type double, and there must be at least two of them (because our
    // integration routine does not work for functions of one variable). The
    // function is const because calling it does not change the state of the
    // object.
    __device__ double
    operator()(double lt, double lnM) const
    {
      // For any data members of type std::optional<X>, we have to use operator*
      // to access the X object (as if we were dereferencing a pointer).
      /*auto constexpr*/ double simulation_cosmic_volume =
        4492125. /*165.0 * 165.0 * 165.0*/;
      double val =
        simulation_cosmic_volume * (hmf)(lnM, zt_) * (sigma)(radius_, lnM, zt_);
      return val;
    }
    // module_label() is a non-member (static) function that returns the label
    // for this module. The name this returns is the name that must be used in
    // the 'ini file' for configuring the module made with this class. We return
    // char const* rather than std::string to avoid some needless memory
    // allocations.
    static char const*
    module_label()
    {
      return "Snapshotsim_ScalarIntegrand_Sigma";
    }

    // The following non-member (static) function creates a vector of
    // integration volumes (the type alias defined above) based on the
    // parameters read from the configuration block for the module.
    static std::vector<volume_t>
    make_integration_volumes(cosmosis::DataBlock& cfg)
    {
      return y3_cluster::make_integration_volumes_wall_of_numbers(
        cfg, Snapshotsim_ScalarIntegrand_Sigma::module_label(), "lt", "lnm");
    }

    // The following non-member (static) function creates a vector of grid
    // points on which the integration results are to be evaluated, based on
    // parameters read from the configuration block for the module.
    static std::vector<grid_point_t>
    make_grid_points(cosmosis::DataBlock& cfg)
    {
      return y3_cluster::make_grid_points_cartesian_product(
        cfg,
        Snapshotsim_ScalarIntegrand_Sigma::module_label(),
        "snapshot_zs",
        "radii");
    }
  };

}

#endif