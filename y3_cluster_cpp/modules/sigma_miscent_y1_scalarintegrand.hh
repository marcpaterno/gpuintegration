#ifndef SIGMA_MISCENT_Y1_SCALARINTEGRAND
#define SIGMA_MISCENT_Y1_SCALARINTEGRAND

#include "utils/make_grid_points.hh"
#include "utils/make_integration_volumes.hh"

#include "cubacpp/integration_result.hh"
#include "cubacpp/integration_volume.hh"
#include "utils/datablock.hh"

#include "Optional/optional.hpp"
#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/int_lc_lt_des_t.hh"
#include "models/int_zo_zt_des_t.hh"
#include "models/lo_lc_t.hh"
#include "models/mor_des_t.hh"
#include "models/omega_z_des.hh"
#include "models/roffset_t.hh"
#include "models/sig_sum.hh"

#include <iostream>
#include <vector>

namespace y3_cluster {
  using std::experimental::optional;
}
using namespace y3_cluster;

using cosmosis::DataBlock;
using cosmosis::ndarray;
using cubacpp::integration_result;

// SigmaMiscentY1ScalarIntegrand is a class that models the concept of
// "CosmoSISScalarIntegrand", and is thus suitable for use as the template
// parameter for the class template CosmoSISScalarIntegrationModule.
//
// Notes:
//    1) optional<T> is used for data members that are not
//    constructible from CosmoSIS configuration parameters.
//
//    2) The object as created by the only constructor does not need to be
//    in a callable state.
//
//    3) After calls to both set_sample and set_grid_point have been made, the
//    object must be in a callable state.
//
//    4) State that *can* be correctly set by the constructor *should* be set by
//    the constructor. Do not needlessly repeat initialization in calls to
//    set_sample.
//
//
class SigmaMiscentY1ScalarIntegrand {
public:
  // Define the data-type describing a grid point; this should be an
  // instance of std::array<double, N> with N set to the number
  // of different paramaters being varied in the grid.
  // The alias we define must be grid_point_t.
  using grid_point_t = std::array<double, 3>; // we only vary radius.

private:
  // We define the type alias volume_t to be the right dimensionality
  // of integration volume for our integrand. If we were to change the
  // number of arguments required by the function call operator (below),
  // we would need to also modify this type alias to keep consistent.
  using volume_t = cubacpp::IntegrationVolume<7>;

  // State obtained from configuration. These things should be set in the
  // constructor.
  // <none in this example>

  // State obtained from each sample.
  // If there were a type X that did not have a default constructor,
  // we would use optional<X> as our data member.
  optional<INT_LC_LT_DES_t> lc_lt;
  optional<MOR_DES_t> mor;
  optional<OMEGA_Z_DES> omega_z;
  optional<DV_DO_DZ_t> dv_do_dz;
  optional<HMF_t> hmf;
  optional<INT_ZO_ZT_DES_t> int_zo_zt;
  optional<ROFFSET_t> roffset;
  optional<LO_LC_t> lo_lc;
  optional<SIG_SUM> sigma;

  // State set for current 'bin' to be integrated.
  double zo_low_ = 0.0;
  double zo_high_ = 0.0;
  double radius_ = 0.0;

public:
  // Default c'tor just for testing outside of CosmoSIS.
  SigmaMiscentY1ScalarIntegrand() = default;

  // Set any data members from values read from the current sample.
  // Do not attempt to copy the sample!.
  void set_sample(cosmosis::DataBlock& sample);
  void set_sample(INT_LC_LT_DES_t const&,
                  MOR_DES_t const&,
                  OMEGA_Z_DES const&,
                  DV_DO_DZ_t const&,
                  HMF_t const&,
                  INT_ZO_ZT_DES_t const&,
                  ROFFSET_t const&,
                  LO_LC_t const&,
                  SIG_SUM const&);

  // Set the data for the current bin.
  void set_grid_point(grid_point_t const& pt);

  // The function to be integrated. All arguments to this function must be of
  // type double, and there must be at least two of them (because our
  // integration routine does not work for functions of one variable). The
  // function is const because calling it does not change the state of the
  // object.
  double operator()(double lo,
                    double lc,
                    double lt,
                    double zt,
                    double lnM,
                    double rmis,
                    double theta) const;

  // module_label() is a non-member (static) function that returns the label for
  // this module. The name this returns
  // is the name that must be used in the 'ini file' for configuring the module
  // made with this class.
  // We return char const* rather than std::string to avoid some needless memory
  // allocations.
  static char const* module_label();

  // The following non-member (static) function creates a vector of integration
  // volumes (the type alias defined above) based on the parameters read from
  // the configuration block for the module.
  static std::vector<volume_t> make_integration_volumes(
    cosmosis::DataBlock& cfg);

  // The following non-member (static) function creates a vector of grid points
  // on which the integration results are to be evaluated, based on parameters
  // read from the configuration block for the module.
  static std::vector<grid_point_t> make_grid_points(cosmosis::DataBlock& cfg);
};

double
SigmaMiscentY1ScalarIntegrand::operator()(double const lo,
                                          double const lc,
                                          double const lt,
                                          double const zt,
                                          double const lnM,
                                          double const rmis,
                                          double const theta) const
{
  // For any data members of type optional<X>, we have to use operator*
  // to access the X object (as if we were dereferencing a pointer).
  static bool do_print = true;
  double const common_term = (*roffset)(rmis) * (*lo_lc)(lo, lc, rmis) *
                             (*lc_lt)(lc, lt, zt) * (*mor)(lt, lnM, zt) *
                             (*dv_do_dz)(zt) * (*hmf)(lnM, zt) *
                             (*omega_z)(zt) / 2.0 / 3.1415926535897;
  double const scaled_Rmis = std::sqrt(radius_ * radius_ + rmis * rmis +
                                       2 * rmis * radius_ * std::cos(theta));
  auto const val = (*sigma)(scaled_Rmis, lnM, zt) *
                   (*int_zo_zt)(zo_low_, zo_high_, zt) * common_term;
  if (do_print) {
    do_print = false;
    if (getenv("Y3_CLUSTER_CPP_DUMP") != nullptr) {
      std::ofstream os("points.dump");
      os << std::hexfloat << "lo :" << lo << '\n'
         << "lc : " << lc << '\n'
         << "lt : " << lt << '\n'
         << "zt : " << zt << '\n'
         << "lnM : " << lnM << '\n'
         << "rmis : " << rmis << '\n'
         << "theta : " << theta << '\n'
         << "radius_ : " << radius_ << '\n'
         << "zo_low_ : " << zo_low_ << '\n'
         << "zo_high_ : " << zo_high_ << '\n'
         << "scaled_Rmis : " << scaled_Rmis << '\n'
         << "lc_lt : " << (*lc_lt)(lc, lt, zt) << '\n'
         << "mor : " << (*mor)(lt, lnM, zt) << '\n'
         << "omega_z : " << (*omega_z)(zt) << '\n'
         << "dv_do_dz : " << (*dv_do_dz)(zt) << '\n'
         << "hmf : " << (*hmf)(lnM, zt) << '\n'
         << "int_zo_zt : " << (*int_zo_zt)(zo_low_, zo_high_, zt) << '\n'
         << "roffset : " << (*roffset)(rmis) << '\n'
         << "lo_lc : " << (*lo_lc)(lo, lc, rmis) << '\n'
         << "sigma : " << (*sigma)(scaled_Rmis, lnM, zt) << '\n';
    };
  };
  return val;
}

#endif
