#include "modules/sigma_miscent_y1_scalarintegrand.hh"
#include <fstream>
#include <utility>

using namespace y3_cluster;
using cosmosis::DataBlock;
using cosmosis::ndarray;
using cubacpp::integration_result;

using cosmosis::DataBlock;
using cubacpp::integration_result;

SigmaMiscentY1ScalarIntegrand::SigmaMiscentY1ScalarIntegrand(DataBlock&)
  : lc_lt()
  , mor()
  , omega_z()
  , dv_do_dz()
  , hmf()
  , int_zo_zt()
  , roffset()
  , lo_lc()
  , sigma()
  , zo_low_()
  , zo_high_()
  , radius_()
{}

void
SigmaMiscentY1ScalarIntegrand::set_sample(DataBlock& sample)
{
  // If we had a data member of type optional<X>, we would set the
  // value using optional::emplace(...) here. emplace takes a set
  // of arguments that it passes to the constructor of X.
  lc_lt.emplace(sample);
  mor.emplace(sample);
  dv_do_dz.emplace(sample);
  hmf.emplace(sample);
  omega_z.emplace(sample);
  roffset.emplace(sample);
  lo_lc.emplace(sample);
  sigma.emplace(sample);

  if (getenv("Y3_CLUSTER_CPP_DUMP") == nullptr)
    return;
  {
    std::ofstream os("MOR_DES_t.dump");
    os << *mor;
  }
  {
    std::ofstream os("DV_DO_DZ_t.dump");
    os << *dv_do_dz;
  }
  {
    std::ofstream os("HMF_t.dump");
    os << *hmf;
  }
  {
    std::ofstream os("ROFFSET_t.dump");
    os << *roffset;
  }
  {
    std::ofstream os("LO_LC_t.dump");
    os << *lo_lc;
  }
  {
    std::ofstream os("SIG_SUM.dump");
    os << *sigma;
  }
}

void
SigmaMiscentY1ScalarIntegrand::set_sample(INT_LC_LT_DES_t const& int_lc_lt_in,
                                          MOR_DES_t const& mor_in,
                                          OMEGA_Z_DES const& omega_z_in,
                                          DV_DO_DZ_t const& dv_do_dz_in,
                                          HMF_t const& hmf_in,
                                          INT_ZO_ZT_DES_t const& int_zo_zt_in,
                                          ROFFSET_t const& roffset_in,
                                          LO_LC_t const& lo_lc_in,
                                          SIG_SUM const& sig_sum_in)
{
  lc_lt = std::move(int_lc_lt_in);
  mor = std::move(mor_in);
  omega_z = std::move(omega_z_in);
  dv_do_dz = std::move(dv_do_dz_in);
  hmf = std::move(hmf_in);
  int_zo_zt = std::move(int_zo_zt_in);
  roffset = std::move(roffset_in);
  lo_lc = std::move(lo_lc_in);
  sigma = std::move(sig_sum_in);
}

void
SigmaMiscentY1ScalarIntegrand::set_grid_point(grid_point_t const& grid_point)
{
  radius_ = grid_point[2];
  zo_low_ = grid_point[0];
  zo_high_ = grid_point[1];
}

char const*
SigmaMiscentY1ScalarIntegrand::module_label()
{
  return "SigmaMiscentY1ScalarIntegrand";
}

// The implementation of make_integration_volumes can be almost the same for
// any CosmoSISIntegrand-type class. Only the names and number of the parameters
// provided need to be changed. It is critical that the names be given in the
// order that correspond to the order of arguments in the class's function call
// operator. While the compiler can verify the number of arguments provided is
// correct, it can not verify that their order matches the order of arguments in
// the function call operator.
std::vector<SigmaMiscentY1ScalarIntegrand::volume_t>
SigmaMiscentY1ScalarIntegrand::make_integration_volumes(
  cosmosis::DataBlock& cfg)
{
  return y3_cluster::make_integration_volumes_wall_of_numbers(
    cfg,
    SigmaMiscentY1ScalarIntegrand::module_label(),
    "lo",
    "lc",
    "lt",
    "zt",
    "lnm",
    "rmis",
    "theta");
}

std::vector<SigmaMiscentY1ScalarIntegrand::grid_point_t>
SigmaMiscentY1ScalarIntegrand::make_grid_points(cosmosis::DataBlock& cfg)
{
  return y3_cluster::make_grid_points_cartesian_product(
    cfg,
    SigmaMiscentY1ScalarIntegrand::module_label(),
    "zo_low",
    "zo_high",
    "radii");
}
