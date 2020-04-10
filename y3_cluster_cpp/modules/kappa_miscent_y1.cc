#include "modules/kappa_miscent_y1.hh"
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

// We write using declarations so that we don't have to type the namespace name
// each time we use these names
using cosmosis::DataBlock;
using cubacpp::integration_results_v;

kappa_miscent_y1::kappa_miscent_y1(DataBlock& config)
  : lc_lt()
  , mor()
  , omega_z()
  , dv_do_dz()
  , hmf()
  , int_zo_zt()
  , roffset()
  , lo_lc()
  , sigma()
  , sci()
  , arc2dist()
  , zo_low_(config.view<std::vector<double>>(kappa_miscent_y1::module_label(),
                                             "zo_low"))
  , zo_high_(config.view<std::vector<double>>(kappa_miscent_y1::module_label(),
                                              "zo_high"))
  , radii_(config.view<std::vector<double>>(kappa_miscent_y1::module_label(),
                                            "radii"))
{}

void
kappa_miscent_y1::set_sample(DataBlock& sample)
{
  // If we had a data member of type std::optional<X>, we would set the
  // value using std::optional::emplace(...) here. emplace takes a set
  // of arguments that it passes to the constructor of X.
  lc_lt.emplace(sample);
  mor.emplace(sample);
  dv_do_dz.emplace(sample);
  hmf.emplace(sample);
  omega_z.emplace(sample);
  roffset.emplace(sample);
  lo_lc.emplace(sample);
  sigma.emplace(sample);
  sci.emplace(sample);
  arc2dist.emplace(sample);
}
