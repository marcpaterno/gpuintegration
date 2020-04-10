#include "catch2/catch.hpp"
#include "modules/sigma_miscent_y1_scalarintegrand.hh"
#include "utils/datablock.hh"

#include <fstream>
#include <stdexcept>
#include <string>

using namespace y3_cluster;

template <class M>
M
make_from_file(char const* filename)
{
  static_assert(std::is_default_constructible<M>::value,
                "Type must be default constructable");
  std::ifstream in(filename);
  if (!in) {
    std::string msg("Failed to open file: ");
    msg += filename;
    throw std::runtime_error(msg);
  }
  M result;
  in >> result;
  return result;
}

TEST_CASE("sigma_miscent_y1_scalarintegrand")
{
  INT_LC_LT_DES_t lc_lt;     // we want the default
  OMEGA_Z_DES omega_z;       // we want the default
  INT_ZO_ZT_DES_t int_zo_zt; // we want the default

  MOR_DES_t mor = make_from_file<MOR_DES_t>("../../data/MOR_DES_t.dump");
  DV_DO_DZ_t dv_do_dz =
    make_from_file<DV_DO_DZ_t>("../../data/DV_DO_DZ_t.dump");
  HMF_t hmf = make_from_file<HMF_t>("../../data/HMF_t.dump");
  ROFFSET_t roffset = make_from_file<ROFFSET_t>("../../data/ROFFSET_t.dump");
  SIG_SUM sig_sum = make_from_file<SIG_SUM>("../../data/SIG_SUM.dump");
  LO_LC_t lo_lc = make_from_file<LO_LC_t>("../../data/LO_LC_t.dump");

  SigmaMiscentY1ScalarIntegrand integrand;
  integrand.set_sample(
    lc_lt, mor, omega_z, dv_do_dz, hmf, int_zo_zt, roffset, lo_lc, sig_sum);
  SECTION("can be invoked")
  {
    double const lo = 0x1.9p+4;
    double const lc = 0x1.b8p+4;
    double const lt = 0x1.b8p+4;
    double const zt = 0x1.cccccccccccccp-2;
    double const lnM = 0x1.0cp+5;
    double const rmis = 0x1p+0;
    double const theta = 0x1.921fb54442eeap+1;

    double const radius_ = 0x1p+0;
    double const zo_low_ = 0x1.999999999999ap-3;
    double const zo_high_ = 0x1.6666666666666p-2;
    double const expected = 0x1.0b1f85994aebbp-44;
    integrand.set_grid_point({zo_low_, zo_high_, radius_});
    double val = integrand(lo, lc, lt, zt, lnM, rmis, theta);
    CHECK(val == expected);
  }
}
