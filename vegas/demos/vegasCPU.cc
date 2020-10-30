#include "y3_cluster_cpp/modules/sigma_miscent_y1_scalarintegrand.hh"
//#include "cudaCuhre/quad/util/cudaArray.cuh"

// #include <iostream>
// #include <chrono>
// #include <vector>

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

// //using namespace y3_cluster;

// //GPU integrator headers

// #include "quad/quad.h"
// #include "quad/util/Volume.cuh"
//#include "quad/util/cudaUtil.h"
#include "vegas/vegas.h"

#include <limits>
// namespace quad {

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

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

int
main()
{

  double const radius_ = 0x1p+0;
  double const zo_low_ = 0x1.999999999999ap-3;
  double const zo_high_ = 0x1.6666666666666p-2;

  y3_cluster::INT_LC_LT_DES_t lc_lt;     // we want the default
  y3_cluster::OMEGA_Z_DES omega_z;       // we want the default
  y3_cluster::INT_ZO_ZT_DES_t int_zo_zt; // we want the default

  y3_cluster::MOR_DES_t mor =
    make_from_file<y3_cluster::MOR_DES_t>("data/MOR_DES_t.dump");
  y3_cluster::DV_DO_DZ_t dv_do_dz =
    make_from_file<y3_cluster::DV_DO_DZ_t>("data/DV_DO_DZ_t.dump");
  y3_cluster::HMF_t hmf = make_from_file<y3_cluster::HMF_t>("data/HMF_t.dump");
  y3_cluster::ROFFSET_t roffset =
    make_from_file<y3_cluster::ROFFSET_t>("data/ROFFSET_t.dump");
  y3_cluster::SIG_SUM sig_sum =
    make_from_file<y3_cluster::SIG_SUM>("data/SIG_SUM.dump");
  y3_cluster::LO_LC_t lo_lc =
    make_from_file<y3_cluster::LO_LC_t>("data/LO_LC_t.dump");

  SigmaMiscentY1ScalarIntegrand integrand;
  integrand.set_sample(
    lc_lt, mor, omega_z, dv_do_dz, hmf, int_zo_zt, roffset, lo_lc, sig_sum);
  integrand.set_grid_point({zo_low_, zo_high_, radius_});
  time_and_call_vegas(integrand);
  return 0;
}
