// #include "modules/sigma_miscent_y1_scalarintegrand.hh"
// #include "../cudaCuhre/quad/util/cudaArray.cuh"

// #include <iostream>
// #include <chrono>					  
// #include <vector> 					

// #include <fstream>
// #include <stdexcept>
// #include <string>
// #include <array>

// //using namespace y3_cluster;

// //GPU integrator headers

// #include "quad/quad.h"
// #include "quad/util/Volume.cuh"
// #include "quad/util/cudaUtil.h"
// #include "vegas.h"

// #include <limits>
// namespace quad {
	 
// using std::cout;
// using std::chrono::high_resolution_clock;
// using std::chrono::duration;

int
main()
{

  // double const lo = 0x1.9p+4;
  // double const lc = 0x1.b8p+4;
  // double const lt = 0x1.b8p+4;
  // double const zt = 0x1.cccccccccccccp-2;
  // double const lnM = 0x1.0cp+5;
  // double const rmis = 0x1p+0;
  // double const theta = 0x1.921fb54442eeap+1;
	
  // double const radius_ = 0x1p+0;
  // double const zo_low_ = 0x1.999999999999ap-3;
  // double const zo_high_ = 0x1.6666666666666p-2;
	
  // y3_cluster::INT_LC_LT_DES_t lc_lt;     // we want the default
  // y3_cluster::OMEGA_Z_DES	 	omega_z;       // we want the default
  // y3_cluster::INT_ZO_ZT_DES_t int_zo_zt; // we want the default
	
  // y3_cluster::MOR_DES_t 	mor 		= make_from_file<y3_cluster::MOR_DES_t>("data/MOR_DES_t.dump");
  // y3_cluster::DV_DO_DZ_t 	dv_do_dz 	= make_from_file<y3_cluster::DV_DO_DZ_t>("data/DV_DO_DZ_t.dump");
  // y3_cluster::HMF_t 		hmf 		= make_from_file<y3_cluster::HMF_t>("data/HMF_t.dump");
  // y3_cluster::ROFFSET_t 	roffset 	= make_from_file<y3_cluster::ROFFSET_t>("data/ROFFSET_t.dump");
  // y3_cluster::SIG_SUM 	sig_sum 	= make_from_file<y3_cluster::SIG_SUM>("data/SIG_SUM.dump");
  // y3_cluster::LO_LC_t 	lo_lc 		= make_from_file<y3_cluster::LO_LC_t>("data/LO_LC_t.dump");
	
  // SigmaMiscentY1ScalarIntegrand integrand;
  // integrand.set_sample(lc_lt, mor, omega_z, dv_do_dz, hmf, int_zo_zt, roffset, lo_lc, sig_sum);
  // integrand.set_grid_point({zo_low_, zo_high_, radius_});
  // time_and_call_vegas(integrand);									 
  return 0;
}
