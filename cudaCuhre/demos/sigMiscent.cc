#include <iostream>
#include <chrono>
#include "cuba.h"
#include "cubacpp/cuhre.hh"
#include "modules/sigma_miscent_y1_scalarintegrand.hh"

template <class M>
M
make_from_file(char const* filename)
{
  static_assert(std::is_default_constructible<M>::value, "Type must be default constructable");
  char const* basedir = std::getenv("Y3_CLUSTER_CPP_DIR");
  if (basedir == nullptr) throw std::runtime_error("Y3_CLUSTER_CPP_DIR was not defined\n");
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

template <typename ALG, typename F>
bool
time_and_call_alt(ALG const& a, F f, double epsrel, double correct_answer, std::string algname, int _final=0)
{
  using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-12;
  cubacpp::array<7> lows  = {20., 5.,  5., .15,  29., 0., 0.};
  cubacpp::array<7> highs = {30., 50., 50.,.75,  38., 1., 6.28318530718};
  cubacpp::integration_volume_for_t<F> vol(lows, highs);
  
  auto t0 = std::chrono::high_resolution_clock::now();
  printf("time-and-call\n");
  auto res = a.integrate(f, epsrel, epsabs, vol);
  
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

  std::cout.precision(15); 
  std::cout<<algname<<","
		   <<std::to_string(correct_answer)<<","
			<<epsrel<<","
			<<epsabs<<","
			<<std::to_string(res.value)<<","
			<<std::to_string(res.error)<<","
			<<res.nregions<<","
			<<res.status<<","
			<<_final<<","
			<<dt.count()<<std::endl;
  if(res.status == 0)
	return true;
  else
	return false;
}

int main(){
	cubacores(0, 0);
	double const radius_ = 0x1p+0;
	double const zo_low_ = 0x1.999999999999ap-3;
	double const zo_high_ = 0x1.6666666666666p-2;
	double true_value = 0.;
	y3_cluster::INT_LC_LT_DES_t lc_lt;     // we want the default
	y3_cluster::OMEGA_Z_DES	 	omega_z;       // we want the default
	y3_cluster::INT_ZO_ZT_DES_t int_zo_zt; // we want the default
	
	y3_cluster::MOR_DES_t 	mor 		= make_from_file<y3_cluster::MOR_DES_t>("data/MOR_DES_t.dump");
	y3_cluster::DV_DO_DZ_t 	dv_do_dz 	= make_from_file<y3_cluster::DV_DO_DZ_t>("data/DV_DO_DZ_t.dump");
	y3_cluster::HMF_t 		hmf 		= make_from_file<y3_cluster::HMF_t>("data/HMF_t.dump");
	y3_cluster::ROFFSET_t 	roffset 	= make_from_file<y3_cluster::ROFFSET_t>("data/ROFFSET_t.dump");
	y3_cluster::SIG_SUM 	sig_sum 	= make_from_file<y3_cluster::SIG_SUM>("data/SIG_SUM.dump");
	y3_cluster::LO_LC_t 	lo_lc 		= make_from_file<y3_cluster::LO_LC_t>("data/LO_LC_t.dump");
	
	SigmaMiscentY1ScalarIntegrand integrand;
	integrand.set_sample(lc_lt, mor, omega_z, dv_do_dz, hmf, int_zo_zt, roffset, lo_lc, sig_sum);
	integrand.set_grid_point({zo_low_, zo_high_, radius_});
	double epsrel = 1.0e-3;
	cubacpp::Cuhre cuhre;
	
	while(time_and_call_alt<cubacpp::Cuhre, SigmaMiscentY1ScalarIntegrand>(cuhre, integrand, epsrel, true_value, "dc_f1", 1)){
		epsrel = epsrel/1.5;
    }
	return 0;
}