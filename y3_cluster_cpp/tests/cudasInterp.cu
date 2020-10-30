#include "catch2/catch.hpp"

#include "modules/sigma_miscent_y1_scalarintegrand.hh"
#include "../cudaCuhre/quad/util/cudaArray.cuh"

#include <iostream>
#include <chrono>					
#include "utils/str_to_doubles.hh"  
#include <vector> 					

#include <fstream>
#include <stdexcept>
#include <string>
#include <array>

#include "models/int_lc_lt_des_t.hh"
#include "models/omega_z_des.hh"
#include "models/int_zo_zt_des_t.hh"
#include "models/mor_des_t.hh"
#include "models/roffset_t.hh"
#include "models/dv_do_dz_t.hh"
#include "models/lo_lc_t.hh"
//using namespace y3_cluster;

//GPU integrator headers
#include "quad/GPUquad/Cuhre.cuh"
#include "quad/quad.h"
#include "quad/util/Volume.cuh"
#include "quad/util/cudaUtil.h"
//#include "function.cuh"
//CPU integrator headers
#include "cuba.h"
#include "cubacpp/cuhre.hh"
//#include "vegas.h"
//#include "RZU.cuh"

#include "cudaCuhre/integrands/sig_miscent.cuh" 

#include <limits>




/*TEST_CASE("HMF_t CONDITIONAL MODEL EXECUTION")
{
	double const zt = 0x1.cccccccccccccp-2;
	double const lnM = 0x1.0cp+5;
	
	hmf_t<CPU> hmf  = make_from_file<hmf_t<CPU>>("data/HMF_t.dump");
	hmf_t<GPU> hmf2 = make_from_file<hmf_t<GPU>>("data/HMF_t.dump");
	
	SECTION("SAME BEHAVIOR WITH <GPU> OBJECT")
	{
		CHECK(hmf2(lnM,  zt) == hmf(lnM,  zt));
	}
	
	SECTION("SAME BEHAVIOR WITH <GPU> OBJECT ON GPU")
	{
		hmf_t<GPU> *dhmf2;
		cudaMallocManaged((void**)&dhmf2, sizeof(hmf_t<GPU>));
		cudaDeviceSynchronize();
		memcpy(dhmf2, &hmf2, sizeof(hmf_t<GPU>));
		
		double* result;
		cudaMallocManaged((void**)&result, sizeof(double));
		
		testKernel<hmf_t<GPU>><<<1,1>>>(dhmf2, lnM, zt, result);
		cudaDeviceSynchronize();
		CHECK(dhmf2->operator()(lnM,  zt) == hmf(lnM,  zt));
		CHECK(*result == hmf(lnM,  zt));
		
		cudaFree(dhmf2);
		cudaFree(result);
	}
	
	SECTION("MOCK INTEGRAL WITH TWO IDENTICAL MODELS, EACH WITH EACH OWN INTERP2D")
	{
		MockIntegrand<GPU> integrand;
		integrand.modelA = make_from_file<hmf_t<GPU>>("data/HMF_t.dump");
		integrand.modelB = make_from_file<hmf_t<GPU>>("data/HMF_t.dump");
		
		MockIntegrand<GPU> *d_integrand;
		cudaMallocManaged((void**)&d_integrand, sizeof(MockIntegrand<GPU>));
		cudaDeviceSynchronize();
		memcpy(d_integrand, &integrand, sizeof(MockIntegrand<GPU>));
		
		double* result;
		cudaMallocManaged((void**)&result, sizeof(double));
		
		testKernel<MockIntegrand<GPU>><<<1,1>>>(d_integrand, lnM, zt, result);
		cudaDeviceSynchronize();
		CHECK((double)(*result)/2 == hmf(lnM,  zt));
		
		cudaFree(d_integrand);
		cudaFree(result);
	}
}*/

  // make_vec creates an std::vector<double> from an std::array, using all the
  // values in the std::array.
  
  // Create an Interp2D from an x-axis, y-axis, and z "matrix", with the matrix
  // unrolled into a one-dimenstional array.

/*TEST_CASE("Model with member Interp2D inialized from std::array, Interp2D on a known point")
{
	double const lc = 0x1.b8p+4;
    double const lt = 0x1.b8p+4;
    double const zt = 0x1.cccccccccccccp-2;
	
	y3_cluster::INT_LC_LT_DES_t lc_lt; 
	double result = lc_lt(lc, lt, zt+.01);
	
	int_lc_lt_des_t<GPU> d_lc_lt;
	double cpu_result = d_lc_lt(lc, lt, zt+.01);
	
	int_lc_lt_des_t<GPU> *d_integrand;
    cudaMallocManaged((void**)&d_integrand, sizeof(int_lc_lt_des_t<GPU>));
    cudaDeviceSynchronize();
	memcpy(d_integrand, &d_lc_lt, sizeof(int_lc_lt_des_t<GPU>));
	
	double *gpu_result;
    cudaMallocManaged((void**)&gpu_result, sizeof(double));
		
	testKernel<<<1, 1>>>(d_integrand, lc, lt, zt+.01, gpu_result);
	cudaDeviceSynchronize();
	
	CHECK(*gpu_result == cpu_result);
	CHECK(cpu_result == result);
}*/

/*TEST_CASE("Omega_z DES to Test quad::Polynomial")
{
	double const zt = 0x1.cccccccccccccp-2;
	y3_cluster::OMEGA_Z_DES omega_z;
	double result = omega_z(zt);
	
	omega_z_des<GPU> cpu_omega_z;
	double cpu_result = cpu_omega_z(zt);
	
	omega_z_des<GPU>* dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(omega_z_des<GPU>));
	cudaDeviceSynchronize();
	memcpy(dhmf2, &cpu_omega_z, sizeof(omega_z_des<GPU>));
		
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
	
	testKernel<omega_z_des<GPU>><<<1,1>>>(dhmf2, zt, gpu_result);
	cudaDeviceSynchronize();
	CHECK(result == cpu_result);
	CHECK(result == *gpu_result);
} */

/*TEST_CASE("Simple model"){
	double const zt = 0x1.cccccccccccccp-2;
	double zo_low_ = 0.0;
	double zo_high_ = 0.0;
	
	y3_cluster::INT_ZO_ZT_DES_t int_zo_zt;
	double result = int_zo_zt(zo_low_, zo_high_, zt);
	
	int_zo_zt_des_t d_int_zo_zt;
	int_zo_zt_des_t *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(int_zo_zt_des_t));
	cudaDeviceSynchronize();
	memcpy(dhmf2, &d_int_zo_zt, sizeof(int_zo_zt_des_t));
		
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
		
	testKernel<int_zo_zt_des_t><<<1,1>>>(dhmf2, zo_low_, zo_high_, zt, gpu_result);
	cudaDeviceSynchronize();
	CHECK(*gpu_result == result);
}*/


/*TEST_CASE("MOR_DES_t utilizing gaussian on gpu and std::array initialization")
{
	double const lt = 0x1.b8p+4;
	double const lnM = 0x1.0cp+5;
	double const zt = 0x1.cccccccccccccp-2;
	y3_cluster::MOR_DES_t mor = make_from_file<y3_cluster::MOR_DES_t>("data/MOR_DES_t.dump");
	mor_des_t<GPU> dmor       = make_from_file<mor_des_t<GPU>>("data/MOR_DES_t.dump");
	
	double result = mor(lt, lnM, zt);
	double cpu_result = dmor(lt, lnM, zt);
	CHECK(result == cpu_result);
	
	mor_des_t<GPU> *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(mor_des_t<GPU>));
	cudaDeviceSynchronize();
	memcpy(dhmf2, &dmor, sizeof(mor_des_t<GPU>));
		
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
		
	testKernel<mor_des_t<GPU>><<<1,1>>>(dhmf2, lt, lnM, zt, gpu_result);
	cudaDeviceSynchronize();
	CHECK(*gpu_result == result);
	printf("true:%.25f\n", result);
	printf("cpu :%.25f\n", cpu_result);
	printf("gpu :%.25f\n\n", *gpu_result);
	
	printf("true:%e\n", result);
	printf("cpu :%e\n", cpu_result);
	printf("gpu :%e\n\n", *gpu_result);
	
	printf("true:%a\n", result);
	printf("cpu :%a\n", cpu_result);
	printf("gpu :%a\n", *gpu_result);
	printf("------------------------------\n");
}*/

/*TEST_CASE("ROFFSET_t")
{
	double const rmis = 0x1p+0;
	y3_cluster::ROFFSET_t roffset 	= make_from_file<y3_cluster::ROFFSET_t>("data/ROFFSET_t.dump");
	double result = roffset(rmis);
	roffset_t droffset = make_from_file<roffset_t>("data/ROFFSET_t.dump");
	double cpu_result = droffset(rmis);
	CHECK(result == cpu_result);
	
	roffset_t *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(roffset_t));
	cudaDeviceSynchronize();
	memcpy(dhmf2, &droffset, sizeof(roffset_t));
		
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
		
	testKernel<roffset_t><<<1,1>>>(dhmf2, rmis, gpu_result);
	cudaDeviceSynchronize();
	CHECK(*gpu_result == result);
	
	cudaFree(dhmf2);
	cudaFree(gpu_result);
}*/

/*TEST_CASE("dv_do_dz_t")
 {
	 double const zt = 0x1.cccccccccccccp-2;
	 y3_cluster::DV_DO_DZ_t dv_do_dz 	= make_from_file<y3_cluster::DV_DO_DZ_t>("data/DV_DO_DZ_t.dump");
	 double result = dv_do_dz(zt);
	 
	 dv_do_dz_t<GPU> d_dv_do_dz = make_from_file<dv_do_dz_t<GPU>>("data/DV_DO_DZ_t.dump");
	 double cpu_result = d_dv_do_dz(zt);
	 CHECK(result == cpu_result);

	 dv_do_dz_t<GPU> *dhmf2;
	 cudaMallocManaged((void**)&dhmf2, sizeof(dv_do_dz_t<GPU>));
	 cudaDeviceSynchronize();
	 memcpy(dhmf2, &d_dv_do_dz, sizeof(dv_do_dz_t<GPU>));
	 CHECK(dhmf2->operator()(zt) == result);
	 
	 double* gpu_result;
	 cudaMallocManaged((void**)&gpu_result, sizeof(double));
	 
	 testKernel<dv_do_dz_t<GPU>><<<1,1>>>(dhmf2, zt, gpu_result);
	 cudaDeviceSynchronize();
	 CHECK(*gpu_result == result);
	 
	 cudaFree(dhmf2);
	 cudaFree(gpu_result);
 }*/
 
/*TEST_CASE("LO_LC_t")
{
	double const lo = 0x1.9p+4;
    double const lc = 0x1.b8p+4;
	double const rmis = 0x1p+0;
	y3_cluster::LO_LC_t lo_lc = make_from_file<y3_cluster::LO_LC_t>("data/LO_LC_t.dump");
	double result = lo_lc(lo, lc, rmis);
	lo_lc_t d_lo_lc = make_from_file<lo_lc_t>("data/LO_LC_t.dump");
	double cpu_result = d_lo_lc(lo, lc, rmis);
	CHECK(result == cpu_result);
	
	lo_lc_t *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(lo_lc_t));
	cudaDeviceSynchronize();
	memcpy(dhmf2, &d_lo_lc, sizeof(lo_lc_t));
	CHECK(dhmf2->operator()(lo, lc, rmis) == result);
	 
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
	 
	testKernel<lo_lc_t><<<1,1>>>(dhmf2, lo, lc, rmis, gpu_result);
	cudaDeviceSynchronize();
	CHECK(*gpu_result == result);
	 
	cudaFree(dhmf2);
	cudaFree(gpu_result);
}*/

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

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
  double absolute_error = std::abs(res.value - correct_answer);
  bool const good = (res.status == 0);
  int converge = !good;
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

template <typename F>
bool
time_and_call(std::string id,
              F integrand,
              double epsrel,
              double true_value,
              char const* algname,
              std::ostream& outfile,
              int _final = 0)
{
	//printf("time_and_call d_integrand Mor des cols:%lu\n", integrand.mor.sig_interp->_cols);
	//printf("inside time and call\n");
	//printf("time_and_call d_integrand Mor des cols:%lu\n", integrand.mor.sig_interp->_cols);
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-12;

  double lows[] =  {20., 5.,  5., .15,  29., 0., 0.};
  double highs[] = {30., 50., 50.,.75,  38., 1., 6.28318530718};

  constexpr int ndim = 7;
  quad::Volume<double, ndim> vol(lows, highs);
  int const key = 0;
  int const verbose = 0;
  int const numdevices = 1;
  quad::Cuhre<double, ndim> alg(0, nullptr, key, verbose, numdevices);
	
  int outfileVerbosity = 1;
  constexpr int phase_I_type = 0; // alternative phase 1

  auto const t0 = std::chrono::high_resolution_clock::now();
  
  cuhreResult const result = alg.integrate<F>(integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);
  
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(result.estimate - true_value);
  bool good = false;

  if (result.status == 0 || result.status == 2) {
    good = true;
  }
  outfile.precision(15);
  outfile << std::fixed << id << ",\t" << std::scientific << true_value << ",\t"
          << std::scientific << epsrel << ",\t\t\t" << std::scientific
          << epsabs << ",\t" << std::scientific << result.estimate << ",\t"
          << std::scientific << result.errorest << ",\t" << std::fixed
          << result.nregions << ",\t" << std::fixed << result.status << ",\t"
          << _final << ",\t" << dt.count() << std::endl;

  return good;
}


int
main()
{

  //TEST_CASE("integral call"){
  
    
  printf("Final Test Case\n");
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
  double result = integrand(lo, lc, lt, zt, lnM, rmis, theta);
  //time_and_call_vegas(integrand);
  //return 0;
  
  cubacores(0, 0);

 //unsigned long long constexpr mmaxeval = std::numeric_limits<unsigned long long>::max();
  //std::cout<<"mmaxeval:"<<mmaxeval<<"\n";
										    
  //unsigned long long constexpr maxeval = 10000000000;
  //double const epsrel_min = 1.0e-12;
  //cubacpp::Cuhre cuhre;
  //int verbose = 3;
  //int verbose = 0;
  int _final  = 1;
  //cuhre.flags = verbose | 4;
  //cuhre.maxeval = maxeval;
  double epsrel = 1.0e-3;
  //double true_value = 0.;
	
  //  while(time_and_call_alt<cubacpp::Cuhre, SigmaMiscentY1ScalarIntegrand>(cuhre, integrand, epsrel, true_value, "dc_f1", 0)){
  //    epsrel = epsrel/1.5;
  //    }

  // return 0;
  
  integral<GPU> d_integrand;
  //constexpr int ndim = 7;
  d_integrand.set_grid_point({zo_low_, zo_high_, radius_});
  //VegasGPU <integral<GPU>, ndim>(d_integrand);  
   while(time_and_call<integral<GPU>>("pdc_f1_latest",
   				     d_integrand,
   				     epsrel,
   				     0.,
   				     "gpucuhre",
   				     std::cout,
   				     _final)){
     epsrel = epsrel/1.5;	
     break;
   }
									 
  return 0;
}
