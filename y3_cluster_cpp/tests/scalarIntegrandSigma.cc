#ifndef SCALARINTEGR_AND_SIGMA
#define SCALARINTEGR_AND_SIGMA

#include "cuba.h"
#include "cubacpp/cuhre.hh"
#include "y3_cluster_cpp/modules/Snapshotsim_ScalarIntegrand_Sigma.hh"
#include <chrono>
#include <iostream>

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
time_and_call_alt(ALG const& a,
                  F f,
                  double epsrel,
                  double correct_answer,
                  std::string algname,
                  int _final = 0)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-15;
  cubacpp::array<2> lows  = {1.0, 32.64165641};
  cubacpp::array<2> highs {2.0, 33.33480359};
  cubacpp::integration_volume_for_t<F> vol(lows, highs);

  auto t0 = std::chrono::high_resolution_clock::now();
  auto res = a.integrate(f, epsrel, epsabs, vol);

  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

  std::cout.precision(15);
  std::cout << algname << "," << std::to_string(correct_answer) << "," << epsrel
            << "," << epsabs << "," << std::to_string(res.value) << ","
            << std::to_string(res.error) << "," << res.nregions << ","
            << res.status << "," << _final << "," << dt.count() << std::endl;
  if (res.status == 0)
    return true;
  else
    return false;
}

int
main()
{
  double const radius_ = 0.45;
  double const zt = .5;
  
  y3_cluster::HMF_t hmf = make_from_file<y3_cluster::HMF_t>("data/HMF_t.dump");
  y3_cluster::SIG_SUM sig_sum = make_from_file<y3_cluster::SIG_SUM>("data/SIG_SUM.dump");
  
  Snapshotsim_ScalarIntegrand_Sigma integrand;
  integrand.set_sample(hmf, sig_sum);
  integrand.set_grid_point({zt, radius_});
  
  //using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  //double constexpr epsabs = 1.0e-12;
  //constexpr int ndim = 2;
  
  //cubacpp::array<ndim> lows  =  {1.0, 32.64165641};
  //cubacpp::array<ndim> highs =  {2.0, 33.33480359};
  //cubacpp::integration_volume_for_t<Snapshotsim_ScalarIntegrand_Sigma> vol(lows, highs);
  
  cubacores(0, 0); // turn off the forking use in CUBA's CUHRE.
  unsigned long long constexpr maxeval = 1000 * 1000 * 1000;
  cubacpp::Cuhre cuhre;
  cuhre.maxeval = maxeval;
  cuhre.flags = 0 | 4;
  double epsrel = 1.0e-3;
  double epsrel_min = 1.0e-10;
  //auto t0 = std::chrono::high_resolution_clock::now();
  //auto res = cuhre.integrate(integrand, epsrel, epsabs, vol);
  //MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  while (time_and_call_alt<cubacpp::Cuhre, Snapshotsim_ScalarIntegrand_Sigma>(cuhre, integrand, epsrel, 0., "dc_f1", 1) && epsrel >= epsrel_min) {
    epsrel = epsrel / 1.5;
  }
  
  return 0;
}

#endif 