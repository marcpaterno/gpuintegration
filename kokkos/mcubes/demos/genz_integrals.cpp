#include "kokkos/mcubes/demos/time_and_call.h"
#include "common/kokkos/Volume.cuh"
#include "common/kokkos/integrands.cuh"

template <typename F,
          int ndim,
		  bool DEBUG_MCUBES = false>
void
time_and_call(std::string integ_id,
	double epsrel,
  VegasParams& params,
  std::ostream& outfile)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-20;
  bool success = false;
  int run = 0;
  F integrand;
  integrand.set_true_value();
  
  do {
    quad::Volume<double, ndim> volume;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto res = kokkos_mcubes::integrate<F, ndim, kokkos_mcubes::Custom_generator, DEBUG_MCUBES>(
      integrand,
      epsrel,
      epsabs,
      params.ncall,
      &volume,
      params.t_iter,
      params.num_adjust_iters,
      params.num_skip_iters);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

    success = (res.status == 0);
    std::cout.precision(15);


    std::cout << integ_id << "," << std::scientific 
				<< ndim << ","
              	<< epsrel << "," 
				<< epsabs << ","
                << std::scientific << std::setprecision(15) 
				<< integrand.true_value << "," 
				<< res.estimate << "," 
				<< res.errorest << ","
				<< res.chi_sq << "," 
				<< params.t_iter << "," 
				<< params.num_adjust_iters << ","
                << params.num_skip_iters << "," 
				<< res.iters << ","
                << params.ncall << ","
                << dt.count() << "," 
				<< res.status << "\n";
				
	if(run != 0)
    	outfile << integ_id << "," << std::scientific 
				<< ndim << ","
              	<< epsrel << "," 
				<< epsabs << ","
                << std::scientific << std::setprecision(15) 
				<< integrand.true_value << "," 
				<< res.estimate << "," 
				<< res.errorest << ","
				<< res.chi_sq << "," 
				<< params.t_iter << "," 
				<< params.num_adjust_iters << ","
                << params.num_skip_iters << "," 
				<< res.iters << ","
                << params.ncall << ","
                << dt.count() << "," 
				<< res.status << "\n";

    if (res.status == 0)
      run++;
	else
		break;
  } while (run < 10);
}

int
main()
{
  Kokkos::initialize();
  {
    std::ofstream outfile("kokkos_mcubes_total_times.csv");
  	int titer = 100;
	int itmax = 20;		
	int skip = 5;
	constexpr bool DEBUG_MCUBES = false;
	std::vector<double> epsrels = {1.e-3, 1.e-4, 1.e-5, 1.e-6};
	std::vector<double> required_ncall = {1.e5, 1.e6, 1.e7, 1.e8, 1.e9, 2.e9, 3.e9, 5.e9};
	VegasParams params(required_ncall[0], titer, itmax, skip);

	for(double epsrel : epsrels){
		for(double ncall : required_ncall){
			constexpr int ndim = 8;
			params.ncall = ncall;
			time_and_call<F_1_8D,ndim>("f1", epsrel, params, outfile);
			time_and_call<F_2_8D,ndim>("f2", epsrel, params, outfile);
			time_and_call<F_3_8D,ndim>("f3", epsrel, params, outfile);
			time_and_call<F_4_8D,ndim>("f4", epsrel, params, outfile);
			time_and_call<F_5_8D,ndim>("f5", epsrel, params, outfile);
			time_and_call<F_6_8D,ndim>("f6", epsrel, params, outfile);
		}
	}
	
	for(double ncall : required_ncall){
		for(double epsrel : epsrels){
			constexpr int ndim = 7;
			params.ncall = ncall;
			time_and_call<F_1_7D,ndim>("f1", epsrel, params, outfile);
			time_and_call<F_2_7D,ndim>("f2", epsrel, params, outfile);
			time_and_call<F_3_7D,ndim>("f3", epsrel, params, outfile);
			time_and_call<F_4_7D,ndim>("f4", epsrel, params, outfile);
			time_and_call<F_5_7D,ndim>("f5", epsrel, params, outfile);
			time_and_call<F_6_7D,ndim>("f6", epsrel, params, outfile);
		}
	}
	for(double ncall : required_ncall){
		for(double epsrel : epsrels){
			constexpr int ndim = 6;
			params.ncall = ncall;
			time_and_call<F_1_6D,ndim>("f1", epsrel, params, outfile);
			time_and_call<F_2_6D, ndim, DEBUG_MCUBES>("f2", epsrel, params, outfile);
			time_and_call<F_3_6D,ndim>("f3", epsrel, params, outfile);
			time_and_call<F_4_6D,ndim>("f4", epsrel, params, outfile);
			time_and_call<F_5_6D,ndim>("f5", epsrel, params, outfile);
			time_and_call<F_6_6D,ndim>("f6", epsrel, params, outfile);
		}	
	}
	
	for(double ncall : required_ncall){
		for(double epsrel : epsrels){
			constexpr int ndim = 5;
			params.ncall = ncall;
			time_and_call<F_1_5D,ndim>("f1", epsrel, params, outfile);
			time_and_call<F_2_5D,ndim>("f2", epsrel, params, outfile);
			time_and_call<F_3_5D,ndim>("f3", epsrel, params, outfile);
			time_and_call<F_4_5D,ndim>("f4", epsrel, params, outfile);
			time_and_call<F_5_5D,ndim>("f5", epsrel, params, outfile);
			time_and_call<F_6_5D,ndim>("f6", epsrel, params, outfile);
		}	
	}
		
	outfile.close();
  }
  Kokkos::finalize();
  return 0;
}
