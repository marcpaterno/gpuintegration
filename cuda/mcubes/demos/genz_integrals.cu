#include <iostream>
#include "cuda/mcubes/demos/demo_utils.cuh"
#include "common/cuda/integrands.cuh"
#include "cuda/pagani/demos/compute_genz_integrals.cuh"

int main(){
	std::ofstream outfile("mcubes_1e5.csv");
    int titer = 100;
	int itmax = 20;		//don't forget to adjust when comparing
	int skip = 5;	//that may need to be set to itmax
	std::vector<double> epsrels = {1.e-3, 1.e-4, 1.e-5, 1.e-6};
	
	std::vector<double> required_ncall = {1.e5, 1.e6, 1.e7, 1.e8, 1.e9, 2.e9, 3.e9, 5.e9};
	VegasParams params(required_ncall[0], titer, itmax, skip);

	for(double epsrel : epsrels){
		for(double ncall : required_ncall){
			constexpr int ndim = 8;
			params.ncall = ncall;
			common_header_mcubes_time_and_call<F_1_8D,ndim>("f1", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_2_8D,ndim>("f2", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_3_8D,ndim>("f3", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_4_8D,ndim>("f4", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_5_8D,ndim>("f5", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_6_8D,ndim>("f6", epsrel, params, outfile);
		}
	}
	
	for(double ncall : required_ncall){
		for(double epsrel : epsrels){
			constexpr int ndim = 7;
			params.ncall = ncall;
			common_header_mcubes_time_and_call<F_1_7D,ndim>("f1", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_2_7D,ndim>("f2", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_3_7D,ndim>("f3", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_4_7D,ndim>("f4", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_5_7D,ndim>("f5", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_6_7D,ndim>("f6", epsrel, params, outfile);
		}
	}
	
	for(double ncall : required_ncall){
		for(double epsrel : epsrels){
			constexpr int ndim = 6;
			params.ncall = ncall;
			common_header_mcubes_time_and_call<F_1_6D,ndim>("f1", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_2_6D,ndim>("f2", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_3_6D,ndim>("f3", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_4_6D,ndim>("f4", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_5_6D,ndim>("f5", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_6_6D,ndim>("f6", epsrel, params, outfile);
		}	
	}
	
	for(double ncall : required_ncall){
		for(double epsrel : epsrels){
			constexpr int ndim = 5;
			params.ncall = ncall;
			common_header_mcubes_time_and_call<F_1_5D,ndim>("f1", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_2_5D,ndim>("f2", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_3_5D,ndim>("f3", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_4_5D,ndim>("f4", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_5_5D,ndim>("f5", epsrel, params, outfile);
			common_header_mcubes_time_and_call<F_6_5D,ndim>("f6", epsrel, params, outfile);
		}	
	}
		
	outfile.close();
    return 0;
}

