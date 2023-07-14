#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "common/cuda/integrands.cuh"

int
main()
{

  std::vector<double> epsrels = {
    /*1.e-3, 1.e-4, 1.e-5, 1.e-6,*/ 1.e-7, 1.e-8, 1.e-9};
  std::ofstream outfile("pagani_genz_integrals_high_prec.csv");
  constexpr bool use_custom = false;
  constexpr int debug = 0;
  constexpr int num_runs = 10;
  outfile << "id, ndim, use_custom, true_value,  ,epsabs, estimate, errorest, "
             "nregions, completed_iters, status, time"
          << std::endl;

  for (double epsrel : epsrels) {
    constexpr int ndim = 8;
    clean_time_and_call<F_1_8D, ndim, use_custom, debug, num_runs>(
      "f1", epsrel, outfile);
    clean_time_and_call<F_2_8D, ndim, use_custom, debug, num_runs>(
      "f2", epsrel, outfile);
    clean_time_and_call<F_3_8D, ndim, use_custom, 0, 1>("f3", epsrel, outfile);
    clean_time_and_call<F_4_8D, ndim, use_custom, debug, num_runs>(
      "f4", epsrel, outfile);
    clean_time_and_call<F_5_8D, ndim, use_custom, debug, num_runs>(
      "f5", epsrel, outfile);
    clean_time_and_call<F_6_8D, ndim, use_custom, debug, num_runs>(
      "f6", epsrel, outfile);
  }

  for (double epsrel : epsrels) {
    constexpr int ndim = 7;
    clean_time_and_call<F_1_7D, ndim, use_custom, debug, num_runs>(
      "f1", epsrel, outfile);
    clean_time_and_call<F_2_7D, ndim, use_custom, debug, num_runs>(
      "f2", epsrel, outfile);
    clean_time_and_call<F_3_7D, ndim, use_custom, debug, num_runs>(
      "f3", epsrel, outfile);
    clean_time_and_call<F_4_7D, ndim, use_custom, debug, num_runs>(
      "f4", epsrel, outfile);
    clean_time_and_call<F_5_7D, ndim, use_custom, debug, num_runs>(
      "f5", epsrel, outfile);
    clean_time_and_call<F_6_7D, ndim, use_custom, debug, num_runs>(
      "f6", epsrel, outfile);
  }

  for (double epsrel : epsrels) {
    constexpr int ndim = 6;
    clean_time_and_call<F_1_6D, ndim, use_custom, debug, num_runs>(
      "f1", epsrel, outfile);
    clean_time_and_call<F_2_6D, ndim, use_custom, debug, num_runs>(
      "f2", epsrel, outfile);
    clean_time_and_call<F_3_6D, ndim, use_custom, debug, num_runs>(
      "f3", epsrel, outfile);
    clean_time_and_call<F_4_6D, ndim, use_custom, debug, num_runs>(
      "f4", epsrel, outfile);
    clean_time_and_call<F_5_6D, ndim, use_custom, debug, num_runs>(
      "f5", epsrel, outfile);
    clean_time_and_call<F_6_6D, ndim, use_custom, debug, num_runs>(
      "f6", epsrel, outfile);
  }

  for (double epsrel : epsrels) {
    constexpr int ndim = 5;
    clean_time_and_call<F_1_5D, ndim, use_custom, debug, num_runs>(
      "f1", epsrel, outfile);
    clean_time_and_call<F_2_5D, ndim, use_custom, debug, num_runs>(
      "f2", epsrel, outfile);
    clean_time_and_call<F_3_5D, ndim, use_custom, debug, num_runs>(
      "f3", epsrel, outfile);
    clean_time_and_call<F_4_5D, ndim, use_custom, debug, num_runs>(
      "f4", epsrel, outfile);
    clean_time_and_call<F_5_5D, ndim, use_custom, debug, num_runs>(
      "f5", epsrel, outfile);
    clean_time_and_call<F_6_5D, ndim, use_custom, debug, num_runs>(
      "f6", epsrel, outfile);
  }

  outfile.close();

  return 0;
}
