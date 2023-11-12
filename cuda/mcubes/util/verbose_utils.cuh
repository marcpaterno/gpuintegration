#ifndef VERBOSE_UTILS_CUH
#define VERBOSE_UTILS_CUH

#include "common/cuda/cudaMemoryUtil.h"
#include "cuda/mcubes/util/vegas_utils.cuh"

template<int ndim>
struct FuncEval{
  double res = 0;
  double point[ndim] = {0.};
};

// this isn't needed anymore
std::ofstream
GetOutFileVar(std::string filename)
{
  std::ofstream myfile;
  myfile.open(filename.c_str());
  return myfile;
}

template <bool DEBUG_MCUBES, int NDIM>
class IterDataLogger {
  std::ofstream myfile_bin_bounds;
  std::ofstream myfile_randoms;
  std::ofstream myfile_funcevals;
  std::ofstream interval_myfile;
  std::ofstream iterations_myfile;

public:
  FuncEval<NDIM>* funcevals = nullptr;
  double* randoms = nullptr;
  IterDataLogger(uint32_t totalNumThreads,
                 int chunkSize,
                 int extra,
                 int npg,
                 int ndim)
  {
    if constexpr (DEBUG_MCUBES) {
      funcevals = quad::cuda_malloc_managed<FuncEval<NDIM>>(
        (totalNumThreads * chunkSize + extra) * npg);

      myfile_bin_bounds.open("pmcubes_bin_bounds.csv");
      myfile_bin_bounds
        << "iter, dim, bin, bin_length, left, right, contribution\n";
      myfile_bin_bounds.precision(15);

      myfile_randoms.open("pmcubes_random_nums.csv");
      myfile_randoms << "it, cube, chunk, sample, dim, ran00\n";
      myfile_randoms.precision(15);

      myfile_funcevals.precision(15);
      myfile_funcevals.open("pmcubes_funcevals.csv");
      myfile_funcevals << "it, cube, sample, funceval,";
      for(int dim = 0; dim < NDIM; ++dim)
        myfile_funcevals << "dim" + std::to_string(dim) << ",";
      myfile_funcevals <<  std::endl;

      interval_myfile.open("pmcubes_intevals.csv");
      interval_myfile.precision(15);

      iterations_myfile.open("pmcubes_iters.csv");
      iterations_myfile
        << "iter, estimate, errorest, chi_sq, iter_estimate, iter_errorest\n";
      iterations_myfile.precision(15);
    }
  }

  ~IterDataLogger()
  {
    if constexpr (DEBUG_MCUBES) {
      myfile_bin_bounds.close();
      myfile_randoms.close();
      myfile_funcevals.close();
      interval_myfile.close();
      iterations_myfile.close();
    }
  }

  void
  PrintIterResults(int iteration,
                   double estimate,
                   double errorest,
                   double chi_sq,
                   double iter_estimate,
                   double iter_errorest)
  {
    iterations_myfile << iteration << "," << estimate << "," << errorest << ","
                      << chi_sq << "," << iter_estimate << "," << iter_errorest
                      << "\n";
  }

  void
  PrintBins(int iter, double* xi, double* d, int ndim)
  {
    int ndmx1 = Internal_Vegas_Params::get_NDMX_p1();
    int ndmx =  Internal_Vegas_Params::get_NDMX();
    int mxdim_p1 =  Internal_Vegas_Params::get_MXDIM_p1();
    
    for (int dim = 1; dim <= ndim; dim++)
      for (int bin = 1; bin <= ndmx; bin++) {

        double bin_length = xi[dim * ndmx1 + bin] - xi[dim * ndmx1 + bin - 1];
        double left = xi[dim * ndmx1 + bin - 1];
        if (bin == 1)
          left = 0.;
        double right = xi[dim * ndmx1 + bin];
        double contribution = d[bin * mxdim_p1 + dim];
        myfile_bin_bounds << iter << "," << dim << "," << bin << ","
                          << bin_length << "," << left << "," << right << ","
                          << contribution << "\n";
      }
  }

  void
  PrintRandomNums(int it, int ncubes, int npg, int ndim)
  {

    size_t nums_per_cube = npg * ndim;
    size_t nums_per_sample = ndim;

    for (int cube = 0; cube < ncubes; cube++)
      for (int sample = 1; sample <= npg; sample++){
        size_t index = cube * nums_per_cube * npg + sample ;
        myfile_randoms << it << "," 
                       << cube << "," 
                       << funcevals[index] << ",";
          for (int dim = 1; dim <= ndim; dim++) {


            myfile_randoms  
              << randoms[cube * nums_per_cube * sample ] << "," // same as chunk for single threaded
                          << randoms[index]
                                    << "\n";
            }
        }
          
    
  }

  void
  PrintFuncEvals(int it, int ncubes, int npg, int ndim)
  {
    size_t num_fevals = ncubes * npg;
    for (int cube = 0; cube < ncubes; cube++)
      for (int sample = 0; sample < npg; sample++) {
        int index = npg * cube + sample;
        myfile_funcevals << it << "," 
                        << cube << "," 
                        << sample << ","
                        << funcevals[index].res << ",";
        for(int dim = 0; dim < NDIM; ++dim){
            myfile_funcevals << funcevals[index].point[dim] << ",";
        }
        myfile_funcevals << std::endl;
      }
  }

  void
  PrintIntervals(int ndim,
                 int ng,
                 uint32_t totalNumThreads,
                 int chunkSize,
                 int it)
  {
    /*constexpr int mxdim_p1 = Internal_Vegas_Params::get_MXDIM_p1();

    if(it == 1)
        interval_myfile<<"m, kg[1], kg[2], kg[3], it\n";

    for(uint32_t m = 0; m < totalNumThreads; m++){
        uint32_t kg[mxdim_p1];
        get_indx(m , &kg[1], ndim, ng);

        interval_myfile<<m<<",";
            for(int ii = 1; ii<= ndim; ii++)
                interval_myfile<<kg[ii]<<",";
            interval_myfile<<it<<"\n";
    }*/
  }
};

#endif
