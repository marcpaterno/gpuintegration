#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include "../deviceProp.h"

__global__
void
warmUpKernel(){
  
}
/**
 *Utility for parsing command line arguments
 */
class CommandLineArgs{
 protected:
  std::map<std::string, std::string> pairs;
  
 public:

  //@brief Constructor
  CommandLineArgs(int argc, char **argv){
    using namespace std;
    for (int i = 1; i < argc; i++){
      string arg = argv[i];
      if ((arg[0] != '-') || (arg[1] != '-')){
	continue;
      }
      string::size_type pos;
      string key, val;
      if ((pos = arg.find( '=')) == string::npos) {
	key = string(arg, 2, arg.length() - 2);
	val = "";
      } else {
	key = string(arg, 2, pos - 2);
	val = string(arg, pos + 1, arg.length() - 1);
      }
      pairs[key] = val;
    }
  }

  /**
   * Checks whether a flag "--<flag>" is present in the commandline
   */
  bool CheckCmdLineFlag(const char* arg_name){
    using namespace std;
    map<string, string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end()) {
      return true;
    }
    return false;
  }
  
  /**
   * Returns the value specified for a given commandline parameter --<flag>=<value>
   */
  template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val){
    using namespace std;
    map<string, string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end()) {
      istringstream str_stream(itr->second);
      str_stream >> val;
    }
  }
  
  /**
   * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
   */
  template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals){
    using namespace std;
    // Recover multi-value string
    map<string, string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end()) {
      // Clear any default values
      vals.clear();

      string val_string = itr->second;
      istringstream str_stream(val_string);
      string::size_type old_pos = 0;
      string::size_type new_pos = 0;

      // Iterate comma-separated values
      T val;
      while ((new_pos = val_string.find(',', old_pos)) != string::npos) {
	if (new_pos != old_pos) {
	  str_stream.width(new_pos - old_pos);
	  str_stream >> val;
	  vals.push_back(val);
	}
	// skip over comma
	str_stream.ignore(1);
	old_pos = new_pos + 1;
      }
      
      // Read last value
      str_stream >> val;
      vals.push_back(val);
    }
  } 
 
  /**
   * The number of pairs parsed
   */
  int ParsedArgc(){
    return pairs.size();
  }



  //@brief Initialize Device
  cudaError_t DeviceInit(int dev = -1){
	  
    cudaError_t error = cudaSuccess;
	
    do{
      int deviceCount;
      error = QuadDebug(cudaGetDeviceCount(&deviceCount));
      if (error) 
		break;
      if (deviceCount == 0) {
		fprintf(stderr, "No devices supporting CUDA.\n");
		exit(1);
      }
	  
	printf("DeviceCount:%i\n", deviceCount);
      for(int i = 0; i < deviceCount; i++){
		int gpu_id;
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		
		QuadDebug(cudaSetDevice(i));	// "% num_gpus" allows more CPU threads than GPU devices
		
		QuadDebug(cudaGetDevice(&gpu_id));
		QuadDebug(cudaDeviceReset());
		
		warmUpKernel<<<FIRST_PHASE_MAXREGIONS, BLOCK_SIZE>>>();
		
		QuadDebug(cudaDeviceSynchronize());
		printf("After first warmup\n");
      }	  
		
		if (dev < 0){
			GetCmdLineArgument("device", dev);
		}

		
		if ((dev > deviceCount - 1) || (dev < 0)){
			dev = 0;
		}
		
		error = QuadDebug(cudaSetDevice(dev));
      if (error)
		  break;
     
      size_t free_physmem, total_physmem;
      QuadDebugExit(cudaMemGetInfo(&free_physmem, &total_physmem));


      cudaDeviceProp deviceProp;
      error = QuadDebug(cudaGetDeviceProperties(&deviceProp, dev));
      if (error) 
		  break;

      if (deviceProp.major < 1) {
	fprintf(stderr, "Device does not support CUDA.\n");
	exit(1);
      }
      int verbose = 0;
      if (CheckCmdLineFlag("verbose")){
	GetCmdLineArgument("verbose", verbose);
      }      
	  printf("last\n");
      if (false&&verbose) {
	printf("Using device %d: %s (SM%d, %d SMs, %lld free / %lld total MB physmem, ECC %s)\n",
	       dev,
	       deviceProp.name,
	       deviceProp.major * 100 + deviceProp.minor * 10,
	       deviceProp.multiProcessorCount,
	       (unsigned long long) free_physmem / 1024 / 1024,
	       (unsigned long long) total_physmem / 1024 / 1024,
	       (deviceProp.ECCEnabled) ? "on" : "off");
	fflush(stdout);
      }
      
    }while (0);
    return error;
  }
};

#define INFTY DBL_MAX
#define MAX(a, b) ((a > b)?a:b)
#ifndef MIN
#define MIN(a, b) ((a < b)?a:b)
#endif
#define Zap(d) memset(d, 0, sizeof(d))

#define MaxErr(avg, epsrel, epsabs) MAX(epsrel*fabs(avg), epsabs)
