#ifndef CUDACUHRE_QUAD_GPUQUAD_CUHRE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_CUHRE_CUH

#include "../util/cudaMemoryUtil.h"
#include "../util/cudaTimerUtil.h"

#include "../util/Volume.cuh"
#include "Kernel.cuh"
#include <mpi.h>
#include <stdio.h>
#include <map>
#include <chrono>

namespace quad {
#if TIMING_DEBUG == 1
  timer::event_pair timer_one;
#endif

  template <typename T, int NDIM>
  class Cuhre {

    // Debug message
    char msg[256];

    // Quadrature Parameters
    // int NDIM;
    int KEY;

    // Verbose
    int VERBOSE;
    int numDevices;

    int argc;
    char** argv;

    T epsrel, epsabs;
    Kernel<T, NDIM>* kernel;
    std::ofstream log;

  public:
    Cuhre(int pargc,
          char** pargv,
          int key = 0,
          int verbose = 0,
          int numDevices = 1)
    {
	  QuadDebug(cudaDeviceReset());
      argc = pargc;
      argv = pargv;
      // NDIM = dim;
      KEY = key;
      VERBOSE = verbose;
      this->numDevices = numDevices;
      kernel = new Kernel<T, NDIM>(std::cout);
      kernel->InitKernel(KEY, VERBOSE, numDevices);
    }

    ~Cuhre()
    {	  
      delete kernel;
	  printf("Deleted Kernel\n");
    }

#define BUFSIZE 256
#define TAG 0

    void
    MPI_CLIENT_checkCUDA(int nodeRank, T epsrel, T epsabs)
    {
      MPI_Status stat;
      int devCount = 0, namelen = 0;
      char processor_name[MPI_MAX_PROCESSOR_NAME];
      // char idstr[256], idstr2[256], buff[BUFSIZE];
      char idstr[320], idstr2[320], buff[BUFSIZE];
      MPI_Recv(buff, BUFSIZE, MPI_CHAR, 0, TAG, MPI_COMM_WORLD, &stat);
      MPI_Get_processor_name(processor_name, &namelen);
      cudaGetDeviceCount(&devCount);
      buff[0] = '\0';
      idstr[0] = '\0';
      if (devCount == 0) {
        // ihavecuda=0;
        sprintf(idstr, "-|%s|%5d|%4d|NONE", processor_name, nodeRank, devCount);
      } else {
        // ihavecuda=1;
        if (devCount >= 1) {
          sprintf(idstr, "+|%s|%5d|%4d|", processor_name, nodeRank, devCount);
          idstr2[0] = '\0';
          for (int gpuNode = 0; gpuNode < devCount; ++gpuNode) {
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, gpuNode);
            sprintf(idstr2, " %s (%i) ", devProp.name, gpuNode);
            strncat(idstr, idstr2, BUFSIZE);
          }
        } else {
          cudaDeviceProp devProp;
          cudaGetDeviceProperties(&devProp, nodeRank);
          sprintf(idstr,
                  "%s|%5d|%4d|%s",
                  processor_name,
                  nodeRank,
                  devCount,
                  devProp.name);
        }
      }
      strncat(buff, idstr, BUFSIZE);
      MPI_Send(buff, BUFSIZE, MPI_CHAR, 0, TAG, MPI_COMM_WORLD);
    }

    template <typename IntegT>
    void
    MPI_CLIENT_receiveQuadratureData(IntegT* d_integrand,
                                     int nodeRank,
                                     int index,
                                     Volume<T, NDIM>* dvol,
                                     T* numInfo)
    { // numCUDADevices, unsigned long int numRegions){

      unsigned long int numCUDADevices = (unsigned long int)numInfo[0];
      unsigned long int numRegions = (unsigned long int)numInfo[1];
      MPI_Status stat;
      int numDevicesAtNode = 0;
      cudaGetDeviceCount(&numDevicesAtNode);
      size_t numRegionsPerDevice = numRegions / numCUDADevices;
      size_t numRegionsPerNode = numDevicesAtNode * numRegionsPerDevice;

      int startIndex = index * numRegionsPerDevice;
      int endIndex = (index + numDevicesAtNode) * numRegionsPerDevice;
      if (index == (numCUDADevices - numDevicesAtNode)) {
        endIndex = numRegions;
      }
      numRegionsPerNode = endIndex - startIndex;
      if (VERBOSE) {
        log << "\nNode " << nodeRank << "\nData start index : " << startIndex
            << "\nData end index : " << endIndex
            << "\nNumber of regions per node : " << numRegionsPerNode
            << std::endl;
      }
      size_t dataSize = 2 * numRegionsPerNode * NDIM;
      T* data = 0;
      data = (T*)malloc(sizeof(T) * dataSize);
      MPI_Recv(data, dataSize, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &stat);

      Kernel<T, NDIM>* thkernel = new Kernel<T, NDIM>(log);
      thkernel->InitKernel(KEY, VERBOSE, numDevicesAtNode);

      thkernel->setRegionsData(data, numRegionsPerNode);
      T th_integral = 0, th_error = 0;
      size_t th_nregions = 0, th_neval = 0;

      int errorFlag = 0;
      T* optionalInfo = (T*)malloc(sizeof(T) * 2);

      if (thkernel->getNumActiveRegions() > 0) {
        timer::event_pair timer_node;
        if (VERBOSE) {
#if TIMING_DEBUG == 1
          timer::start_timer(&timer_node);
#endif
        }
        errorFlag = thkernel->IntegrateSecondPhase(d_integrand,
                                                   epsrel,
                                                   epsabs,
                                                   th_integral,
                                                   th_error,
                                                   th_nregions,
                                                   th_neval,
                                                   optionalInfo);

        if (VERBOSE) {
          log << "\nIntegral:" << th_integral << "\nError:" << th_error
              << "\nRegions:" << th_nregions
              << "\nFunction Evaluations:" << th_neval
              << "\nError:" << errorFlag << std::endl;

#if TIMING_DEBUG == 1
          T time = timer::stop_timer_returntime(&timer_node, "Second Phase");
          sprintf(
            msg, "Second Phase time at node %d : %.2lfms", nodeRank, time);
          Print(msg);
#endif
        }
      }
      T result[6];
      result[0] = th_integral;
      result[1] = th_error;
      result[2] = th_nregions;
      result[3] = th_neval;
      result[4] = errorFlag;
      result[5] = optionalInfo[0];
      MPI_Send(result, 6, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
    }

    std::vector<std::string>
    splitString(std::string input, std::string delimiter)
    {
      std::vector<std::string> output;
      char* pch;
      char* str = strdup(input.c_str());
      pch = strtok(str, delimiter.c_str());
      while (pch != nullptr) {
        output.push_back(pch);
        pch = strtok(nullptr, delimiter.c_str());
      }
      free(str);
      return output;
    }

    std::map<int, std::vector<std::string>>
    MPI_MASTER_findActiveCUDANodes(int nodeRank,
                                   int numprocs,
                                   char* processor_name)
    {
      MPI_Status stat;
      char buff[BUFSIZE];

      std::map<std::string, int> nodes;
      std::map<int, std::vector<std::string>> CUDANodeDetails;
      if (VERBOSE) {
        log << "Request for " << numprocs << " processors" << std::endl;
        log << "Spawning from " << processor_name << std::endl;
        log << "CUDA MPI\n" << std::endl;
      }
      for (int node_id = 1; node_id < numprocs; ++node_id) {
        buff[0] = '\0';
        MPI_Send(buff, BUFSIZE, MPI_CHAR, node_id, TAG, MPI_COMM_WORLD);
      }

      if (VERBOSE) {
        log << "\n     Probing nodes..." << std::endl;
        log << "Node          Psid  CUDA Cards (devID)\n";
        log << "------------ ----- ---- ----------\n";
      }

      for (int node_id = 1; node_id < numprocs; ++node_id) {
        MPI_Recv(buff, BUFSIZE, MPI_CHAR, node_id, TAG, MPI_COMM_WORLD, &stat);
        if (VERBOSE) {
          log << buff << std::endl;
        }
        if (buff[0] == '+') {
          std::string str(buff);
          std::vector<std::string> nodeDetails = splitString(str, "|");
          std::string nodeName = nodeDetails[1];
          if (nodeName.compare(processor_name) != 0 &&
              nodes.find(nodeName) == nodes.end()) {
            CUDANodeDetails[node_id] = nodeDetails;
            nodes[nodeName] = node_id;
          }
        }
      }
      if (VERBOSE) {
        log << "\nNodes with CUDA Cards...\n";
        log << "Psid | Node        | CUDA Cards\n";
        log << "-----  ------------- ----- \n";
        for (std::map<int, std::vector<std::string>>::iterator it =
               CUDANodeDetails.begin();
             it != CUDANodeDetails.end();
             ++it) {
          log << it->first << " | " << it->second[1].c_str() << " | "
              << it->second[3].c_str() << std::endl;
        }
      }
      return CUDANodeDetails;
    }

    template <typename IntegT>
    int
    MPI_MASTER_integrateSecondPhase(Kernel<T, NDIM>* thkernel,
                                    IntegT* d_integrand,
                                    unsigned long int numCUDADevices,
                                    T& integral,
                                    T& error,
                                    size_t& nregions,
                                    size_t& neval,
                                    Volume<T, NDIM>* dvol,
                                    T* optionalInfo)
    {
      int devCount = 0;
      cudaGetDeviceCount(&devCount);

      int index = 0, nodeRank = 0;
      size_t numRegions = thkernel->getNumActiveRegions();

      size_t numRegionsPerDevice = numRegions / numCUDADevices;
      size_t numRegionsPerNode = numRegionsPerDevice * devCount;
      int startIndex = index * numRegionsPerDevice;
      int endIndex = (index + devCount) * numRegionsPerDevice;
      if (index == (numCUDADevices - 1)) {
        endIndex = numRegions;
      }
      numRegionsPerNode = endIndex - startIndex;
      if (VERBOSE) {
        log << "\nNode " << nodeRank << "\nData start index : " << startIndex
            << "\nData end index : " << endIndex
            << "\nNumber of regions per node : " << numRegionsPerNode
            << std::endl;
      }
      T* data = thkernel->getRegions(numRegionsPerNode, startIndex);

      thkernel->setRegionsData(data, numRegionsPerNode);

#if TIMING_DEBUG == 1
      timer::event_pair timer_node;
      timer::start_timer(&timer_node);
#endif
      int errorFlag = thkernel->IntegrateSecondPhase(d_integrand,
                                                     epsrel,
                                                     epsabs,
                                                     integral,
                                                     error,
                                                     nregions,
                                                     neval,
                                                     optionalInfo);

      if (VERBOSE) {
#if TIMING_DEBUG == 1
        T time = timer::stop_timer_returntime(&timer_node, "Second Phase");
        sprintf(msg, "Second Phase time at node 0 : %.2lfms", time);
        Print(msg);
#endif
      }
      if (VERBOSE) {
        log << "\nIntegral:" << integral << "\nError:" << error
            << "\nRegions:" << nregions << "\nFunction Evaluations:" << neval
            << "\nError:" << errorFlag << std::endl;
      }
      return errorFlag;
    }

    void
    MPI_MASTER_sendQuadratureData(
      Kernel<T, NDIM>* thkernel,
      std::map<int, std::vector<std::string>> CUDANodeDetails,
      int numCUDADevices)
    {
      int masterDevCount = 0;
      cudaGetDeviceCount(&masterDevCount);
      int index = masterDevCount;
      size_t numRegions = thkernel->getNumActiveRegions();
      for (std::map<int, std::vector<std::string>>::iterator it =
             CUDANodeDetails.begin();
           it != CUDANodeDetails.end();
           ++it) {
        int node_id = it->first;
        size_t numRegionsPerDevice = numRegions / numCUDADevices;
        int numDevicesAtNode = atoi(it->second[3].c_str());
        size_t numRegionsPerNode = numRegionsPerDevice * numDevicesAtNode;
        int startIndex = index * numRegionsPerDevice;
        int endIndex = (index + numDevicesAtNode) * numRegionsPerDevice;
        if (index == (numCUDADevices - numDevicesAtNode)) {
          endIndex = numRegions;
        }
        numRegionsPerNode = endIndex - startIndex;
        if (VERBOSE) {
          log << "\nNode " << node_id << "\nData start index : " << startIndex
              << "\nData end index : " << endIndex
              << "\nNumber of regions per node : " << numRegionsPerNode
              << std::endl;
        }

        T* data = thkernel->getRegions(numRegionsPerNode, startIndex);
        size_t dataSize = 2 * numRegionsPerNode * NDIM;
        MPI_Send(data, dataSize, MPI_DOUBLE, node_id, TAG, MPI_COMM_WORLD);

        index += numDevicesAtNode;
      }
    }

    template <typename IntegT>
    T
    MPI_MASTER_integrateFirstPhase(Kernel<T, NDIM>* thkernel,
                                   IntegT* d_integrand,
                                   T epsrel,
                                   T epsabs,
                                   T& integral,
                                   T& error,
                                   size_t& nregions,
                                   size_t& neval,
                                   Volume<T, NDIM>* dvol)
    {

      if (VERBOSE) {
        sprintf(msg,
                "Cuhre input parameters:\nndim %i\nepsrel %f\nepsabs %f\nkey "
                "%i\n\n",
                NDIM,
                epsrel,
                epsabs,
                KEY);
        Print(msg);
      }
      int masterDevCount = 0;
      cudaGetDeviceCount(&masterDevCount);

      thkernel->InitKernel(KEY, VERBOSE, masterDevCount);
      thkernel->GenerateInitialRegions();

      if (VERBOSE) {
        // Show memory usage of GPU
        size_t free_byte, total_byte;
        QuadDebug(cudaMemGetInfo(&free_byte, &total_byte));

        T free_db = (T)free_byte;
        T total_db = (T)total_byte;
        T used_db = total_db - free_db;
        sprintf(msg,
                "\nMemory Usages:\nGPU memory usage\t: used = %.2f MB, free = "
                "%.2f MB, total = %.2f MB\n",
                used_db / 1024.0 / 1024.0,
                free_db / 1024.0 / 1024.0,
                total_db / 1024.0 / 1024.0);
        Print(msg);
      }

      T firstPhaseTime = 0;
#if TIMING_DEBUG == 1
      timer::start_timer(&timer_one);
#endif

      thkernel->IntegrateFirstPhase(
        d_integrand, epsrel, epsabs, integral, error, nregions, neval, dvol);

#if TIMING_DEBUG == 1
      firstPhaseTime = timer::stop_timer_returntime(&timer_one, "First Phase");
#endif
      return firstPhaseTime;
    }
	
    template <typename IntegT>
    int
    MPI_INTEGRATE(IntegT* integrand,
                  T epsrel,
                  T epsabs,
                  T& integral,
                  T& error,
                  size_t& nregions,
                  size_t& neval,
                  Volume<T, NDIM>* dvol)
    {
      int numprocs, rank, namelen, id;
      int errorFlag = 0;
      char processor_name[MPI_MAX_PROCESSOR_NAME];
      // commented out by Ioannis, throws warning
      // freopen(
      //   "/dev/null", "w", stderr); // Hide errors from nodes with no CUDA
      //   cards
      // MPI_Init(&argc,&argv);
      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Get_processor_name(processor_name, &namelen);
      MPI_Comm_rank(MPI_COMM_WORLD, &id);

      IntegT* d_integrand = 0;
      cudaMalloc((void**)&d_integrand, sizeof(IntegT));
      cudaMemcpy(
        d_integrand, integrand, sizeof(IntegT), cudaMemcpyHostToDevice);

      if (id == 0) {
        T t1, t2;
        t1 = MPI_Wtime();

        std::string logfilename(processor_name);
        logfilename += ".Mlog";
        if (VERBOSE)
          log.open(logfilename.c_str());

        Kernel<T, NDIM>* thkernel = new Kernel<T, NDIM>(log);
        FIRST_PHASE_MAXREGIONS *= (numprocs * 4); // TODO:Assuming four devices

        T firstPhaseTime = 0;

        firstPhaseTime = MPI_MASTER_integrateFirstPhase(thkernel,
                                                        d_integrand,
                                                        epsrel,
                                                        epsabs,
                                                        integral,
                                                        error,
                                                        nregions,
                                                        neval,
                                                        dvol);

        size_t numRegions = thkernel->getNumActiveRegions();
        std::map<int, std::vector<std::string>> CUDANodeDetails =
          MPI_MASTER_findActiveCUDANodes(id, numprocs, processor_name);

        int totalCUDADevices = 0;
        for (std::map<int, std::vector<std::string>>::iterator it =
               CUDANodeDetails.begin();
             it != CUDANodeDetails.end();
             ++it) {
          totalCUDADevices += atoi(it->second[3].c_str());
        }
        int masterDevCount = 0;
        cudaGetDeviceCount(&masterDevCount);
        // Total number of cudaDevices = Client CUDA devices + master
        int numCUDADevices = totalCUDADevices + masterDevCount;

        // numInfo => 0->numCUDADevices, 1->numRegions, 2 -> num of CUDA nodes
        T numInfo[5];
        numInfo[0] = (T)numCUDADevices;
        numInfo[1] = (T)numRegions;
        numInfo[2] = (T)CUDANodeDetails.size();
        numInfo[3] = integral;
        numInfo[4] = error;
        int activeCUDADevicesIdx[numprocs], index = masterDevCount;
        for (int node_id = 1; node_id < numprocs; ++node_id) {
          MPI_Send(&numInfo, 5, MPI_DOUBLE, node_id, TAG, MPI_COMM_WORLD);
          activeCUDADevicesIdx[node_id] = -1;
        }

        for (std::map<int, std::vector<std::string>>::iterator it =
               CUDANodeDetails.begin();
             it != CUDANodeDetails.end();
             ++it) {
          activeCUDADevicesIdx[it->first] = index;
          index += atoi(it->second[3].c_str());
        }

        // Send node information to the nodes with CUDA device
        for (int node_id = 1; node_id < numprocs; ++node_id) {
          MPI_Send(activeCUDADevicesIdx,
                   numprocs,
                   MPI_INT,
                   node_id,
                   TAG,
                   MPI_COMM_WORLD);
        }

        // Send Quadrature data
        MPI_MASTER_sendQuadratureData(
          thkernel, CUDANodeDetails, numCUDADevices);

        t2 = MPI_Wtime();
        T mpiCommTime = (t2 - t1) * 1000;

        t1 = MPI_Wtime();

        T* optionalInfo = (T*)malloc(sizeof(T) * 2);
        if (thkernel->getNumActiveRegions() > 0) {
          errorFlag = MPI_MASTER_integrateSecondPhase(thkernel,
                                                      d_integrand,
                                                      numCUDADevices,
                                                      integral,
                                                      error,
                                                      nregions,
                                                      neval,
                                                      dvol,
                                                      optionalInfo);
        }

        // Get result
        MPI_Status stat;
        T kernelTime = optionalInfo[0]; // Time from Node 0
        for (std::map<int, std::vector<std::string>>::iterator it =
               CUDANodeDetails.begin();
             it != CUDANodeDetails.end();
             ++it) {
          T result[6];
          MPI_Recv(
            result, 6, MPI_DOUBLE, it->first, TAG, MPI_COMM_WORLD, &stat);
          integral += result[0];
          error += result[1];
          nregions += (size_t)result[2];
          neval += (size_t)result[3];
          errorFlag += (size_t)result[4];
          if (kernelTime < result[5])
            kernelTime = result[5];
        }

        if (error <= MaxErr(integral, epsrel, epsabs))
          errorFlag = 0;
        t2 = MPI_Wtime();
        T totalTime = (t2 - t1) * 1000;

#if TIMING_DEBUG == 1
        printf("First Phase execution time:\t%.2lf\n", firstPhaseTime);
        printf("MPI Communication overhead:\t%.2lf\n", mpiCommTime);
        printf("MPI Kernel Execution Time:\t%.2lf\n", kernelTime);
#endif
        printf(
          "%d\t%e\t%.10lf\t%.10f\t%-15ld\t%-15ld\t%d\t%-10.2lf\t%-10.2lf\n",
          NDIM,
          epsrel,
          integral,
          error,
          nregions,
          neval,
          errorFlag,
          kernelTime + firstPhaseTime,
          mpiCommTime + totalTime);

        // printf( "\nTotal Execution time with %d nodes - %.2fms\n",
        // numCUDADevices, (t2 - t1)*1000 );

        MPI_Finalize();
      } else {
        MPI_Status stat;
        MPI_CLIENT_checkCUDA(id, epsrel, epsabs);

        T numInfo[5];
        // Get number of CUDA devices
        MPI_Recv(numInfo, 5, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &stat);

        // array(node_id) => index/-1
        int activeCUDADevicesIdx[numprocs];
        MPI_Recv(activeCUDADevicesIdx,
                 numprocs,
                 MPI_INT,
                 0,
                 TAG,
                 MPI_COMM_WORLD,
                 &stat);

        if (activeCUDADevicesIdx[id] != -1) {
          std::string logfilename(processor_name);
          logfilename = logfilename + ".log";
          if (VERBOSE)
            log.open(logfilename.c_str());
          MPI_CLIENT_receiveQuadratureData(
            d_integrand, id, activeCUDADevicesIdx[id], dvol, numInfo);
        }

        MPI_Finalize();
      }
      // MPI_Finalize();
      return errorFlag;
    }
	
    template <typename IntegT>
    cuhreResult
    integrate(IntegT integrand,
              double epsrel,
              double epsabs,
              Volume<T, NDIM>* volume = nullptr,
              int verbosity = 0,
              int Final = 0,
			  int phase1type = 0)
    {
      cuhreResult res;

      res.value = 0;
      res.error = 0;
      res.neval = 0;
      res.nregions = 0;

      res.status = integrate<IntegT>(integrand,
                        epsrel,
                        epsabs,
                        res.value,
                        res.error,
                        res.nregions,
                        res.neval,
                        volume,
                        verbosity,
                        Final,
						phase1type);
      return res;
    }
	
    template <typename IntegT>
    int
    integrate(IntegT integrand,
              T epsrel,
              T epsabs,
              T& integral,
              T& error,
              size_t& nregions,
              size_t& neval,
              Volume<T, NDIM>* volume = nullptr,
              int verbosity = 0,
              int Final = 0,
			  int phase1type = 0)
    {
      this->epsrel = epsrel;
      this->epsabs = epsabs;
	  
      kernel->SetFinal(Final);
      kernel->SetVerbosity(verbosity);
	  kernel->SetPhase_I_type(phase1type);
	  
      int errorFlag = 0;
      int numprocs = 0;

      IntegT* d_integrand = 0;
      cudaMalloc((void**)&d_integrand, sizeof(IntegT));
      cudaMemcpy(
        d_integrand, &integrand, sizeof(IntegT), cudaMemcpyHostToDevice);

      if (numprocs > 1) {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        errorFlag = MPI_INTEGRATE(
          &integrand, epsrel, epsabs, integral, error, nregions, neval, volume);
        MPI_Finalize();
      } else {
		
		bool convergence = false;
        kernel->GenerateInitialRegions();
        FIRST_PHASE_MAXREGIONS *= numDevices;
		
		//using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
		//auto t1 = std::chrono::high_resolution_clock::now();
		
		if(!phase1type){
			convergence = kernel->IntegrateFirstPhase(d_integrand,
                                    epsrel,
                                    epsabs,
                                    integral,
                                    error,
                                    nregions,
                                    neval,
                                    volume);
		}
		else{							
			convergence = kernel->IntegrateFirstPhaseDCUHRE(d_integrand,
										epsrel,
										epsabs,
										integral,
										error,
										nregions,
										neval,
										volume);
		}
		//MilliSeconds dt = std::chrono::high_resolution_clock::now() - t1;
		//std::cout<<"Phase 1 time in ms:"<< dt.count()<<std::endl;
		
		errorFlag = kernel->GetErrorFlag();
        T* optionalInfo = nullptr;
		
		//auto t2 = std::chrono::high_resolution_clock::now();
		//printf("number of active regions before phase 2:%lu\n", kernel->getNumActiveRegions());
        if (kernel->getNumActiveRegions() > 0 && convergence == false) {
		
			errorFlag = kernel->IntegrateSecondPhase(d_integrand,
                                                   epsrel,
                                                   epsabs,
                                                   integral,
                                                   error,
                                                   nregions,
                                                   neval,
                                                   optionalInfo);
			printf("ratio:%f error:%f MaxErr:%f\n",  error/MaxErr(integral, epsrel, epsabs), error,MaxErr(integral, epsrel, epsabs) );
		 
			if (error <= MaxErr(integral, epsrel, epsabs)) {
				errorFlag = 0;
			}
        }
      }
	  
	  cudaFree(d_integrand);
      return errorFlag;
    }
  };
}

#endif
