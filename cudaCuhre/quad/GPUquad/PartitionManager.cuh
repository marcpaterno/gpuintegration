#ifndef CUDACUHRE_QUAD_GPUQUAD_PARTITIONMANAGER_CUH
#define CUDACUHRE_QUAD_GPUQUAD_PARTITIONMANAGER_CUH
	
#include <algorithm>
#include "../util/cudaMemoryUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <vector>

namespace quad{
	template <class K>
    void
    display(K* array, size_t size, std::string msg = std::string(), int specific_index = -1, int specific_index2 = -1)
    {
      K* tmp = (K*)malloc(sizeof(K) * size);
      cudaMemcpy(tmp, array, sizeof(K) * size, cudaMemcpyDeviceToHost);
	  
	  if(specific_index!= -1){
		//printf("About to display\n");
		//printf("display, %i, %e\n", specific_index, (double)tmp[specific_index]);
		
		if(specific_index2 != -1)
			//printf("display, %i, %e\n", specific_index2, (double)tmp[specific_index2]);	
		return;
	  }
		  
      for (size_t i = 0; i < size; ++i) {
        //printf("%s, %i, %e\n", msg.c_str(), i, (double)tmp[i]);
      }
	  
      free(tmp);
    }

template<size_t NDIM>
class Partition{
	public:
	
	Partition()
	{
		regions = nullptr;
		regionsLength = nullptr;
		parentsIntegral = nullptr;
		parentsError = nullptr;
	}
	
	
	void Deallocate()
	{
		free(regions);
		free(regionsLength);
		free(parentsIntegral);
		free(parentsError);
	}
	
	void ShallowCopy(double* low_bounds, double* length_per_boud, double* parentEstimates, double* parentErrs, size_t size, int depthReached)
	{
		//printf("inside shallow copy with %lu regions\n", size);
		numRegions = size;
		regions = low_bounds;
		regionsLength = length_per_boud;
		parentsIntegral = parentEstimates;
		parentsError = parentErrs;
		depth = depthReached;
		//printf("Saving parition with depth %i\n", depth);
	}
	
	void DeepCopyHostToDevice(Partition<NDIM> source)
	{
		numRegions = source.numRegions;
		depth = source.depth;
		//printf("Setting depth to %i for partition to be loaded for next iteration\n", depth);
		QuadDebug(cudaMemcpy(regions, source.regions, sizeof(double) * source.numRegions * NDIM, cudaMemcpyHostToDevice));
		CudaCheckError();	
		QuadDebug(cudaMemcpy(regionsLength, source.regionsLength, sizeof(double) * source.numRegions * NDIM, cudaMemcpyHostToDevice));
		CudaCheckError();	
		QuadDebug(cudaMemcpy(parentsIntegral, source.parentsIntegral, sizeof(double) * source.numRegions, cudaMemcpyHostToDevice));
		CudaCheckError();				   
		QuadDebug(cudaMemcpy(parentsError, source.parentsError, sizeof(double) * source.numRegions, cudaMemcpyHostToDevice));
		CudaCheckError();	
	}
	
	void HostAllocate(size_t size, int sourcePartDepth)
	{
		//printf("Started host allocation\n");
		regions = (double*)Host->AllocateMemory(&regions, sizeof(double) *  size * NDIM);	
		regionsLength = (double*)Host->AllocateMemory(&regionsLength, sizeof(double) *  size * NDIM);	
		parentsIntegral = (double*)Host->AllocateMemory(&parentsIntegral, sizeof(double) * size);
		parentsError = (double*)Host->AllocateMemory(&parentsError, sizeof(double) * size);
		numRegions = size;
		depth = sourcePartDepth;
		//printf("inside hostAllocate partion has size:%lu\n", numRegions);
	}
	
	void DeviceAllocate(size_t size)
	{
		QuadDebug(Device->AllocateMemory((void**)&regions, sizeof(double) *  size * NDIM));	
		QuadDebug(Device->AllocateMemory((void**)&regionsLength, sizeof(double) *  size * NDIM));	
		QuadDebug(Device->AllocateMemory((void**)&parentsIntegral, sizeof(double) * size));
		QuadDebug(Device->AllocateMemory((void**)&parentsError, sizeof(double) * size));
		numRegions = size;
	}
	
	double* regions;
	double* regionsLength;
	double* parentsIntegral;
	double* parentsError;
	int depth;
	
	HostMemory<double>* Host;
	DeviceMemory<double>* Device;	
		
	size_t numRegions;
};

template<size_t NDIM>
class PartitionManager{
	public:
	
		HostMemory<double>* Host;
		DeviceMemory<double>* Device;
		
		double queued_reg_estimate;
		double queued_reg_errorest;
		
		std::vector<Partition<NDIM>> partitions;
		std::vector<size_t> partitionSizes;
		std::vector<double> partionContributionsIntegral;
		std::vector<double> partionContributionsError;
		size_t numPartitions;    
	
	void Init(HostMemory<double>* host, DeviceMemory<double>* device)
	{
		Host = host;
		Device = device;
		numPartitions = 0;
		queued_reg_estimate = 0.;
		queued_reg_errorest = 0.;
	}
	
	void ExpandNumPartitions(size_t numToExpandBy)
	{
		//size_t* temppartitionSizes 				 = new size_t[numPartitions + numToExpandBy];
		//double* tempPartionContributionsIntegral = new double[numPartitions + numToExpandBy];
		//double* tempPartionContributionsError  	 = new double[numPartitions + numToExpandBy];
		
		//std::copy(partitionSizes, partitionSizes + numPartitions, temppartitionSizes);
		//std::copy(partionContributionsIntegral, partionContributionsIntegral + numPartitions, tempPartionContributionsIntegral);
		//std::copy(partionContributionsError, partionContributionsError + numPartitions, tempPartionContributionsError);
	}
	
	size_t partitionRunSum(size_t id) 
	{
		//consider only the partitions that are to be added in this iteration
		//that's why we add numPartitions to i
		size_t temp = 0;
		for(int i=0; i<id; i++)
			temp += partitionSizes[numPartitions+i];
		return temp;
	}
	
	void SetPartitionContributions(double* dParentsIntegral, double* dParentsError)
	{
		//i don't like i+numPartitions indexing
		thrust::device_ptr<double> wrapped_ptr;
		wrapped_ptr = thrust::device_pointer_cast(dParentsIntegral);
		
		for(int i=0; i<4; i++){
			size_t partionNumParents = partitionSizes[i + numPartitions]/2;
			size_t parentsProcessed = partitionRunSum(i)/2;
			//printf("Setting partition %i parents from %lu to %lu\n",  i, parentsProcessed, parentsProcessed + partionNumParents);
			partionContributionsIntegral.push_back(thrust::reduce(wrapped_ptr + parentsProcessed, wrapped_ptr + parentsProcessed + partionNumParents));
			
		}
		
		wrapped_ptr = thrust::device_pointer_cast(dParentsError);
		
		for(int i=0; i<4; i++){
			size_t partionNumParents = partitionSizes[i+ numPartitions]/2;
			size_t parentsProcessed = partitionRunSum(i)/2;
			partionContributionsError.push_back(thrust::reduce(wrapped_ptr + parentsProcessed, wrapped_ptr + parentsProcessed + partionNumParents));
		}

		/*for(int i=0; i<numPartitions + 4; i++)
		{
			printf("Contributions %i/%lu %e +- %e size:%lu\n", i, numPartitions, partionContributionsIntegral[i], partionContributionsError[i], partitionSizes[i]);
		}*/
	}

	void GetEvenQuarterSize(size_t num, size_t& quarterSize, size_t& remainder)
	{
		//printf("GetEvenQuarterSize with num:%lu\n", num);
		remainder = 0;
		auto isOdd = [](size_t num)->bool
		{
			return num % 2;
		};
		
		while(isOdd(num/4)){
			remainder += 2;
			num -= 2;
		}
		quarterSize = num/4;
		//printf("returning quarter:%lu , remainder:%lu\n", quarterSize, remainder);
	};	

	void SetpartitionSizess(size_t size)
	{	
		//printf("Inside set partition size numPartitions:%lu size:%lu\n", numPartitions, size);
		const size_t numNewPartitions = 4;
		assert(numNewPartitions == 4);
		assert(!(size % 2));
		//this sets four partitions, each of which must be of even size
		//the paritions dont' need to be equal to each other 
			
		/*auto isOdd = [](size_t num)->bool
		{
			return num % 2;
		};*/
					
		size_t extra = size % numNewPartitions; //extra=2
		size_t floor_quarter_size =  size/numNewPartitions; //quarter_size: 4474069
		GetEvenQuarterSize(size, floor_quarter_size, extra);
		
		for(int i=0; i<numNewPartitions; i++)
			partitionSizes.push_back(floor_quarter_size);
		partitionSizes[numNewPartitions+numPartitions-1] += extra;
		
		//printf("splitting %lu four ways, extra:%lu floor_quarter_size:%lu\n", size, extra,floor_quarter_size);
		/*if(extra == 0 && !isOdd(floor_quarter_size))
		{
			printf("A\n");
			for(int i=0; i< numNewPartitions; i++){
				//partitionSizes[numPartitions +i] = floor_quarter_size;
				//printf("About to pushback A %lu\n", floor_quarter_size);
				partitionSizes.push_back(floor_quarter_size);
			}
		}
		else if(extra == 0 && isOdd(floor_quarter_size) == true)
		{
			printf("B\n");
			for(int i=0; i<numNewPartitions; i++){
				partitionSizes.push_back((size-numNewPartitions)/numNewPartitions);
			}
			//printf("About to add %lu regions to partition %i ->%lu\n", numNewPartitions, numNewPartitions+partitionSizes[numPartitions + numNewPartitions-1], numPartitions + numNewPartitions-1);
			partitionSizes[numPartitions + numNewPartitions-1] += numNewPartitions;
		}
		else if(extra != 0)
		{
			printf("C\n");
			//extra can only be 0 or 2 if we divide size by 4 and guarantee size is even
			for(int i=0; i<numNewPartitions; i++){	
				//partitionSizes[numPartitions+ i] = floor_quarter_size;
				//printf("About to pushback C %lu\n", floor_quarter_size);
				partitionSizes.push_back((size-extra)/numNewPartitions);
			}
			partitionSizes[numPartitions + numNewPartitions-1] +=  2;
		}
			
		*/
		for(int i=0; i<numPartitions + numNewPartitions; i++){
			assert(partitionSizes[i] % 2 == 0);
			//printf("partion %i size:%lu\n", i, partitionSizes[i]);
		}
		////printf("Done with settign partition sizes\n");
	}
	
	void DeepCopyDeviceToHost(Partition<NDIM>& destPartition, Partition<NDIM> sourcePartition, size_t sourceRegionsFirstIndex)
	{
		
		/*
			Give a big partition from the gpu, and copy it to a smaller partition on the cpu
			we can use Partition.ShallowCopy to quickly organize the Kernel's gpu structures and simplify the interface
		*/
		size_t numPairsInPartition = destPartition.numRegions/2;
		
        for (size_t dim = 0; dim < NDIM; ++dim) {
		  //place left children in first half and then at second half
		  size_t dimStartIndex = dim * sourcePartition.numRegions;
          QuadDebug(cudaMemcpy(destPartition.regions + dim *  destPartition.numRegions,
                               sourcePartition.regions + dimStartIndex + sourceRegionsFirstIndex,
                               sizeof(double) *  numPairsInPartition,
                               cudaMemcpyDeviceToHost));
		  CudaCheckError();				   
		
		  QuadDebug(cudaMemcpy(destPartition.regions + dim *  destPartition.numRegions + destPartition.numRegions/2,
                               sourcePartition.regions + dimStartIndex + sourceRegionsFirstIndex + sourcePartition.numRegions/2,
                               sizeof(double) *  numPairsInPartition,
                               cudaMemcpyDeviceToHost));
			
		  CudaCheckError();	
		  QuadDebug(cudaMemcpy(destPartition.regionsLength + dim *  destPartition.numRegions,
                               sourcePartition.regionsLength + dimStartIndex + sourceRegionsFirstIndex,
                               sizeof(double) *  numPairsInPartition,
                               cudaMemcpyDeviceToHost));
		  CudaCheckError();	

		  QuadDebug(cudaMemcpy(destPartition.regionsLength + dim *  destPartition.numRegions + destPartition.numRegions/2,
                               sourcePartition.regionsLength + dimStartIndex + sourceRegionsFirstIndex + sourcePartition.numRegions/2,
                               sizeof(double) *  numPairsInPartition,
                               cudaMemcpyDeviceToHost));
		  CudaCheckError();	
        }
		
		
		QuadDebug(cudaMemcpy(destPartition.parentsIntegral,
                             sourcePartition.parentsIntegral + sourceRegionsFirstIndex,
                             sizeof(double) * numPairsInPartition,
                             cudaMemcpyDeviceToHost));	
							 
		QuadDebug(cudaMemcpy(destPartition.parentsIntegral + numPairsInPartition,
                             sourcePartition.parentsIntegral + sourceRegionsFirstIndex,
                             sizeof(double) * numPairsInPartition,
                             cudaMemcpyDeviceToHost));	
		
		QuadDebug(cudaMemcpy(destPartition.parentsError,
                             sourcePartition.parentsError + sourceRegionsFirstIndex,
                             sizeof(double) * numPairsInPartition,
                             cudaMemcpyDeviceToHost));	
		  
		QuadDebug(cudaMemcpy(destPartition.parentsError + numPairsInPartition,
                             sourcePartition.parentsError + sourceRegionsFirstIndex,
                             sizeof(double) * numPairsInPartition,
                             cudaMemcpyDeviceToHost));	
		CudaCheckError();
	}
	
	void
    StoreRegionsInHost(Partition<NDIM>& sourcePartition)
    {
	  //printf("storing %lu regions in host with %lu current partitions \n", sourcePartition.numRegions, numPartitions);
	  int numNewPartitions = 4; //always expand by four partitions, we can change this later
	  SetpartitionSizess(sourcePartition.numRegions);
	  SetPartitionContributions(sourcePartition.parentsIntegral, sourcePartition.parentsError);

	  CudaCheckError();
      //Host.ReleaseMemory(curr_hRegions);
      //Host.ReleaseMemory(curr_hRegionsLength);
	  	
      for (int partitionID = 0; partitionID < numNewPartitions; partitionID++) {
		Partition<NDIM> temp;
		partitions.push_back(temp);
        partitions[partitionID + numPartitions].HostAllocate(partitionSizes[partitionID + numPartitions], sourcePartition.depth);
		//printf("after allocating %i new partition\n",partitionID);
		//printf("added partitions have depth%i\n", partitions[partitionID + numPartitions].depth);
      }
		
      CudaCheckError();
	  
      size_t startRegionIndex = 0;
      for (int partitionID = 0; partitionID < numNewPartitions; partitionID++) {
		//printf("About to do DeepCopyDeviceToHost numPartitions:%lu partitionSize:%lu sourceSize:%lu\n", numPartitions, partitions[partitionID + numPartitions].numRegions, sourcePartition.numRegions);  
		DeepCopyDeviceToHost(partitions[partitionID + numPartitions], sourcePartition, startRegionIndex);CudaCheckError();
        startRegionIndex += partitions[partitionID+ numPartitions].numRegions/2;
      }
	  
	  numPartitions += numNewPartitions;
	  //printf("Now have %lu partitions\n", numPartitions);
      CudaCheckError();
    }
	
	void
    SetDeviceRegionsFromHost(Partition<NDIM>& destPartition, Partition<NDIM> sourcePartition)
    {
	  destPartition.DeviceAllocate(sourcePartition.numRegions);
	  CudaCheckError();
	  destPartition.DeepCopyHostToDevice(sourcePartition);
	  CudaCheckError();
	  sourcePartition.Deallocate();
    }
	
	void LoadNextActivePartition(Partition<NDIM>& wrapper)
	{
		StoreRegionsInHost(wrapper);	//1. save current progress in the form of four partitions
		CudaCheckError();printf("Stored regions in host\n");
		size_t maxErrID = 0;			//2. Find the partition with the largest error-estimate
		
		for(size_t i=1; i<numPartitions; i++){
			if(partionContributionsError[maxErrID] <  partionContributionsError[i]){
				maxErrID = i;
			}
		}
		
		for(size_t i=0; i<numPartitions; i++){
			//printf("partition %lu size:%lu\n", i, partitionSizes[i]);
		}
		//maxErrID = 3;
		printf("Loading partition with %e +- %e errorest\n",partionContributionsIntegral[maxErrID] , partionContributionsError[maxErrID]);
		Partition<NDIM> priorityP = partitions[maxErrID];	//3. get a pointer to that host partition
		//printf("PRIORITY p depth %i partition index:%lu\n", priorityP.depth, maxErrID);
		//4. erase it from partition manager
		partitions.erase(partitions.begin() + maxErrID);	
		partionContributionsIntegral.erase(partionContributionsIntegral.begin() + maxErrID);
		partionContributionsError.erase(partionContributionsError.begin() + maxErrID);
		partitionSizes.erase(partitionSizes.begin() + maxErrID);
		numPartitions--;
		
		//5. Free the memory pointed to by current device partition
		//printf("releasing dRegions\n");
		Device->ReleaseMemory(wrapper.regions);CudaCheckError();
		Device->ReleaseMemory(wrapper.regionsLength);CudaCheckError();
		Device->ReleaseMemory(wrapper.parentsIntegral);CudaCheckError();
		Device->ReleaseMemory(wrapper.parentsError);CudaCheckError();
		wrapper.numRegions = 0;
		
		//copy on the device the host partition from step 3.
		SetDeviceRegionsFromHost(wrapper, priorityP);
		CudaCheckError();
		//update statistics on stored regions without including step 3. partition
		queued_reg_estimate = 0;
		queued_reg_errorest = 0;
		
		for(int i=0; i<numPartitions; i++){
			queued_reg_estimate += partionContributionsIntegral[i]; 
			queued_reg_errorest += partionContributionsError[i];
		}
	}

};
	
	
}
#endif