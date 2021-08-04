#ifndef CUDACUHRE_QUAD_GPUQUAD_PARTITIONMANAGER_CUH
#define CUDACUHRE_QUAD_GPUQUAD_PARTITIONMANAGER_CUH
#include "cudaPagani/quad/util/cudaMemoryUtil.h"
#include <algorithm>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace quad {
  template <class K>
  void
  display(K* array,
          size_t size,
          std::string msg = std::string(),
          int specific_index = -1,
          int specific_index2 = -1)
  {
    K* tmp = (K*)malloc(sizeof(K) * size);
    cudaMemcpy(tmp, array, sizeof(K) * size, cudaMemcpyDeviceToHost);

    if (specific_index != -1) {
      // printf("About to display\n");
      // printf("display, %i, %e\n", specific_index,
      // (double)tmp[specific_index]);

      if (specific_index2 != -1)
        // printf("display, %i, %e\n", specific_index2,
        // (double)tmp[specific_index2]);
        return;
    }

    for (size_t i = 0; i < size; ++i) {
      // printf("%s, %i, %e\n", msg.c_str(), i, (double)tmp[i]);
    }

    free(tmp);
  }

  template <typename T, size_t NDIM>
  class Partition {
  public:
    Partition()
    {
      regions = nullptr;
      regionsLength = nullptr;
      parentsIntegral = nullptr;
      parentsError = nullptr;
    }

    void
    Deallocate()
    {
      free(regions);
      free(regionsLength);
      free(parentsIntegral);
      free(parentsError);
    }

    void
    ShallowCopy(T* low_bounds,
                T* length_per_boud,
                T* parentEstimates,
                T* parentErrs,
                size_t size,
                int depthReached)
    {
      // printf("inside shallow copy with %lu regions\n", size);
      numRegions = size;
      regions = low_bounds;
      regionsLength = length_per_boud;
      parentsIntegral = parentEstimates;
      parentsError = parentErrs;
      depth = depthReached;
      // printf("Saving parition with depth %i\n", depth);
    }

    void
    DeepCopyHostToDevice(Partition<T, NDIM> source)
    {
      numRegions = source.numRegions;
      depth = source.depth;
      // printf("Setting depth to %i for partition to be loaded for next
      // iteration\n", depth);
      QuadDebug(cudaMemcpy(regions,
                           source.regions,
                           sizeof(T) * source.numRegions * NDIM,
                           cudaMemcpyHostToDevice));
      CudaCheckError();
      QuadDebug(cudaMemcpy(regionsLength,
                           source.regionsLength,
                           sizeof(T) * source.numRegions * NDIM,
                           cudaMemcpyHostToDevice));
      CudaCheckError();
      QuadDebug(cudaMemcpy(parentsIntegral,
                           source.parentsIntegral,
                           sizeof(T) * source.numRegions,
                           cudaMemcpyHostToDevice));
      CudaCheckError();
      QuadDebug(cudaMemcpy(parentsError,
                           source.parentsError,
                           sizeof(T) * source.numRegions,
                           cudaMemcpyHostToDevice));
      CudaCheckError();
    }

    void
    HostAllocate(size_t size, int sourcePartDepth)
    {
      // printf("Started host allocation\n");
      regions =
        (T*)Host->AllocateMemory(&regions, sizeof(T) * size * NDIM);
      regionsLength = (T*)Host->AllocateMemory(
        &regionsLength, sizeof(T) * size * NDIM);
      parentsIntegral =
        (T*)Host->AllocateMemory(&parentsIntegral, sizeof(T) * size);
      parentsError =
        (T*)Host->AllocateMemory(&parentsError, sizeof(T) * size);
      numRegions = size;
      depth = sourcePartDepth;
      // printf("inside hostAllocate partion has size:%lu\n", numRegions);
    }

    void
    DeviceAllocate(size_t size)
    {
      QuadDebug(
        Device->AllocateMemory((void**)&regions, sizeof(T) * size * NDIM));
      QuadDebug(Device->AllocateMemory((void**)&regionsLength,
                                       sizeof(T) * size * NDIM));
      QuadDebug(Device->AllocateMemory((void**)&parentsIntegral,
                                       sizeof(T) * size));
      QuadDebug(
        Device->AllocateMemory((void**)&parentsError, sizeof(T) * size));
      numRegions = size;
    }

    T* regions;
    T* regionsLength;
    T* parentsIntegral;
    T* parentsError;
    int depth;

    HostMemory<T>* Host;
    DeviceMemory<T>* Device;

    size_t numRegions;
  };

  template <typename T, size_t NDIM>
  class PartitionManager {
  public:
    HostMemory<T>* Host;
    DeviceMemory<T>* Device;

    T queued_reg_estimate;
    T queued_reg_errorest;

    std::vector<Partition<T, NDIM>> partitions;
    std::vector<size_t> partitionSizes;
    std::vector<T> partionContributionsIntegral;
    std::vector<T> partionContributionsError;
    size_t numPartitions;
    const size_t numSplits = 4;

    ~PartitionManager()
    {
      for (int i = 0; i < numPartitions; i++)
        partitions[i].Deallocate();
    }

    size_t
    GetNRegions()
    {
      size_t sum = 0;
      for (int i = 0; i < numPartitions; ++i) {
        sum += partitionSizes[i];
      }
      return sum;
    }

    bool
    Empty()
    {
      return numPartitions == 0;
    }

    size_t
    NumRegionsStored()
    {
      size_t size = 0;
      for (int i = 0; i < numPartitions; i++)
        size += partitionSizes[i];
      return size;
    }

    void
    Init(HostMemory<T>* host, DeviceMemory<T>* device)
    {
      Host = host;
      Device = device;
      numPartitions = 0;
      queued_reg_estimate = 0.;
      queued_reg_errorest = 0.;
    }

    size_t
    partitionRunSum(size_t id)
    {
      // consider only the partitions that are to be added in this iteration
      // that's why we add numPartitions to i
      size_t temp = 0;
      for (int i = 0; i < id; i++)
        temp += partitionSizes[numPartitions + i];
      return temp;
    }

    void
    SetPartitionContributions(T* dParentsIntegral, T* dParentsError)
    {
      // i don't like i+numPartitions indexing
      thrust::device_ptr<T> wrapped_ptr;
      wrapped_ptr = thrust::device_pointer_cast(dParentsIntegral);
      T fourthPartitionsContribution = 0.;

      for (int i = 0; i < numSplits; i++) {
        size_t partionNumParents = partitionSizes[i + numPartitions] / 2;
        size_t parentsProcessed = partitionRunSum(i) / 2;

        T parentsEstimate =
          thrust::reduce(wrapped_ptr + parentsProcessed,
                         wrapped_ptr + parentsProcessed + partionNumParents);
        fourthPartitionsContribution += parentsEstimate;
        partionContributionsIntegral.push_back(parentsEstimate);
      }
 
      wrapped_ptr = thrust::device_pointer_cast(dParentsError);

      for (int i = 0; i < numSplits; i++) {
        size_t partionNumParents = partitionSizes[i + numPartitions] / 2;
        size_t parentsProcessed = partitionRunSum(i) / 2;
        partionContributionsError.push_back(
          thrust::reduce(wrapped_ptr + parentsProcessed,
                         wrapped_ptr + parentsProcessed + partionNumParents));
      }
    }

    void
    GetEvenSplitSize(size_t numToSplit, size_t& quarterSize, size_t& remainder)
    {
      assert(numToSplit % 2 == 0);
      size_t origNum = numToSplit;

      remainder = 0;
      auto isOdd = [](size_t num) -> bool { return num % 2; };

      auto NonZeroRemainder = [](size_t numerator, size_t divisor) -> bool {
        return numerator % divisor != 0;
      };

      while (isOdd(origNum / numSplits) ||
             NonZeroRemainder(origNum, numSplits)) {
        remainder += 2;
        origNum -= 2;
      }
      quarterSize = origNum / numSplits;

      if (3 * quarterSize + quarterSize + remainder != numToSplit) {
        printf("Error creating four non-odd partitions for partition size:%lu",
               numToSplit);
        exit(EXIT_FAILURE);
      }
    };

    void
    SetpartitionSizess(size_t size)
    {
      const size_t numNewPartitions = numSplits;
      assert(!(size % 2));

      size_t extra = size % numNewPartitions; // extra=2
      size_t floor_quarter_size =
        size / numNewPartitions; // quarter_size: 4474069
      GetEvenSplitSize(size, floor_quarter_size, extra);

      for (int i = 0; i < numNewPartitions; i++)
        partitionSizes.push_back(floor_quarter_size);
      partitionSizes[numNewPartitions + numPartitions - 1] += extra;

      for (int i = 0; i < numPartitions + numNewPartitions; i++) {
        assert(partitionSizes[i] % 2 == 0);
        if (partitionSizes[i] % 2 != 0)
          printf("error, uneven partition size\n");
      }
    }

    void
    DeepCopyDeviceToHost(Partition<T, NDIM>& destPartition,
                         Partition<T, NDIM> sourcePartition,
                         size_t sourceRegionsFirstIndex)
    {

      /*
              Give a big partition from the gpu, and copy it to a smaller
         partition on the cpu we can use Partition.ShallowCopy to quickly
         organize the Kernel's gpu structures and simplify the interface
      */
      size_t numPairsInPartition = destPartition.numRegions / 2;

      for (size_t dim = 0; dim < NDIM; ++dim) {
        // place left children in first half and then at second half
        size_t dimStartIndex = dim * sourcePartition.numRegions;
        QuadDebug(cudaMemcpy(
          destPartition.regions + dim * destPartition.numRegions,
          sourcePartition.regions + dimStartIndex + sourceRegionsFirstIndex,
          sizeof(T) * numPairsInPartition,
          cudaMemcpyDeviceToHost));
        CudaCheckError();

        QuadDebug(
          cudaMemcpy(destPartition.regions + dim * destPartition.numRegions +
                       destPartition.numRegions / 2,
                     sourcePartition.regions + dimStartIndex +
                       sourceRegionsFirstIndex + sourcePartition.numRegions / 2,
                     sizeof(T) * numPairsInPartition,
                     cudaMemcpyDeviceToHost));

        CudaCheckError();
        QuadDebug(cudaMemcpy(destPartition.regionsLength +
                               dim * destPartition.numRegions,
                             sourcePartition.regionsLength + dimStartIndex +
                               sourceRegionsFirstIndex,
                             sizeof(T) * numPairsInPartition,
                             cudaMemcpyDeviceToHost));
        CudaCheckError();

        QuadDebug(cudaMemcpy(
          destPartition.regionsLength + dim * destPartition.numRegions +
            destPartition.numRegions / 2,
          sourcePartition.regionsLength + dimStartIndex +
            sourceRegionsFirstIndex + sourcePartition.numRegions / 2,
          sizeof(T) * numPairsInPartition,
          cudaMemcpyDeviceToHost));
        CudaCheckError();
      }

      QuadDebug(
        cudaMemcpy(destPartition.parentsIntegral,
                   sourcePartition.parentsIntegral + sourceRegionsFirstIndex,
                   sizeof(T) * numPairsInPartition,
                   cudaMemcpyDeviceToHost));

      QuadDebug(
        cudaMemcpy(destPartition.parentsIntegral + numPairsInPartition,
                   sourcePartition.parentsIntegral + sourceRegionsFirstIndex,
                   sizeof(T) * numPairsInPartition,
                   cudaMemcpyDeviceToHost));

      QuadDebug(
        cudaMemcpy(destPartition.parentsError,
                   sourcePartition.parentsError + sourceRegionsFirstIndex,
                   sizeof(T) * numPairsInPartition,
                   cudaMemcpyDeviceToHost));

      QuadDebug(
        cudaMemcpy(destPartition.parentsError + numPairsInPartition,
                   sourcePartition.parentsError + sourceRegionsFirstIndex,
                   sizeof(T) * numPairsInPartition,
                   cudaMemcpyDeviceToHost));
      CudaCheckError();
    }

    void
    StoreRegionsInHost(Partition<T, NDIM>& sourcePartition)
    {
      if (sourcePartition.numRegions == 0) {
        return;
      }

      int numNewPartitions =
        4; // always expand by four partitions, we can change this later
      SetpartitionSizess(sourcePartition.numRegions);
      SetPartitionContributions(sourcePartition.parentsIntegral,
                                sourcePartition.parentsError);

      CudaCheckError();

      for (int partitionID = 0; partitionID < numNewPartitions; partitionID++) {
        Partition<T, NDIM> temp;
        partitions.push_back(temp);
        partitions[partitionID + numPartitions].HostAllocate(
          partitionSizes[partitionID + numPartitions], sourcePartition.depth);
      }

      CudaCheckError();

      size_t startRegionIndex = 0;
      for (int partitionID = 0; partitionID < numNewPartitions; partitionID++) {
        DeepCopyDeviceToHost(partitions[partitionID + numPartitions],
                             sourcePartition,
                             startRegionIndex);
        CudaCheckError();
        startRegionIndex +=
          partitions[partitionID + numPartitions].numRegions / 2;
      }

      numPartitions += numNewPartitions;
      // printf("Now have %lu partitions\n", numPartitions);
      CudaCheckError();
    }

    void
    SetDeviceRegionsFromHost(Partition<T, NDIM>& destPartition,
                             Partition<T, NDIM> sourcePartition)
    {
      destPartition.DeviceAllocate(sourcePartition.numRegions);
      CudaCheckError();
      destPartition.DeepCopyHostToDevice(sourcePartition);
      CudaCheckError();
      sourcePartition.Deallocate();
    }

    void
    LoadNextActivePartition(Partition<T, NDIM>& wrapper)
    {
      StoreRegionsInHost(
        wrapper); // 1. save current progress in the form of four partitions
      CudaCheckError();
      // printf("Stored regions in host\n");
      size_t maxErrID =
        0; // 2. Find the partition with the largest error-estimate
      // printf("Partition[%lu]:%e +- %e\n", 0, partionContributionsIntegral[0],
      // partionContributionsError[0] );
      for (size_t i = 1; i < numPartitions; i++) {
        if (partionContributionsError[maxErrID] /*/partitionSizes[maxErrID]*/ <
            partionContributionsError[i] /*/partitionSizes[i]*/) {
          maxErrID = i;
        }
        // printf("Partition[%lu]:%e +- %e\n", i,
        // partionContributionsIntegral[i], partionContributionsError[i] );
      }

      Partition<T, NDIM> priorityP =
        partitions[maxErrID]; // 3. get a pointer to that host partition
      // printf("PRIORITY p depth %i partition index:%lu\n", priorityP.depth,
      // maxErrID);
      // 4. erase it from partition manager
      partitions.erase(partitions.begin() + maxErrID);
      partionContributionsIntegral.erase(partionContributionsIntegral.begin() +
                                         maxErrID);
      partionContributionsError.erase(partionContributionsError.begin() +
                                      maxErrID);
      partitionSizes.erase(partitionSizes.begin() + maxErrID);
      numPartitions--;

      // 5. Free the memory pointed to by current device partition
      // printf("releasing dRegions\n");
      Device->ReleaseMemory(wrapper.regions);
      CudaCheckError();
      Device->ReleaseMemory(wrapper.regionsLength);
      CudaCheckError();
      Device->ReleaseMemory(wrapper.parentsIntegral);
      CudaCheckError();
      Device->ReleaseMemory(wrapper.parentsError);
      CudaCheckError();
      wrapper.numRegions = 0;

      // copy on the device the host partition from step 3.
      SetDeviceRegionsFromHost(wrapper, priorityP);
      CudaCheckError();
      // update statistics on stored regions without including step 3. partition
      queued_reg_estimate = 0;
      queued_reg_errorest = 0;

      // printf("Partitions after poping worst one\n");
      for (int i = 0; i < numPartitions; i++) {
        queued_reg_estimate += partionContributionsIntegral[i];
        queued_reg_errorest += partionContributionsError[i];
        // printf("partition %i %e +- %e, %lu\n", i,
        // partionContributionsIntegral[i], partionContributionsError[i],
        // partitionSizes[i]);
      }

      // printf("Partition.queued_est:%.20f +- %.20f\n", queued_reg_estimate,
      // queued_reg_errorest);
    }
  };

}
#endif