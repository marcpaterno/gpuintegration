#ifndef KOKKOSCUHRE_KERNEL_CUH
#define KOKKOSCUHRE_KERNEL_CUH

#include "Phases.cuh"
#include "Rule.cuh"
#include "quad.h"
#include "util/print.cuh"
#include "util/util.cuh"
#include <KokkosBlas1_dot.hpp>
//#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas1_team_dot.hpp>
#include <utility> //for make_pair to make a subview

struct IntegrandAcrossIters {
  double currentIntegral = 0.;
  double lastIterIntegral = 0.;
  double secondToLastIterIntegral = 0.;

  double currentError = 0.;
  double lastIterError = 0.;
  double secondToLastIterError = 0.;

  void
  UpdateValues(double newIntegral, double newError)
  {
    secondToLastIterIntegral = lastIterIntegral;
    lastIterIntegral = currentIntegral;
    currentIntegral = newIntegral;

    secondToLastIterError = lastIterError;
    lastIterError = currentError;
    currentError = newError;
  }
};

template <typename T, int NDIM>
class Kernel {
public:
  size_t depthBeingProcessed;
  size_t numRegions;
  size_t fEvalPerRegion;
  int positiveSemiDefinite = 0;
  bool estimateHasConverged = false;
  double Jacobian;
  // size_t nFinishedRegions = 0;

  ViewVectorDouble lows;
  ViewVectorDouble highs;

  // Structures<T> constMem;
  // Rule<double> rule;
  int NSETS;
  IntegrandAcrossIters integrandAcrossIters;

  Kernel(int nsets)
  {
    NSETS = nsets;
    lows = ViewVectorDouble("lows", NDIM);
    highs = ViewVectorDouble("highs", NDIM);

    // ViewVectorDouble::HostMirror hlows = Kokkos::create_mirror_view(lows);
    //ViewVectorDouble::HostMirror hhighs = Kokkos::create_mirror_view(highs);
    //Kokkos::deep_copy(hhighs, highs);
    depthBeingProcessed = 0;

    //for (int i = 0; i < NDIM; i++)
    //  hhighs[i] = 1.;
   // Kokkos::deep_copy(highs, hhighs);
    fEvalPerRegion = (1 + 2 * NDIM + 2 * NDIM + 2 * NDIM + 2 * NDIM +
                      2 * NDIM * (NDIM - 1) + 4 * NDIM * (NDIM - 1) +
                      4 * NDIM * (NDIM - 1) * (NDIM - 2) / 3 + (1 << NDIM));
    // int key = 0;
    // int verbose = 0;
    // Kokkos::Profiling::pushRegion("Integration Rule Initialization");
    // rule.Init(NDIM, fEvalPerRegion, key, verbose, &constMem);
    // Kokkos::Profiling::popRegion();
  }

  bool
  CheckTerminationCondition(double leaves_estimate,
                            double leaves_errorest,
                            double& integral,
                            double& error,
                            size_t& nregions,
                            int& fail,
                            int mustFinish,
                            double epsrel,
                            double epsabs,
                            int iteration)
  {
    if ((iteration != 0 &&
         leaves_errorest <= MaxErr(leaves_estimate, epsrel, epsabs)) ||
        mustFinish || numRegions == 0) {
      integral = leaves_estimate;
      error = leaves_errorest;
      nregions += numRegions;

      if (leaves_errorest <= MaxErr(leaves_estimate, epsrel, epsabs)) {
        fail = 0;
      } else {
        fail = 1;
      }
      numRegions = 0;
      return true;
    } else {
      return false;
    }
  }

  void
  AlignRegions(ViewVectorDouble dRegions,
               ViewVectorDouble dRegionsLength,
               ViewVectorInt activeRegions,
               ViewVectorDouble dRegionsIntegral,
               ViewVectorDouble dRegionsError,
               ViewVectorDouble dRegionsParentIntegral,
               ViewVectorDouble dRegionsParentError,
               ViewVectorInt subDividingDimension,
               ViewVectorInt scannedArray,
               ViewVectorDouble newActiveRegions,
               ViewVectorDouble newActiveRegionsLength,
               ViewVectorInt newActiveRegionsBisectDim,
               size_t numRegions,
               size_t newNumRegions,
               size_t numOfDivisionOnDimension)
  {

    size_t numThreads = 64;
    size_t numBlocks =
      numRegions / numThreads + ((numRegions % numThreads) ? 1 : 0);

    Kokkos::TeamPolicy<Kokkos::LaunchBounds<64, 18>> team_policy1(numBlocks,
                                                                  numThreads);
    auto team_policy = Kokkos::Experimental::require(
      team_policy1, Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for(
      "AlignRegions",
      team_policy,
      KOKKOS_LAMBDA(const member_type team_member) {
        int threadIdx = team_member.team_rank();
        int blockIdx = team_member.league_rank();

        size_t tid = blockIdx * numThreads + threadIdx;

        if (tid < numRegions && activeRegions(tid) == 1.) {
          size_t interval_index = (size_t)scannedArray(tid);

          for (size_t i = 0; i < NDIM; ++i) {
            newActiveRegions(i * newNumRegions + interval_index) =
              dRegions(i * numRegions + tid);
            newActiveRegionsLength(i * newNumRegions + interval_index) =
              dRegionsLength(i * numRegions + tid);
          }

          dRegionsParentIntegral(interval_index) = dRegionsIntegral(tid);
          dRegionsParentError(interval_index) = dRegionsError(tid);

          for (size_t i = 0; i < numOfDivisionOnDimension; ++i) {
            newActiveRegionsBisectDim(i * newNumRegions + interval_index) =
              subDividingDimension(tid);
          }
        }
      });
  }

  void
  DivideIntervalsGPU(ViewVectorDouble genRegions,
                     ViewVectorDouble genRegionsLength,
                     ViewVectorDouble activeRegions,
                     ViewVectorDouble activeRegionsLength,
                     ViewVectorInt activeRegionsBisectDim,
                     size_t numActiveRegions,
                     int numOfDivisionOnDimension)
  {

    size_t numThreads = 64;
    size_t numBlocks =
      numActiveRegions / numThreads + ((numActiveRegions % numThreads) ? 1 : 0);
    Kokkos::TeamPolicy<Kokkos::LaunchBounds<64, 18>> team_policy1(numBlocks,
                                                                  numThreads);
    auto team_policy = Kokkos::Experimental::require(
      team_policy1, Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    
    Kokkos::parallel_for(
      "DivideIntervalsGPU",
      team_policy,
      KOKKOS_LAMBDA(const member_type team_member) {
        int threadIdx = team_member.team_rank();
        int blockIdx = team_member.league_rank();
        size_t tid = blockIdx * numThreads + threadIdx;

        if (tid < numActiveRegions) {

          int bisectdim = activeRegionsBisectDim(tid);
          size_t data_size = numActiveRegions * numOfDivisionOnDimension;

          for (int i = 0; i < numOfDivisionOnDimension; ++i) {
            for (int dim = 0; dim < NDIM; ++dim) {
              genRegions(i * numActiveRegions + dim * data_size + tid) =
                activeRegions(dim * numActiveRegions + tid);
              genRegionsLength(i * numActiveRegions + dim * data_size + tid) =
                activeRegionsLength(dim * numActiveRegions + tid);
            }
          }

          for (int i = 0; i < numOfDivisionOnDimension; ++i) {

            double interval_length =
              activeRegionsLength(bisectdim * numActiveRegions + tid) /
              numOfDivisionOnDimension;

            genRegions(bisectdim * data_size + i * numActiveRegions + tid) =
              activeRegions(bisectdim * numActiveRegions + tid) +
              i * interval_length;
            genRegionsLength(i * numActiveRegions + bisectdim * data_size +
                             tid) = interval_length;
          }
        }
      });
  }

  void
  ReturnLastIndexValues(ViewVectorInt listA,
                        ViewVectorInt listB,
                        double& lastA,
                        double& lastB)
  {
    int sizeA = listA.extent(0);
    int sizeB = listB.extent(0);
    ViewVectorInt A_sub(listA, std::make_pair(sizeA - 1, sizeA));
    ViewVectorInt B_sub(listB, std::make_pair(sizeB - 1, sizeB));

    ViewVectorInt::HostMirror hostA_sub = Kokkos::create_mirror_view(A_sub);
    ViewVectorInt::HostMirror hostB_sub = Kokkos::create_mirror_view(B_sub);

    deep_copy(hostA_sub, A_sub);
    deep_copy(hostB_sub, B_sub);
    lastA = hostA_sub(0);
    lastB = hostB_sub(0);
  }

  size_t
  GenerateActiveIntervals(ViewVectorDouble& dRegions,
                          ViewVectorDouble& dRegionsLength,
                          ViewVectorInt activeRegions,
                          ViewVectorInt subDividingDimension,
                          ViewVectorDouble dRegionsIntegral,
                          ViewVectorDouble dRegionsError,
                          ViewVectorDouble& dParentsIntegral,
                          ViewVectorDouble& dParentsError,
                          constViewVectorDouble generators)
  {
    ViewVectorInt scannedArray("scannedArray", numRegions);
    //deep_copy(scannedArray, activeRegions);
    Kokkos::Profiling::pushRegion("Exclusive scan");
    exclusive_prefix_scan(activeRegions, scannedArray);
    Kokkos::Profiling::popRegion();
    double lastElement = 0.;
    double lastScanned = 0.;

    ReturnLastIndexValues(
      activeRegions, scannedArray, lastElement, lastScanned);
    size_t numActiveRegions = (size_t)lastScanned;
    size_t numInActiveRegions = 0;

    if (lastElement == 1.)
      numActiveRegions++;

    if (numActiveRegions > 0) {
      Kokkos::Profiling::pushRegion("Aligning regions");
      size_t numOfDivisionOnDimension = 2;
      ViewVectorDouble newActiveRegions("newActiveRegions",
                                        numActiveRegions * NDIM);
      ViewVectorDouble newActiveRegionsLength("newActiveRegionsLength",
                                              numActiveRegions * NDIM);
      ViewVectorInt newActiveRegionsBisectDim("newActiveRegionsBisectDim",
                                              numActiveRegions *
                                                numOfDivisionOnDimension);

      ExpandcuArray(dParentsIntegral, numRegions / 2, numActiveRegions);
      ExpandcuArray(dParentsError, numRegions / 2, numActiveRegions);
      
      AlignRegions(dRegions,
                   dRegionsLength,
                   activeRegions,
                   dRegionsIntegral,
                   dRegionsError,
                   dParentsIntegral,
                   dParentsError,
                   subDividingDimension,
                   scannedArray,
                   newActiveRegions,
                   newActiveRegionsLength,
                   newActiveRegionsBisectDim,
                   numRegions,
                   numActiveRegions,
                   numOfDivisionOnDimension);
      Kokkos::Profiling::popRegion();
      
      Kokkos::Profiling::pushRegion("Dividing intervals");
      ViewVectorDouble genRegions(
        "genRegions", numActiveRegions * NDIM * numOfDivisionOnDimension);
      ViewVectorDouble genRegionsLength(
        "genRegionsLength", numActiveRegions * NDIM * numOfDivisionOnDimension);
      
      DivideIntervalsGPU(genRegions,
                         genRegionsLength,
                         newActiveRegions,
                         newActiveRegionsLength,
                         newActiveRegionsBisectDim,
                         numActiveRegions,
                         numOfDivisionOnDimension);
      
      dRegions = genRegions;
      dRegionsLength = genRegionsLength;
      numInActiveRegions = numRegions - numActiveRegions;
      Kokkos::Profiling::popRegion();
      numRegions = numActiveRegions * numOfDivisionOnDimension;

    } else {
      numInActiveRegions = numRegions;
      numRegions = 0;
    }

    return numInActiveRegions;
  }

  void
  FixErrorBudgetOverflow(ViewVectorInt activeRegions,
                         double& integral,
                         double& error,
                         double& iter_finished_estimate,
                         double& iter_finished_errorest,
                         double newLeavesEstimate,
                         double epsrel,
                         int& fail)
  {

    if (error > abs(newLeavesEstimate) * epsrel) {
      // printf("REVERTING ACTIVE REGIONS\n");
      Kokkos::parallel_for(
        "RefineError", numRegions, KOKKOS_LAMBDA(const int64_t index) {
          activeRegions(index) = 1.;
        });

      error -= iter_finished_errorest;
      integral -= iter_finished_estimate;
      iter_finished_estimate = 0.;
      iter_finished_errorest = 0.;
      cudaDeviceSynchronize();
    }
  }

  size_t
  ComputeNumUnPolishedRegions(ViewVectorInt unpolishedRegions)
  {

    int _numUnPolishedRegions = 0;
    Kokkos::parallel_reduce(
      "ComputeNumUnPolishedRegions",
      numRegions,
      KOKKOS_LAMBDA(const int64_t index, int& valueToUpdate) {
        valueToUpdate += unpolishedRegions(index);
      },
      _numUnPolishedRegions);
    return (size_t)_numUnPolishedRegions;
  }

  bool
  CheckZeroNumRegionsTermination(double& integral,
                                 double& error,
                                 double leaves_estimate,
                                 double leaves_errorest)
  {
    if (numRegions == 0) {
      integral = leaves_estimate;
      error = leaves_errorest;
      return true;
    }
    return false;
  }

  void
  Filter(ViewVectorDouble dRegionsError,
         ViewVectorInt unpolishedRegions,
         ViewVectorInt activeRegions,
         size_t numRegions,
         double errThreshold)
  {

    Kokkos::parallel_for(
      "Filter", numRegions, KOKKOS_LAMBDA(const int64_t index) {
        double selfErr = dRegionsError(index);
        // only "real active" regions can be polished (rename activeRegions in
        // this context to polishedRegions)
        unpolishedRegions(index) =
          (selfErr > errThreshold) * activeRegions(index);
      });
  }

  void
  GetNextThreshold(double min, double max, int rightDirection, double& current)
  {
    if (rightDirection) {
      double diff = abs(max - current);
      current += diff * .5;
    } else {
      double diff = abs(min - current);
      current -= diff * .5;
    }
  }

  bool
  AdjustErrorThreshold(double ErrorestMarkedFinished,
                       double MaxErrorestToAllowedToFinish,
                       double percentRegionsToKeep,
                       int& currentDirection,
                       double& lastThreshold,
                       double& threshold,
                       double& minThreshold,
                       double& maxThreshold,
                       int& numDirectionChanges)
  {
    // printf("adjusting threshold with min max:%15e, %.15e\n", minThreshold,
    // maxThreshold);
    int priorDirection = currentDirection;
    if (ErrorestMarkedFinished > MaxErrorestToAllowedToFinish ||
        percentRegionsToKeep < .5) {
      if (currentDirection == 1) {
        numDirectionChanges++;
        if (lastThreshold < threshold) {
          minThreshold = lastThreshold;
          maxThreshold = threshold;
        }
      } else
        lastThreshold = threshold;

      currentDirection = 0;
      GetNextThreshold(minThreshold, maxThreshold, currentDirection, threshold);
    } else if (percentRegionsToKeep > .5) {
      if (currentDirection == 0) {
        numDirectionChanges++;
        if (threshold < lastThreshold) {
          maxThreshold = lastThreshold;
          minThreshold = threshold;
        }

      } else
        lastThreshold = threshold;

      currentDirection = 1;
      GetNextThreshold(minThreshold,
                       maxThreshold,
                       currentDirection,
                       threshold); // the flag was priorly set to zero
    }
    // printf("new from within adjustment min max:%15e, %.15e\n", minThreshold,
    // maxThreshold);
    return currentDirection != priorDirection &&
           priorDirection != -1; // return whether there is a direction change
  }

  void
  HSClassify(ViewVectorDouble dRegionsIntegral,
             ViewVectorDouble dRegionsError,
             ViewVectorInt& activeRegions,
             double& integral,
             double& error,
             size_t& nregions,
             double iterEstimate,
             double iterErrorest,
             double iter_finished_estimate,
             double iter_finished_errorest,
             double leaves_estimate,
             double leaves_errorest,
             double epsrel,
             double epsabs,
             bool& mustFinish,
             int iteration)
  {

    int requiredDigits =
      ceil(log10(1 / epsrel)); // this is no longer used in RelErrClassify, move
                               // below when used by Filter

    estimateHasConverged =
      estimateHasConverged == false ?
        (iteration >= 2 ?
           sigDigitsSame(integrandAcrossIters.lastIterIntegral,
                         integrandAcrossIters.secondToLastIterIntegral,
                         leaves_estimate,
                         requiredDigits) :
           false) :
        true;
    integrandAcrossIters.secondToLastIterIntegral =
      integrandAcrossIters.lastIterIntegral;
    integrandAcrossIters.lastIterIntegral = leaves_estimate;

    if (GetGPUMemNeededForNextIteration(numRegions, NDIM) <
          GetAmountFreeMem() &&
        !estimateHasConverged) {
      return;
    }
    
    double targetError = abs(leaves_estimate) * epsrel;

    double MaxPercentOfErrorBudget = .25;
    double acceptableThreshold = 0.;
    ViewVectorInt unpolishedRegions("numRegions", numRegions); 

    size_t targetRegionNum = numRegions / 2;
    double ErrThreshold = iterErrorest / (numRegions); 

    double lastThreshold = ErrThreshold;
    size_t numActiveRegions = numRegions;
    double iter_polished_errorest = 0.;
    double iter_polished_estimate = 0.;
    int numDirectionChanges = 0;
    int direction = -1;

    double min = 0., max = 0.;
    min = ComputeMin(dRegionsError);
    max = ComputeMax(dRegionsError);

    bool directionChange = false;
    int maxDirectionChanges = 9;

    do {
      iter_polished_estimate = 0.;
      iter_polished_errorest = 0.;

      if (numDirectionChanges >= maxDirectionChanges) {
        ErrThreshold = acceptableThreshold; 
      } else if (numDirectionChanges > 2 && numDirectionChanges <= 9 &&
                 directionChange) {
        MaxPercentOfErrorBudget =  numDirectionChanges > 1 ?
            MaxPercentOfErrorBudget + .1 :
            MaxPercentOfErrorBudget;
      }

      numActiveRegions = 0;
      Filter(dRegionsError,
             unpolishedRegions,
             activeRegions,
             numRegions,
             ErrThreshold);
      numActiveRegions = ComputeNumUnPolishedRegions(unpolishedRegions);

      if (numActiveRegions <= targetRegionNum) {
          
        double polishedEstimate = 0.;
        Kokkos::parallel_reduce("polished errorest", numRegions,
            KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
                valueToUpdate += dRegionsIntegral(index)*unpolishedRegions(index);
            },
            polishedEstimate);
          
        iter_polished_estimate =
          iterEstimate - iter_finished_estimate - polishedEstimate;
        //iter_polished_estimate =
        //  iterEstimate - iter_finished_estimate -
        //  KokkosBlas::dot(unpolishedRegions, dRegionsIntegral);
          
        double polishedErrorest = 0.;
        Kokkos::parallel_reduce("polished errorest", numRegions,
            KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
                valueToUpdate += dRegionsError(index)*unpolishedRegions(index);
            },
            polishedErrorest);
        
        iter_polished_errorest =
          iterErrorest - iter_finished_errorest - polishedErrorest;
        //iter_polished_errorest =
        //  iterErrorest - iter_finished_errorest -
        //  KokkosBlas::dot(unpolishedRegions, dRegionsError);
          
        if ((iter_polished_errorest <=
               MaxPercentOfErrorBudget * (targetError - error) ||
             numDirectionChanges == maxDirectionChanges)) {
          integral += iter_polished_estimate;
          error += iter_polished_errorest;
          break;
        } else if (iter_polished_errorest <= .95 * (targetError - error)) {
          acceptableThreshold = ErrThreshold;
        }
      }

      double unpolishedPercentage =
        (double)(numActiveRegions) / (double)numRegions;

      directionChange =
        AdjustErrorThreshold(iter_polished_errorest,
                             MaxPercentOfErrorBudget * (targetError - error),
                             unpolishedPercentage,
                             direction,
                             lastThreshold,
                             ErrThreshold,
                             min,
                             max,
                             numDirectionChanges);

      if (numDirectionChanges == maxDirectionChanges &&
          acceptableThreshold == 0.) {
        if (!estimateHasConverged || GetGPUMemNeededForNextIteration(
                                       numRegions, NDIM) >= GetAmountFreeMem())
          mustFinish = true;
        return;
      }

    } while (numActiveRegions > targetRegionNum ||
             iter_polished_errorest >
               MaxPercentOfErrorBudget * (targetError - error) ||
             error > targetError);

    if (numActiveRegions == numRegions) {
      mustFinish = true;
    } else {
      activeRegions = unpolishedRegions;
    }
  }

  void
  RelErrClassify(int heuristicID,
                 ViewVectorInt activeRegions,
                 ViewVectorDouble dRegionsIntegral,
                 ViewVectorDouble& dRegionsError,
                 ViewVectorDouble dParentsIntegral,
                 ViewVectorDouble dParentsError,
                 double epsrel,
                 int iteration)
  {
    if (iteration == 0)
      return;
    int currIterRegions = numRegions;
    ViewVectorDouble newErrs("newErrs", numRegions);
    // int heuristicID = positiveSemiDefinite;
    Kokkos::parallel_for(
      "RefineError", numRegions, KOKKOS_LAMBDA(const int64_t index) {
        double selfErr = dRegionsError(index);
        double selfRes = dRegionsIntegral(index);

        size_t inRightSide = (2 * index >= currIterRegions);
        size_t inLeftSide = (0 >= inRightSide);
        size_t siblingIndex = index + (inLeftSide * currIterRegions / 2) -
                              (inRightSide * currIterRegions / 2);
        size_t parIndex = index - inRightSide * (currIterRegions * .5);

        double siblErr = dRegionsError(siblingIndex);
        double siblRes = dRegionsIntegral(siblingIndex);
        double parRes = dParentsIntegral(parIndex);
        double diff = siblRes + selfRes - parRes;
        diff = fabs(.25 * diff);

        double err = selfErr + siblErr;

        if (err > 0.0) {
          double c = 1 + 2 * diff / err;
          selfErr *= c;
        }

        selfErr += diff;
        newErrs(index) = selfErr;

        int PassRatioTest =
          heuristicID != 1 && selfErr < MaxErr(selfRes, epsrel, 1e-200);
        activeRegions(index) = (double)(!PassRatioTest);
      });
    dRegionsError = newErrs;
  }

  template <typename IntegT>
  int
  FirstPhaseIteration(IntegT d_integrand,
                      int heuristicID,
                      ViewVectorDouble& dRegions,
                      ViewVectorDouble& dRegionsLength,
                      ViewVectorDouble& dRegionsIntegral,
                      ViewVectorDouble& dRegionsError,
                      ViewVectorDouble& dParentsIntegral,
                      ViewVectorDouble& dParentsError,
                      constViewVectorDouble generators,
                      const Structures<double>& constMem,
                      double& integral,
                      double& error,
                      size_t& nregions,
                      size_t& nFinishedRegions,
                      double epsrel,
                      double epsabs,
                      int& fail,
                      bool& mustFinish,
                      int it)
  {
    // set KOKKOS_ENABLE_CUDA_LDG_INTRINSIC to use __ldg__
    // because we need to take dot-product, make activeRegions a double view
    //printf("Start of iteration\n");
    Kokkos::Profiling::pushRegion("Iteration Allocations");
    ViewVectorInt activeRegions("activeRegions", numRegions);
    ViewVectorInt subDividingDimension("subDividingDimension", numRegions);

    dRegionsIntegral = ViewVectorDouble("dRegionsIntegral", numRegions);
    dRegionsError = ViewVectorDouble("dRegionsError", numRegions);

    if (it == 0) {
      dParentsIntegral = ViewVectorDouble("dParentsIntegral", numRegions);
      dParentsError = ViewVectorDouble("dParentsError", numRegions);
    }
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("INTEGRATE_GPU_PHASE1");
    // printf("Launching kernel numRegions:%lu\n", numRegions);
    INTEGRATE_GPU_PHASE1<IntegT, NDIM>(d_integrand,
                                       dRegions.data(),
                                       dRegionsLength.data(),
                                       numRegions,
                                       dRegionsIntegral.data(),
                                       dRegionsError.data(),
                                       activeRegions.data(),
                                       subDividingDimension.data(),
                                       //epsrel,
                                       //epsabs,
                                       /*constMem&*/ constMem._gpuG.data(),
                                       constMem._GPUScale.data(),
                                       constMem._GPUNorm.data(),
                                       constMem._gpuGenPermGIndex.data(),
                                       constMem._cRuleWt.data(),
                                       fEvalPerRegion,
                                       NSETS,
                                       lows.data(),
                                       highs.data(),
                                      // it,
                                       ldexp(1., -depthBeingProcessed), //depthBeingProcessed,
                                       Jacobian,
                                       generators.data(),
                                       it);
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("RelErrClassify");
    RelErrClassify(heuristicID,
                   activeRegions,
                   dRegionsIntegral,
                   dRegionsError,
                   dParentsIntegral,
                   dParentsError,
                   epsrel,
                   it);
    Kokkos::Profiling::popRegion();

    // Compute integral and error estimates through reductions
    
    //printf("Reduction 1 %lu regions\n", numRegions);
    Kokkos::Profiling::pushRegion("Reduction 1");
    double iter_estimate = 0.;
    Kokkos::parallel_reduce(
      "Estimate computation",
      numRegions,
      KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
        valueToUpdate += dRegionsIntegral(index);
      },
      iter_estimate);

    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("Reduction 2");
    double iter_errorest = 0.;
    Kokkos::parallel_reduce(
      "Estimate computation",
      numRegions,
      KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
        valueToUpdate += dRegionsError(index);
      },
      iter_errorest);
    Kokkos::Profiling::popRegion();
    // printf("iter_estimate:%.15f +- %.15f numRegions:%lu iteration:%i\n",
    // iter_estimate, iter_errorest, numRegions, it);

    double leaves_estimate = integral + iter_estimate;
    double leaves_errorest = error + iter_errorest;
    
    Kokkos::Profiling::pushRegion("Inner Product 1");
    double activeEst = 0.;
    Kokkos::parallel_reduce(
      "ProParRed1",
      numRegions,
      KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
        valueToUpdate += dRegionsIntegral(index)*activeRegions(index);
      },
      activeEst);
    double iter_finished_estimate =
      iter_estimate - activeEst;
    //double iter_finished_estimate =
    //  iter_estimate - KokkosBlas::dot(activeRegions, dRegionsIntegral);
      
    Kokkos::Profiling::popRegion();
    
    Kokkos::Profiling::pushRegion("Inner Product 2");
    double activeErrorest = 0.;
    Kokkos::parallel_reduce(
      "ProParRed2",
      numRegions,
      KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
        valueToUpdate += dRegionsError(index)*activeRegions(index);
      },
      activeErrorest);
    double iter_finished_errorest =
      iter_errorest - activeErrorest;
      
    //double iter_finished_errorest =
    //  iter_errorest - KokkosBlas::dot(activeRegions, dRegionsError);
      
    Kokkos::Profiling::popRegion();
    integral += iter_finished_estimate;
    error += iter_finished_errorest;

    /*printf("leaves_estimate:%.15f +- %.15f numRegions:%zu iteration:%i\n",
     leaves_estimate, leaves_errorest, numRegions, it);*/
    Kokkos::Profiling::pushRegion("FixErrorBudgetOverflow");
    FixErrorBudgetOverflow(activeRegions,
                           integral,
                           error,
                           iter_finished_estimate,
                           iter_finished_errorest,
                           leaves_estimate,
                           epsrel,
                           fail);
    Kokkos::Profiling::popRegion();

    if (CheckTerminationCondition(leaves_estimate,
                                  leaves_errorest,
                                  integral,
                                  error,
                                  nregions,
                                  fail,
                                  mustFinish,
                                  epsrel,
                                  epsabs,
                                  it)) {
      return true;
    }

    Kokkos::Profiling::pushRegion("HSClassify");
    HSClassify(dRegionsIntegral,
               dRegionsError,
               activeRegions,
               integral,
               error,
               nregions,
               iter_estimate,
               iter_errorest,
               iter_finished_estimate,
               iter_finished_errorest,
               leaves_estimate,
               leaves_errorest,
               epsrel,
               epsabs,
               mustFinish,
               it);
    Kokkos::Profiling::popRegion();

    if (CheckTerminationCondition(leaves_estimate,
                                  leaves_errorest,
                                  integral,
                                  error,
                                  nregions,
                                  fail,
                                  mustFinish,
                                  epsrel,
                                  epsabs,
                                  it)) {
      return true;
    }

    if (it < 700 && fail == 1) {
      Kokkos::Profiling::pushRegion("GenerateActiveIntervals");
      //printf("Generate active intervals\n");
      size_t numInActiveIntervals =
        GenerateActiveIntervals(dRegions,
                                dRegionsLength,
                                activeRegions,
                                subDividingDimension,
                                dRegionsIntegral,
                                dRegionsError,
                                dParentsIntegral,
                                dParentsError,
                                generators);
      depthBeingProcessed++;
      nregions += numInActiveIntervals;
      nFinishedRegions += numInActiveIntervals;
      Kokkos::Profiling::popRegion();
      if (CheckZeroNumRegionsTermination(
            integral, error, leaves_estimate, leaves_errorest)) {
        // printf("Have zero regions so will trigger termination\n");
        return true;
      }
    } else {
      nregions += numRegions;
    }
    return false;
  }

  void AllocVolArrays(Volume<double, NDIM>* vol){
    Kokkos::Profiling::pushRegion("AllocVolArrays");
    lows = ViewVectorDouble("lows", NDIM);
    highs = ViewVectorDouble("highs", NDIM);  
    
    ViewVectorDouble::HostMirror h_highs = Kokkos::create_mirror_view(highs);
    ViewVectorDouble::HostMirror h_lows = Kokkos::create_mirror_view(lows);
    Jacobian = 1.;
    
    if(vol){
        for (int i = 0; i < NDIM; i++){
            h_highs(i) = vol->highs[i];
            h_lows(i) = vol->lows[i];
            Jacobian *= vol->highs[i]- vol->lows[i];
        }
    }
    else{
        Volume<T, NDIM> tempVol;
        for(int i=0; i<NDIM; i++){
            h_highs(i) = tempVol.highs[i];
            h_lows(i) = tempVol.lows[i];
            Jacobian *= tempVol.highs[i] - tempVol.lows[i];
        }
    }
    
    Kokkos::deep_copy(highs, h_highs);
    Kokkos::deep_copy(lows, h_lows);   
    Kokkos::Profiling::popRegion();
  }

  template <typename IntegT>
  bool
  IntegrateFirstPhase(IntegT _integrand,
                      double epsrel,
                      double epsabs,
                      int heuristicID,
                      double& integral,
                      double& error,
                      size_t& nregions,
                      size_t& nFinishedRegions,
                      int& fail,
                      size_t maxIters,
                      ViewVectorDouble& dRegions,
                      ViewVectorDouble& dRegionsLength,
                      ViewVectorDouble& dRegionsIntegral,
                      ViewVectorDouble& dRegionsError,
                      ViewVectorDouble& dParentsIntegral,
                      ViewVectorDouble& dParentsError,
                      const Structures<double>& constMem,
                      Volume<T, NDIM>* vol = nullptr)
  {
    Kokkos::View<IntegT*, Kokkos::CudaSpace> d_integrand("d_integrand", 1);
    ViewVectorDouble _generators("generators", NDIM * fEvalPerRegion);
    //printf("About to compute generators\n");
    ComputeGenerators<NDIM>(_generators, fEvalPerRegion, constMem);
    AllocVolArrays(vol);
    constViewVectorDouble generators = _generators;

    bool mustFinish = false;
    bool terminate = false;

    for (int it = 0; it < maxIters && terminate == false; it++) {
      terminate = FirstPhaseIteration(d_integrand,
                                      heuristicID,
                                      dRegions,
                                      dRegionsLength,
                                      dRegionsIntegral,
                                      dRegionsError,
                                      dParentsIntegral,
                                      dParentsError,
                                      generators,
                                      constMem,
                                      integral,
                                      error,
                                      nregions,
                                      nFinishedRegions,
                                      epsrel,
                                      epsabs,
                                      fail,
                                      mustFinish,
                                      it);
      // printf("Done with iter maxIters:%i terminate:%i \n", maxIters,
      // terminate);
    }

    // printf("At IntegratePhaseI side estimate:%.15f\n", integral);
    return 0;
  }

  void
  GenerateInitialRegions(ViewVectorDouble& dRegions,
                         ViewVectorDouble& dRegionsLength)
  {
    ViewVectorDouble::HostMirror Regions = Kokkos::create_mirror_view(dRegions);
    ViewVectorDouble::HostMirror RegionsLength =
      Kokkos::create_mirror_view(dRegionsLength);

    for (size_t i = 0; i < NDIM; i++) {
      Regions(i) = 0;
      RegionsLength(i) = 1;
    }

    Kokkos::deep_copy(dRegions, Regions);
    Kokkos::deep_copy(dRegionsLength, RegionsLength);

    size_t numOfDivisionPerRegionPerDimension = 4;
    if (NDIM == 5)
      numOfDivisionPerRegionPerDimension = 2;
    if (NDIM == 6)
      numOfDivisionPerRegionPerDimension = 2;
    if (NDIM == 7)
      numOfDivisionPerRegionPerDimension = 2;
    if (NDIM > 7)
      numOfDivisionPerRegionPerDimension = 2;
    if (NDIM > 10)
      numOfDivisionPerRegionPerDimension = 1;

    depthBeingProcessed = log2(numOfDivisionPerRegionPerDimension) * NDIM;
    numRegions = (size_t)pow((T)numOfDivisionPerRegionPerDimension, (T)NDIM);

    ViewVectorDouble newRegions("newRegions", numRegions * NDIM);
    ViewVectorDouble newRegionsLength("newRegionsLength", numRegions * NDIM);

    size_t numThreads = 512;
    size_t numBlocks = (size_t)ceil(
      pow((T)numOfDivisionPerRegionPerDimension, (T)NDIM) / numThreads);
    size_t shMemBytes = NDIM * sizeof(double);
    size_t currentNumRegions = 1;

    // need c++17 for KOKKOS_LAMBDA that is defined as [=,*this] __host__
    // __device__, interestingly adding __host__ below causes problems

    Kokkos::parallel_for(
      "GenerateInitialRegions",
      team_policy(numBlocks, numThreads)
        .set_scratch_size(0, Kokkos::PerTeam(shMemBytes)),
      /*KOKKOS_LAMBDA*/ [=, *this] __device__(const member_type team_member) {
        int threadIdx = team_member.team_rank();
        int blockIdx = team_member.league_rank();

        ScratchViewDouble slength(team_member.team_scratch(0), NDIM);
        size_t threadId = blockIdx * numThreads + threadIdx;

        if (threadIdx < NDIM) {
          slength[threadIdx] =
            dRegionsLength[threadIdx] / numOfDivisionPerRegionPerDimension;
        }

        team_member.team_barrier();

        if (threadId < numRegions) {
          size_t interval_index =
            threadId / pow((T)numOfDivisionPerRegionPerDimension, (double)NDIM);
          size_t local_id =
            threadId %
            (size_t)pow((T)numOfDivisionPerRegionPerDimension, (double)NDIM);

          for (int dim = 0; dim < NDIM; ++dim) {
            size_t id =
              (size_t)(local_id /
                       pow((T)numOfDivisionPerRegionPerDimension, (T)dim)) %
              numOfDivisionPerRegionPerDimension;
            newRegions[numRegions * dim + threadId] =
              dRegions[currentNumRegions * dim + interval_index] +
              id * slength[dim];
            newRegionsLength[numRegions * dim + threadId] = slength[dim];
          }
        }
      });

    dRegions = newRegions;
    dRegionsLength = newRegionsLength;
  }
};

#endif