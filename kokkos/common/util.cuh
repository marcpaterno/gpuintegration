#ifndef UTIL_CUH
#define UTIL_CUH

#include <stdio.h>
#include <string.h>

std::string
doubleToString(double val, int prec_level)
{
  std::ostringstream out;
  out.precision(prec_level);
  out << std::fixed << val;
  return out.str();
}

size_t
GetAmountFreeMem()
{
#ifdef KOKKOS_COMPILER_NVCC
  size_t free_physmem = 0.;
  size_t total_physmem = 0.;
  cudaMemGetInfo(&free_physmem, &total_physmem);
  return free_physmem;
#endif
  return 0;
}

size_t
GetTotalMem()
{
#ifdef KOKKOS_COMPILER_NVCC
  size_t free_physmem = 0.;
  size_t total_physmem = 0.;
  cudaMemGetInfo(&free_physmem, &total_physmem);
  return total_physmem;
#endif
  return 0;
}

size_t
GetGPUMemNeededForNextIteration_CallBeforeSplit(size_t numRegions, int NDIM)
{
  // doubles needed to classify and divide now
  //----------------------------------------------------------
  size_t scanned = numRegions;
  size_t newActiveRegions = numRegions * NDIM;
  size_t newActiveRegionsLength = numRegions * NDIM;
  size_t parentExpansionEstimate = numRegions;
  size_t parentExpansionErrorest = numRegions;
  size_t genRegions = numRegions * NDIM * 2;
  size_t genRegionsLength = numRegions * NDIM * 2;

  // ints needed to classify and divide now
  size_t activeBisectDim = numRegions;
  //----------------------------------------------------------
  // doubles needed for sampling next iteration
  size_t regions = 2 * numRegions * NDIM;
  size_t regionsLength = 2 * numRegions * NDIM;
  size_t regionsIntegral = 2 * numRegions;
  size_t regionsError = 2 * numRegions;
  size_t parentsIntegral = numRegions;
  size_t parentsError = numRegions;
  // ints needed for sampling next iteration
  size_t subDivDim = 2 * numRegions;

  //----------------------------------------------------------

  // we also need to worry about the temporary arrays that are created to do the
  // copies

  size_t Ints_Size = 4 * (activeBisectDim + subDivDim + scanned);
  size_t Doubles_Size =
    8 * (newActiveRegions + newActiveRegionsLength + parentExpansionEstimate +
         parentExpansionErrorest + genRegions + genRegionsLength + regions +
         regionsLength + regionsIntegral + regionsError + parentsIntegral +
         parentsError);
  // the above doesn't consider the intermediate memory needed
  return Ints_Size + Doubles_Size;
}

size_t
GetGPUMemNeededForNextIteration(size_t numRegions, int NDIM)
{
  // doubles needed to classify and divide now
  //----------------------------------------------------------
  size_t scanned = numRegions;
  size_t newActiveRegions = numRegions * NDIM;
  size_t newActiveRegionsLength = numRegions * NDIM;
  size_t parentExpansionEstimate = numRegions;
  size_t parentExpansionErrorest = numRegions;
  size_t genRegions = numRegions * NDIM * 2;
  size_t genRegionsLength = numRegions * NDIM * 2;

  // ints needed to classify and divide now
  size_t activeBisectDim = numRegions;
  //----------------------------------------------------------
  // doubles needed for sampling next iteration
  size_t regions = 2 * numRegions * NDIM;
  size_t regionsLength = 2 * numRegions * NDIM;
  size_t regionsIntegral = 2 * numRegions;
  size_t regionsError = 2 * numRegions;
  size_t parentsIntegral = numRegions;
  size_t parentsError = numRegions;
  // ints needed for sampling next iteration
  size_t subDivDim = 2 * numRegions;

  //----------------------------------------------------------

  // we also need to worry about the temporary arrays that are created to do the
  // copies

  size_t Ints_Size = 4 * (activeBisectDim + subDivDim);
  size_t Doubles_Size =
    8 * (newActiveRegions + scanned + newActiveRegionsLength +
         parentExpansionEstimate + parentExpansionErrorest + genRegions +
         genRegionsLength + regions + regionsLength + regionsIntegral +
         regionsError + parentsIntegral + parentsError);
  // the above doesn't consider the intermediate memory needed
  return Ints_Size + Doubles_Size;
}

bool
sigDigitsSame(double x, double y, double z, int requiredDigits)
{
  double third = abs(x);
  double second = abs(y);
  double first = abs(z);
  // printf("Comparing digits on %.15f %.15f %.15f\n", first, second, third);
  while (first < 1.) {
    first *= 10;
  }
  while (second < 1.) {
    second *= 10;
  }
  while (third < 1.) {
    third *= 10;
  }
  // printf("Compared digits\n");
  std::string second_to_last = doubleToString(third, 15);
  std::string last = doubleToString(second, 15);
  std::string current = doubleToString(first, 15);

  bool verdict = true;
  int sigDigits = 0;

  for (int i = 0;
       i < requiredDigits + 1 && sigDigits < requiredDigits && verdict == true;
       ++i) {
    verdict =
      current[i] == last[i] && last[i] == second_to_last[i] ? true : false;
    sigDigits += (verdict == true && current[i] != '.') ? 1 : 0;
  }
  return verdict;
}

double
ComputeMax(ViewVectorDouble list)
{
  double max;
  Kokkos::parallel_reduce(
    list.extent(0),
    KOKKOS_LAMBDA(const int& index, double& lmax) {
      if (lmax < list(index))
        lmax = list(index);
    },
    Kokkos::Max<double>(max));
  return max;
}

double
ComputeMin(ViewVectorDouble list)
{
  double min;
  Kokkos::parallel_reduce(
    list.extent(0),
    KOKKOS_LAMBDA(const int& index, double& lmin) {
      if (lmin > list(index))
        lmin = list(index);
    },
    Kokkos::Min<double>(min));
  return min;
}

double
exclusive_prefix_scan(ViewVectorInt input, ViewVectorInt output)
{
  int update = 0.;

  Kokkos::parallel_scan(
    input.extent(0), KOKKOS_LAMBDA(const int i, int& update, const bool final) {
      const int val_i = input(i);
      if (final) {
        output(i) = update;
      }
      update += val_i;
    });
  return update;
}

void
ExpandcuArray(ViewVectorDouble& array, int currentSize, int newSize)
{
  int copy_size = std::min(currentSize, newSize);
  // CHANGE THAT TO REALLOC AFTER, NO NEED TO COPY AT ALL
  Kokkos::realloc(array, newSize);
  /*if (newSize > currentSize) {
    // printf("resizing parents\n");
    Kokkos::resize(array, newSize);
  } else {
    // printf("reallocating parents\n");
    Kokkos::realloc(array, newSize);
  }*/
}

#endif