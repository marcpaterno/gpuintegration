#ifndef CUDACUHRE_QUAD_GPUQUAD_RULE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_RULE_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "dpct-exp/common/cuda/cudaMemoryUtil.h"
#include "dpct-exp/cuda/pagani/quad/quad.h"
#include <cmath>

namespace quad {
  template <typename T>
  class Rule {

    T* cpuG;
    T* CPURuleWt;
    T *CPUScale, *CPUNorm;
    int *cpuGenCount, *indxCnt, *CPUGeneratorCount;
    int *cpuGenPermVarCount, *cpuGenPermGIndex, *cpuGenPermVarStart;
    int* genPtr;
    int KEY, RULE, NSETS, FEVAL, NDIM, PERMUTATIONS_POS_ARRAY_SIZE, VERBOSE;

    // int NRULES;
    size_t fEvalPerRegion;
    DeviceMemory<T> Device;
    HostMemory<T> Host;

    void
    Rule9Generate()
    {
      int indxCnt9[9];
      indxCnt9[0] = 0;
      indxCnt9[1] = 1;
      indxCnt9[2] = 1;
      indxCnt9[3] = 1;
      indxCnt9[4] = 1;
      indxCnt9[5] = 2;
      indxCnt9[6] = 2;

      T cRule9Wt[] = {
        NDIM *
            (NDIM * (NDIM * (T)(-.002361170967785511788400941242259231309691) +
                     (T).1141539002385732526821323741697655347686) +
             (T)(-.6383392007670238909386026193674701393074)) +
          (T).7484998850468520800423030047583803945205,
        NDIM *
            (T)(NDIM *
                  (T)(NDIM * (T)(-.001432401703339912514196154599769007103671) +
                      (T).05747150786448972594860897296200006759892) +
                (T)(-.1422510457143424323449521620935950679394)) -
          (T)(-.06287502873828697998942424881040490136987),
        (T)NDIM * (T)((T).2545911332489590890011611142429070613156) -
          (T)(NDIM *

                (T)(NDIM *
                      (T)(NDIM *
                            (T)(-.001432401703339912514196154599769007103671) +
                          (T).05747150786448972594860897296200006759892) +
                    (T)(-.1422510457143424323449521620935950679394)) -
              (T)(-.06287502873828697998942424881040490136987)),
        NDIM * (T)(NDIM * (T)(-1.207328566678236261002219995185143356737) +
                   .8956736576416067650809467826488567200939) -
          1 +
          NDIM *
            (T)(NDIM *
                  (T)(NDIM * (T)(-.002361170967785511788400941242259231309691) +
                      .1141539002385732526821323741697655347686) +
                (T)(-.6383392007670238909386026193674701393074)) +
          (T).7484998850468520800423030047583803945205,
        NDIM * (T)(-.3647935698604914666100134551377381205297) + 1 -
          (T)(NDIM *
                (T)(NDIM *
                      (T)(NDIM *
                            (T)(-.002361170967785511788400941242259231309691) +
                          (T).1141539002385732526821323741697655347686) +
                    (-.6383392007670238909386026193674701393074)) +
              .7484998850468520800423030047583803945205),

        NDIM * (T)(NDIM * (T).003541756451678267682601411863388846964536 +
                   (T)(-.07260936739589367960492815865074633743652)) +
          (T).1055749162521899101218622863269817454540,
        NDIM * (T)(NDIM * (T).002148602555009868771294231899653510655506 +
                   (-(T).03226856389295394999786630399875134318006)) +
          (T).01063678399023121748083624225818915724455,
        (T).01468910249614349017540783437728097691502 -
          (T)(NDIM * (T)(NDIM * .002148602555009868771294231899653510655506 +
                         (T)(-.03226856389295394999786630399875134318006)) +
              (T).01063678399023121748083624225818915724455),
        NDIM * (T).5113470834646759143109387357149329909126 +
          (T).4597644812080634464633352781605214342691 +
          NDIM * (T)(NDIM * .003541756451678267682601411863388846964536 +
                     (T)(-.07260936739589367960492815865074633743652)) +
          (T).1055749162521899101218622863269817454540,
        (T).1823967849302457333050067275688690602649 -
          (T)(NDIM * (T)(NDIM * .003541756451678267682601411863388846964536 +
                         (T)(-.07260936739589367960492815865074633743652)) +
              (T).1055749162521899101218622863269817454540),

        NDIM * (T)(-(T).04508628929435784075980562738240804429658) +
          (T).2141588352435279340097929526588394300172,
        NDIM * (T)(-.02735154652654564472203690086290223507436) +
          (T).05494106704871123410060080562462135546101,
        (T).1193759620257077529708962121565290178730 -

          (T)(NDIM * (-(T).02735154652654564472203690086290223507436) +
              (T).05494106704871123410060080562462135546101), // her
        NDIM * (T).6508951939192025059314756320878023215278 +
          (T).1474493982943446016775696826942585013243,
        -(T)(NDIM * (T)(-.04508628929435784075980562738240804429658) +
             (T).2141588352435279340097929526588394300172),
        (T).05769338449097348357291272840392627722165,
        .03499962660214358382244159694487155861542,
        -.05769338449097348357291272840392627722165,
        -1.386862771927828143599782668709014266770,
        -.05769338449097348357291272840392627722165,

        0,
        0,
        -.2386668732575008878964134721962088068396,
        0,
        0,

        (T).01553241727660705326386197156586357005224 -
          NDIM * (T).003541756451678267682601411863388846964536,
        (T).003532809960709087023561817517751309380604 -
          NDIM * (T).002148602555009868771294231899653510655506,
        -(T)(.003532809960709087023561817517751309380604 -

             NDIM * (T).002148602555009868771294231899653510655506),
        (T).09231719987444221619017126187763868745587 +
          (T).01553241727660705326386197156586357005224 -
          NDIM * (T).003541756451678267682601411863388846964536,
        -(T)(.01553241727660705326386197156586357005224 -
             NDIM * (T).003541756451678267682601411863388846964536),

        .02254314464717892037990281369120402214829,
        .01367577326327282236101845043145111753718,
        -.01367577326327282236101845043145111753718,
        -.3254475969596012529657378160439011607639,
        -.02254314464717892037990281369120402214829,

        .001770878225839133841300705931694423482268,
        .001074301277504934385647115949826755327753,
        -.001074301277504934385647115949826755327753,
        .001770878225839133841300705931694423482268,
        -.001770878225839133841300705931694423482268,
        (T).2515001149531479199576969952416196054795 / (T)(1 << NDIM),
        -(T).06287502873828697998942424881040490136987 / (T)(1 << NDIM),
        -(T)(-.06287502873828697998942424881040490136987 / (T)(1 << NDIM)),
        (T).2515001149531479199576969952416196054795 / (T)(1 << NDIM),
        -(T)(.2515001149531479199576969952416196054795 / (T)(1 << NDIM))};

      CPURuleWt =
        (T*)Host.AllocateMemory((void**)CPURuleWt, sizeof(T) * NSETS * NRULES);
      for (int i = 0; i < NSETS * (int)NRULES; ++i) {
        CPURuleWt[i] = cRule9Wt[i];
      }

      int CPUGeneratorCount9[] = {1,
                                  2 * NDIM,
                                  2 * NDIM,
                                  2 * NDIM,
                                  2 * NDIM,
                                  2 * NDIM * (NDIM - 1),
                                  4 * NDIM * (NDIM - 1),
                                  4 * NDIM * (NDIM - 1) * (NDIM - 2) / 3,
                                  1 << NDIM};
      int cpuGenCount9[] = {1,
                            2 * NDIM,
                            2 * NDIM,
                            2 * NDIM,
                            2 * NDIM,
                            2 * NDIM * (NDIM - 1),
                            4 * NDIM * (NDIM - 1),
                            4 * NDIM * (NDIM - 1) * (NDIM - 2) / 3,
                            1 << NDIM};

      // #if DIM >= 3
      if (NDIM >= 3) {
        indxCnt9[7] = 3;
        indxCnt9[8] = NDIM;
      } else if (NDIM == 2) {
        indxCnt9[7] = 0;
        indxCnt9[8] = NDIM;
      }
      // int indxCnt9[]={0, 1, 1, 1, 1, 2, 2, 3, NDIM};
      // #elif DIM == 2
      // int CPUGeneratorCount9[]={1, 2*NDIM, 2*NDIM, 2*NDIM, 2*NDIM,
      // 2*NDIM*(NDIM - 1), 2*NDIM*(NDIM - 1), 4*NDIM*(NDIM - 1), 0, 1 << NDIM};
      // int cpuGenCount9[]={1, 2*NDIM, 2*NDIM, 2*NDIM, 2*NDIM, 2*NDIM*(NDIM -
      // 1), 4*NDIM*(NDIM - 1), 4*NDIM*(NDIM - 1)*(NDIM - 2)/3, 1 << NDIM}; int
      // indxCnt9[]={0, 1, 1, 1, 1, 2, 2, 0, NDIM}; 	#endif
      CPUGeneratorCount = (int*)Host.AllocateMemory((void**)CPUGeneratorCount,
                                                    sizeof(int) * NSETS);
      for (int i = 0; i < NSETS; ++i) {
        CPUGeneratorCount[i] = CPUGeneratorCount9[i];
      }

      /*T cpuRule9G[] = {
                .47795365790226950619, .20302858736911986780,
                .44762735462617812882, .125,
                .34303789878087814570
      };*/

      T cpuRule9G[] = {.4779536579022695061928604197171830064732,
                       .2030285873691198677998034402373279133258,
                       .4476273546261781288207704806530998539285,
                       .125,
                       .3430378987808781457001426145164678603407};

      cpuG = (T*)Host.AllocateMemory((void*)cpuG, sizeof(T) * NDIM * NSETS);
      cpuGenCount =
        (int*)Host.AllocateMemory((void*)cpuGenCount, sizeof(int) * NSETS);
      indxCnt = (int*)Host.AllocateMemory((void*)indxCnt, sizeof(int) * NSETS);

      for (int iter = 0; iter < 9; ++iter) {
        cpuGenCount[iter] = cpuGenCount9[iter];
        indxCnt[iter] = indxCnt9[iter];
      }

      for (int i = 0; i < NDIM * NSETS; ++i) {
        cpuG[i] = 0.0;
      }

      // Compute Generators in CPU
      //{0, 0, 0, 0,...., 0}

      //{a1, 0, 0, 0,...., 0}
      cpuG[NDIM] = cpuRule9G[0];
      //{a2, 0, 0, 0,...., 0}
      cpuG[NDIM * 2] = cpuRule9G[1];
      //{a3, 0, 0, 0,...., 0}
      cpuG[NDIM * 3] = cpuRule9G[2];
      //{a4, 0, 0, 0,...., 0}
      cpuG[NDIM * 4] = cpuRule9G[3];

      //{b, b, 0, 0,...., 0}
      cpuG[NDIM * 5] = cpuRule9G[0];
      cpuG[NDIM * 5 + 1] = cpuRule9G[0];

      //{y, d, 0, 0,...., 0}
      cpuG[NDIM * 6] = cpuRule9G[0];
      cpuG[NDIM * 6 + 1] = cpuRule9G[1];

      //{e, e, e, 0,...., 0}
      cpuG[NDIM * 7] = cpuRule9G[0];
      cpuG[NDIM * 7 + 1] = cpuRule9G[0];
      cpuG[NDIM * 7 + 2] = cpuRule9G[0];

      //{l, l, l, ...., l}
      for (int dim = 0; dim < NDIM; ++dim) {
        cpuG[NDIM * 8 + dim] = cpuRule9G[4];
      }

      CPUScale =
        (T*)Host.AllocateMemory((void*)CPUScale, sizeof(T) * NSETS * NRULES);
      CPUNorm =
        (T*)Host.AllocateMemory((void*)CPUNorm, sizeof(T) * NSETS * NRULES);

      for (int idx = 0; idx < NSETS; ++idx) {
        T* s_weight = &cRule9Wt[idx * NRULES];
        for (int r = 1; r < NRULES - 1; ++r) {
          T scale = (s_weight[r] == 0) ? 100 : -s_weight[r + 1] / s_weight[r];
          T sum = 0;
          for (int x = 0; x < NSETS; ++x) {
            T* weight = &cRule9Wt[x * NRULES];
            sum +=
              CPUGeneratorCount9[x] * fabs(weight[r + 1] + scale * weight[r]);
          }
          CPUScale[idx * NRULES + r] = scale;
          CPUNorm[idx * NRULES + r] = 1 / sum;

          // printf("CPUNorm[%i]:%.15f\n",idx*(int)NRULES+r, CPUNorm[idx *
          // NRULES + r]);
        }
      }
    }
    //@brief Template function to display GPU device array variables
    template <class K>
    void
    display(K* array, size_t size)
    {
      K* tmp = (K*)malloc(sizeof(K) * size);
      dpct::get_default_queue().memcpy(tmp, array, sizeof(K) * size).wait();
      for (int i = 0; i < size; ++i) {
        // printf("%.20lf \n", (T)tmp[i]);
        std::cout.precision(17);
        std::cout << "list[" << i << "]:" << tmp[i] << std::endl;
      }
    }

  public:
    inline int
    GET_FEVAL()
    {
      return FEVAL;
    }
    inline int
    GET_NSETS()
    {
      return NSETS;
    }

    ~Rule()
    {
      Host.ReleaseMemory(cpuG);
      Host.ReleaseMemory(CPURuleWt);
      Host.ReleaseMemory(CPUScale);
      Host.ReleaseMemory(CPUNorm);
      Host.ReleaseMemory(cpuGenCount);
      Host.ReleaseMemory(indxCnt);
      Host.ReleaseMemory(CPUGeneratorCount);
      Host.ReleaseMemory(cpuGenPermVarCount);
      Host.ReleaseMemory(cpuGenPermGIndex);
      Host.ReleaseMemory(cpuGenPermVarStart);
      Host.ReleaseMemory(genPtr);
    }

    void
    loadDeviceConstantMemory(Structures<T>* constMem)
    {
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();
      /*
      DPCT1003:93: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(((constMem->gpuG) = (<dependent type>)sycl::malloc_device(
                   sizeof(T) * NDIM * NSETS, q_ct1),
                 0));

      /*
      DPCT1003:94: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(((constMem->cRuleWt) = (<dependent type>)sycl::malloc_device(
                   sizeof(T) * NRULES * NSETS, q_ct1),
                 0));
      /*
      DPCT1003:95: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(
        ((constMem->cGeneratorCount) =
           (<dependent type>)sycl::malloc_device(sizeof(size_t) * NSETS, q_ct1),
         0));
      /*
      DPCT1003:96: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(((constMem->GPUScale) = (<dependent type>)sycl::malloc_device(
                   sizeof(T) * NSETS * NRULES, q_ct1),
                 0));
      /*
      DPCT1003:97: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(((constMem->GPUNorm) = (<dependent type>)sycl::malloc_device(
                   sizeof(T) * NSETS * NRULES, q_ct1),
                 0));
      /*
      DPCT1003:98: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(((constMem->gpuGenPos) = (<dependent type>)sycl::malloc_device(
                   sizeof(int) * PERMUTATIONS_POS_ARRAY_SIZE, q_ct1),
                 0));
      /*
      DPCT1003:99: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(
        ((constMem->gpuGenPermVarCount) =
           (<dependent type>)sycl::malloc_device(sizeof(int) * FEVAL, q_ct1),
         0));
      /*
      DPCT1003:100: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(
        ((constMem->gpuGenPermGIndex) =
           (<dependent type>)sycl::malloc_device(sizeof(int) * FEVAL, q_ct1),
         0));
      /*
      DPCT1003:101: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(((constMem->gpuGenPermVarStart) = (<dependent type>)
                   sycl::malloc_device(sizeof(int) * (FEVAL + 1), q_ct1),
                 0));
      /*
      DPCT1003:102: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(
        (q_ct1.memcpy(constMem->gpuG, cpuG, sizeof(T) * NDIM * NSETS).wait(),
         0));
      /*
      DPCT1003:103: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(
        (q_ct1.memcpy(constMem->cRuleWt, CPURuleWt, sizeof(T) * NRULES * NSETS)
           .wait(),
         0));
      /*
      DPCT1003:104: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug((q_ct1
                   .memcpy(constMem->cGeneratorCount,
                           CPUGeneratorCount,
                           sizeof(size_t) * NSETS)
                   .wait(),
                 0));
      /*
      DPCT1003:105: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(
        (q_ct1.memcpy(constMem->GPUScale, CPUScale, sizeof(T) * NSETS * NRULES)
           .wait(),
         0));
      /*
      DPCT1003:106: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug(
        (q_ct1.memcpy(constMem->GPUNorm, CPUNorm, sizeof(T) * NSETS * NRULES)
           .wait(),
         0));
      /*
      DPCT1003:107: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug((q_ct1
                   .memcpy(constMem->gpuGenPos,
                           genPtr,
                           sizeof(int) * PERMUTATIONS_POS_ARRAY_SIZE)
                   .wait(),
                 0));
      /*
      DPCT1003:108: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug((q_ct1
                   .memcpy(constMem->gpuGenPermVarCount,
                           cpuGenPermVarCount,
                           sizeof(int) * FEVAL)
                   .wait(),
                 0));
      /*
      DPCT1003:109: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug((q_ct1
                   .memcpy(constMem->gpuGenPermGIndex,
                           cpuGenPermGIndex,
                           sizeof(int) * FEVAL)
                   .wait(),
                 0));
      /*
      DPCT1003:110: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      QuadDebug((q_ct1
                   .memcpy(constMem->gpuGenPermVarStart,
                           cpuGenPermVarStart,
                           sizeof(int) * (FEVAL + 1))
                   .wait(),
                 0));
    }

    void
    Init(size_t ndim,
         size_t fEval,
         int key,
         int verbose,
         Structures<T>* constMem)
    {
      NDIM = ndim;
      KEY = key;
      VERBOSE = verbose;
      fEvalPerRegion = fEval;

      if (key == 13 && ndim == 2)
        RULE = 13;
      else if (key == 1 && ndim == 3)
        RULE = 13;
      else if (key == 9)
        RULE = 9;
      else if (key == 7)
        RULE = 7;
      else if (ndim == 2)
        RULE = 9;
      // else if (ndim == 3)
      //  RULE = 11;
      else
        RULE = 9;

      // temporary
      RULE = 9;

      if (RULE == 13)
        NSETS = 14;
      else if (RULE == 11)
        NSETS = 13;
      else if (RULE == 9)
        NSETS = 9;
      else if (RULE == 7)
        NSETS = 6;

      FEVAL = (1 + 2 * ndim + 2 * ndim + 2 * ndim + 2 * ndim +
               2 * ndim * (ndim - 1) + 4 * ndim * (ndim - 1) +
               4 * ndim * (ndim - 1) * (ndim - 2) / 3 + (1 << ndim));
      PERMUTATIONS_POS_ARRAY_SIZE =
        (1 + 1 * 1 + 2 * ndim * 1 + 2 * ndim * 1 + 2 * ndim * 1 + 2 * ndim * 1 +
         2 * ndim * (ndim - 1) * 2 + 4 * ndim * (ndim - 1) * 2 +
         4 * ndim * (ndim - 1) * (ndim - 2) * 3 / 3 + ndim * (1 << ndim));
      // NRULES = 5;
      Rule9Generate();
      cpuGenPermVarCount = (int*)Host.AllocateMemory((void*)cpuGenPermVarCount,
                                                     sizeof(int) * fEval);
      cpuGenPermVarStart = (int*)Host.AllocateMemory((void*)cpuGenPermVarStart,
                                                     sizeof(int) * fEval + 1);
      cpuGenPermGIndex = (int*)Host.AllocateMemory((void*)cpuGenPermGIndex,
                                                   sizeof(int) * (fEval));
      T* cpuGCopy = 0;
      cpuGCopy =
        (T*)Host.AllocateMemory((void*)cpuGCopy, sizeof(T) * NDIM * NSETS);
      for (int iter = 0; iter < NDIM * NSETS; ++iter) {
        cpuGCopy[iter] = cpuG[iter];
      }

      genPtr = (int*)Host.AllocateMemory(
        (void*)genPtr, sizeof(int) * PERMUTATIONS_POS_ARRAY_SIZE);
      int genPtrPosIndex = 0, permCnt = 0;

      cpuGenPermVarStart[0] =
        0; // To make sure length of first permutation is ZERO

      for (int gIndex = 0; gIndex < NSETS; ++gIndex) {
        int n = cpuGenCount[gIndex], flag = 1;
        int num_permutation = 0;
        T* g = &cpuGCopy[NDIM * gIndex];
        while (num_permutation < n) {
          num_permutation++;
          flag = 1;
          int genPosCnt = 0;
          cpuGenPermVarStart[permCnt] = genPtrPosIndex;
          int isAccess[NDIM];

          for (int dim = 0; dim < NDIM; ++dim) {
            isAccess[dim] = 0;
          }

          for (int i = 0; i < indxCnt[gIndex]; ++i) {
            for (int dim = 0; dim < NDIM; ++dim) {
              if (cpuG[NDIM * gIndex + i] == fabs(g[dim]) && !isAccess[dim]) {
                ++genPosCnt;
                isAccess[dim] = 1;

                if (g[dim] < 0) {
                  genPtr[genPtrPosIndex++] = -(dim + 1);
                } else {
                  genPtr[genPtrPosIndex++] = dim + 1;
                }
                break;
              }
            }
          }

          permCnt++;
          cpuGenPermVarCount[permCnt - 1] = genPosCnt;
          cpuGenPermGIndex[permCnt - 1] = gIndex;

          for (int dim = 0; (dim < NDIM) && (flag == 1);) {
            g[dim] = -g[dim];
            if (g[dim++] < -0.0000000000000001) {
              flag = 0;
              break;
            }
          }

          for (int dim = 1; (dim < NDIM) && (flag == 1); ++dim) {
            T gd = g[dim];
            if (g[dim - 1] > gd) {
              size_t i, j = dim, ix = dim, dx = dim - 1;
              for (i = 0; i < --j; ++i) {
                T tmp = g[i];
                g[i] = g[j];
                g[j] = tmp;
                if (tmp <= gd)
                  --dx;
                if (g[i] > gd)
                  ix = i;
              }
              if (g[dx] <= gd)
                dx = ix;
              g[dim] = g[dx];
              g[dx] = gd;
              flag = 0;
              break;
            }
            if (flag == 0)
              break;
          }
        }
      }

      cpuGenPermVarStart[permCnt] = genPtrPosIndex;
      genPtrPosIndex = 0;
      loadDeviceConstantMemory(constMem);
    }
  };
}
#endif
