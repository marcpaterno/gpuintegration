#ifndef KOKKOSCUHRE_RULE_CUH
#define KOKKOSCUHRE_RULE_CUH
#include "kokkos/kokkosPagani/quad/quad.h"
#include "kokkos/kokkosPagani/quad/util/print.cuh"

template <typename T>
class Rule {
  HostVectorDouble cpuG;
  HostVectorDouble CPURuleWt;
  HostVectorDouble CPUScale, CPUNorm;

  HostVectorInt cpuGenCount, indxCnt;
  HostVectorSize_t CPUGeneratorCount;
  HostVectorInt cpuGenPermVarCount, cpuGenPermGIndex, cpuGenPermVarStart;
  HostVectorInt genPtr;

  int KEY, RULE, NSETS, FEVAL, NDIM, PERMUTATIONS_POS_ARRAY_SIZE, VERBOSE;
  size_t fEvalPerRegion;

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
      NDIM * (NDIM * (NDIM * (-.002361170967785511788400941242259231309691) +
                      .1141539002385732526821323741697655347686) +
              (-.6383392007670238909386026193674701393074)) +
        .7484998850468520800423030047583803945205,
      NDIM * (NDIM * (NDIM * (-.001432401703339912514196154599769007103671) +
                      .05747150786448972594860897296200006759892) +
              (-.1422510457143424323449521620935950679394)) -
        (-.06287502873828697998942424881040490136987),
      NDIM * (.2545911332489590890011611142429070613156) -
        (NDIM * (NDIM * (NDIM * (-.001432401703339912514196154599769007103671) +
                         .05747150786448972594860897296200006759892) +
                 (-.1422510457143424323449521620935950679394)) -
         (-.06287502873828697998942424881040490136987)),
      NDIM * (NDIM * (-1.207328566678236261002219995185143356737) +
              .8956736576416067650809467826488567200939) -
        1 +
        NDIM * (NDIM * (NDIM * (-.002361170967785511788400941242259231309691) +
                        .1141539002385732526821323741697655347686) +
                (-.6383392007670238909386026193674701393074)) +
        .7484998850468520800423030047583803945205,
      NDIM * (-.3647935698604914666100134551377381205297) + 1 -
        (NDIM * (NDIM * (NDIM * (-.002361170967785511788400941242259231309691) +
                         .1141539002385732526821323741697655347686) +
                 (-.6383392007670238909386026193674701393074)) +
         .7484998850468520800423030047583803945205),

      NDIM * (NDIM * .003541756451678267682601411863388846964536 +
              (-.07260936739589367960492815865074633743652)) +
        .1055749162521899101218622863269817454540,
      NDIM * (NDIM * .002148602555009868771294231899653510655506 +
              (-.03226856389295394999786630399875134318006)) +
        .01063678399023121748083624225818915724455,
      .01468910249614349017540783437728097691502 -
        (NDIM * (NDIM * .002148602555009868771294231899653510655506 +
                 (-.03226856389295394999786630399875134318006)) +
         .01063678399023121748083624225818915724455),
      NDIM * .5113470834646759143109387357149329909126 +
        .4597644812080634464633352781605214342691 +
        NDIM * (NDIM * .003541756451678267682601411863388846964536 +
                (-.07260936739589367960492815865074633743652)) +
        .1055749162521899101218622863269817454540,
      .1823967849302457333050067275688690602649 -
        (NDIM * (NDIM * .003541756451678267682601411863388846964536 +
                 (-.07260936739589367960492815865074633743652)) +
         .1055749162521899101218622863269817454540),

      NDIM * (-.04508628929435784075980562738240804429658) +
        .2141588352435279340097929526588394300172,
      NDIM * (-.02735154652654564472203690086290223507436) +
        .05494106704871123410060080562462135546101,
      .1193759620257077529708962121565290178730 -
        (NDIM * (-.02735154652654564472203690086290223507436) +
         .05494106704871123410060080562462135546101),
      NDIM * .6508951939192025059314756320878023215278 +
        .1474493982943446016775696826942585013243,
      -(NDIM * (-.04508628929435784075980562738240804429658) +
        .2141588352435279340097929526588394300172),

      .05769338449097348357291272840392627722165,
      .03499962660214358382244159694487155861542,
      -.05769338449097348357291272840392627722165,
      -1.386862771927828143599782668709014266770,
      -.05769338449097348357291272840392627722165,

      0,
      0,
      -.2386668732575008878964134721962088068396,
      0,
      0,

      .01553241727660705326386197156586357005224 -
        NDIM * .003541756451678267682601411863388846964536,
      .003532809960709087023561817517751309380604 -
        NDIM * .002148602555009868771294231899653510655506,
      -(.003532809960709087023561817517751309380604 -
        NDIM * .002148602555009868771294231899653510655506),
      .09231719987444221619017126187763868745587 +
        .01553241727660705326386197156586357005224 -
        NDIM * .003541756451678267682601411863388846964536,
      -(.01553241727660705326386197156586357005224 -
        NDIM * .003541756451678267682601411863388846964536),

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

      .2515001149531479199576969952416196054795 / (1 << NDIM),
      -.06287502873828697998942424881040490136987 / (1 << NDIM),
      -(-.06287502873828697998942424881040490136987 / (1 << NDIM)),
      .2515001149531479199576969952416196054795 / (1 << NDIM),
      -(.2515001149531479199576969952416196054795 / (1 << NDIM))};

    CPURuleWt = HostVectorDouble("CPURuleWt", NSETS * NRULES);
    for (int i = 0; i < NSETS * NRULES; ++i) {
      CPURuleWt(i) = cRule9Wt[i];
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
    if (NDIM >= 3) {
      indxCnt9[7] = 3;
      indxCnt9[8] = NDIM;
    } else if (NDIM == 2) {
      indxCnt9[7] = 0;
      indxCnt9[8] = NDIM;
    }

    CPUGeneratorCount = HostVectorSize_t("CPUGeneratorCount", NSETS);
    for (int i = 0; i < NSETS; ++i) {
      CPUGeneratorCount(i) = CPUGeneratorCount9[i];
    }

    T cpuRule9G[] = {.4779536579022695061928604197171830064732,
                     .2030285873691198677998034402373279133258,
                     .4476273546261781288207704806530998539285,
                     .125,
                     .3430378987808781457001426145164678603407};

    cpuG = HostVectorDouble("cpuG", NDIM * NSETS);
    cpuGenCount = HostVectorInt("cpuGenCount", NSETS);
    indxCnt = HostVectorInt("indxCnt", NSETS);
    for (int iter = 0; iter < 9; ++iter) {
      cpuGenCount(iter) = cpuGenCount9[iter];
      indxCnt(iter) = indxCnt9[iter];
    }

    for (int i = 0; i < NDIM * NSETS; ++i) {
      cpuG(i) = 0.0;
    }

    cpuG(NDIM) = cpuRule9G[0];
    //{a2, 0, 0, 0,...., 0}
    cpuG(NDIM * 2) = cpuRule9G[1];
    //{a3, 0, 0, 0,...., 0}
    cpuG(NDIM * 3) = cpuRule9G[2];
    //{a4, 0, 0, 0,...., 0}
    cpuG(NDIM * 4) = cpuRule9G[3];

    //{b, b, 0, 0,...., 0}
    cpuG(NDIM * 5) = cpuRule9G[0];
    cpuG(NDIM * 5 + 1) = cpuRule9G[0];

    //{y, d, 0, 0,...., 0}
    cpuG(NDIM * 6) = cpuRule9G[0];
    cpuG(NDIM * 6 + 1) = cpuRule9G[1];

    //{e, e, e, 0,...., 0}
    cpuG(NDIM * 7) = cpuRule9G[0];
    cpuG(NDIM * 7 + 1) = cpuRule9G[0];
    cpuG(NDIM * 7 + 2) = cpuRule9G[0];

    //{l, l, l, ...., l}
    for (int dim = 0; dim < NDIM; ++dim) {
      cpuG(NDIM * 8 + dim) = cpuRule9G[4];
    }

    CPUScale = HostVectorDouble("CPUScale", NSETS * NRULES);
    CPUNorm = HostVectorDouble("CPUNorm", NSETS * NRULES);

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
        CPUScale(idx * NRULES + r) = scale;
        CPUNorm(idx * NRULES + r) = 1 / sum;
      }
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

  void
  loadDeviceConstantMemory(Structures<T>* constMem)
  {

    ViewVectorDouble _gpuG("_gpuG", NDIM * NSETS);
    ViewVectorDouble _cRuleWt("_cRuleWt", NRULES * NSETS);

    ViewVectorSize_t _cGeneratorCount("_cGeneratorCount", NSETS);
    ViewVectorDouble _GPUScale("_GPUScale", NSETS * NRULES);
    ViewVectorDouble _GPUNorm("_GPUNorm", NSETS * NRULES);
    ViewVectorInt _gpuGenPos("_gpuGenPos", PERMUTATIONS_POS_ARRAY_SIZE);
    ViewVectorInt _gpuGenPermVarCount("_gpuGenPermVarCount", FEVAL);
    ViewVectorInt _gpuGenPermGIndex("_gpuGenPermGIndex", FEVAL);
    ViewVectorInt _gpuGenPermVarStart("_gpuGenPermVarStart", FEVAL + 1);

    Kokkos::deep_copy(_gpuG, cpuG);
    Kokkos::deep_copy(_cRuleWt, CPURuleWt);
    Kokkos::deep_copy(_cGeneratorCount, CPUGeneratorCount);
    Kokkos::deep_copy(_GPUScale, CPUScale);
    Kokkos::deep_copy(_GPUNorm, CPUNorm);
    Kokkos::deep_copy(_gpuGenPos, genPtr);
    Kokkos::deep_copy(_gpuGenPermVarCount, cpuGenPermVarCount);
    Kokkos::deep_copy(_gpuGenPermGIndex, cpuGenPermGIndex);
    Kokkos::deep_copy(_gpuGenPermVarStart, cpuGenPermVarStart);

    constMem->_gpuG = _gpuG;
    constMem->_cRuleWt = _cRuleWt;

    constMem->_cGeneratorCount = _cGeneratorCount;
    constMem->_GPUScale = _GPUScale;
    constMem->_GPUNorm = _GPUNorm;
    constMem->_gpuGenPos = _gpuGenPos;
    constMem->_gpuGenPermVarCount = _gpuGenPermVarCount;
    constMem->_gpuGenPermGIndex = _gpuGenPermGIndex;
    constMem->_gpuGenPermVarStart = _gpuGenPermVarStart;

    /*constMem->_gpuG = ViewVectorDouble("_gpuG", NDIM * NSETS);
            constMem->_cRuleWt = ViewVectorDouble("_cRuleWt", NRULES * NSETS);

            constMem->_cGeneratorCount = ViewVectorSize_t("_cGeneratorCount",
       NSETS); constMem->_GPUScale = ViewVectorDouble("_GPUScale", NSETS *
       NRULES); constMem->_GPUNorm = ViewVectorDouble("_GPUNorm", NSETS *
       NRULES); constMem->_gpuGenPos = ViewVectorInt("_gpuGenPos",
       PERMUTATIONS_POS_ARRAY_SIZE); constMem->_gpuGenPermVarCount =
       ViewVectorInt("_gpuGenPermVarCount", FEVAL); constMem->_gpuGenPermGIndex
       = ViewVectorInt("_gpuGenPermGIndex", FEVAL);
            constMem->_gpuGenPermVarStart = ViewVectorInt("_gpuGenPermVarStart",
       FEVAL + 1);*/

    /*Kokkos::deep_copy(constMem->_gpuG, cpuG);
            Kokkos::deep_copy(constMem->_cRuleWt, CPURuleWt);
            Kokkos::deep_copy(constMem->_cGeneratorCount, CPUGeneratorCount);
            Kokkos::deep_copy(constMem->_GPUScale, CPUScale);
            Kokkos::deep_copy(constMem->_GPUNorm, CPUNorm);
            Kokkos::deep_copy(constMem->_gpuGenPos, genPtr);
            Kokkos::deep_copy(constMem->_gpuGenPermVarCount,
       cpuGenPermVarCount); Kokkos::deep_copy(constMem->_gpuGenPermGIndex,
       cpuGenPermGIndex); Kokkos::deep_copy(constMem->_gpuGenPermVarStart,
       cpuGenPermVarStart);*/
  }

  void
  Init(int ndim, size_t fEval, int key, int verbose, Structures<T>* constMem)
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
    cpuGenPermVarCount = HostVectorInt("cpuGenPermVarCount", fEval);
    cpuGenPermVarStart = HostVectorInt("cpuGenPermVarStart", fEval + 1);
    cpuGenPermGIndex = HostVectorInt("cpuGenPermGIndex", (fEval /*+ 1*/));

    HostVectorDouble cpuGCopy("cpuGCopy", NDIM * NSETS);

    for (int iter = 0; iter < NDIM * NSETS; ++iter) {
      cpuGCopy(iter) = cpuG(iter);
    }

    genPtr = HostVectorInt("genPtr", PERMUTATIONS_POS_ARRAY_SIZE);

    int genPtrPosIndex = 0, permCnt = 0;
    cpuGenPermVarStart(0) =
      0; // To make sure length of first permutation is ZERO

    for (int gIndex = 0; gIndex < NSETS; ++gIndex) {
      int n = cpuGenCount(gIndex), flag = 1;
      int num_permutation = 0;
      T* g = &cpuGCopy(NDIM * gIndex);
      while (num_permutation < n) {
        num_permutation++;
        flag = 1;
        int genPosCnt = 0;
        cpuGenPermVarStart(permCnt) = genPtrPosIndex;
        int isAccess[NDIM];
        for (int dim = 0; dim < NDIM; ++dim) {
          isAccess[dim] = 0;
        }
        for (int i = 0; i < indxCnt[gIndex]; ++i) {
          // Find pos of cpuG[i]
          for (int dim = 0; dim < NDIM; ++dim) {
            if (cpuG(NDIM * gIndex + i) == fabs(g[dim]) && !isAccess[dim]) {
              ++genPosCnt;
              isAccess[dim] = 1;

              if (g[dim] < 0) {
                genPtr(genPtrPosIndex++) = -(dim + 1);
              } else {
                genPtr(genPtrPosIndex++) = dim + 1;
              }
              break;
            }
          }
        }

        permCnt++;
        cpuGenPermVarCount(permCnt - 1) = genPosCnt;
        cpuGenPermGIndex(permCnt - 1) = gIndex;
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

    cpuGenPermVarStart(permCnt) = genPtrPosIndex;
    genPtrPosIndex = 0;
    loadDeviceConstantMemory(constMem);
  }
};

#endif