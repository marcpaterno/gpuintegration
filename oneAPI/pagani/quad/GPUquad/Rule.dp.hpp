#ifndef CUDACUHRE_QUAD_GPUQUAD_RULE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_RULE_CUH

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include "oneAPI/pagani/quad/util/cudaMemoryUtil.h"
#include "oneAPI/pagani/quad/quad.h"
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
        NDIM * (NDIM * (NDIM * (-.002361170967785511788400941242259231309691) +
                        .1141539002385732526821323741697655347686) +
                (-.6383392007670238909386026193674701393074)) +
          .7484998850468520800423030047583803945205,
        NDIM * (NDIM * (NDIM * (-.001432401703339912514196154599769007103671) +
                        .05747150786448972594860897296200006759892) +
                (-.1422510457143424323449521620935950679394)) -
          (-.06287502873828697998942424881040490136987),
        NDIM * (.2545911332489590890011611142429070613156) -
          (NDIM *
             (NDIM * (NDIM * (-.001432401703339912514196154599769007103671) +
                      .05747150786448972594860897296200006759892) +
              (-.1422510457143424323449521620935950679394)) -
           (-.06287502873828697998942424881040490136987)),
        NDIM * (NDIM * (-1.207328566678236261002219995185143356737) +
                .8956736576416067650809467826488567200939) -
          1 +
          NDIM *
            (NDIM * (NDIM * (-.002361170967785511788400941242259231309691) +
                     .1141539002385732526821323741697655347686) +
             (-.6383392007670238909386026193674701393074)) +
          .7484998850468520800423030047583803945205,
        NDIM * (-.3647935698604914666100134551377381205297) + 1 -
          (NDIM *
             (NDIM * (NDIM * (-.002361170967785511788400941242259231309691) +
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

      /*T cRule9Wt[]={
            NDIM*(NDIM*(NDIM*(-.0023611709677855117884) + .11415390023857325268)
  + (-.63833920076702389094)) + .74849988504685208004,
  NDIM*(NDIM*(NDIM*(-.0014324017033399125142) + .057471507864489725949) +
  (-.14225104571434243234)) - (-.062875028738286979989),
  NDIM*(.254591133248959089) - (NDIM*(NDIM*(NDIM*(-.0014324017033399125142) +
  .057471507864489725949) + (-.14225104571434243234)) -
  (-.062875028738286979989)), NDIM*(NDIM*(-1.207328566678236261) +
  .89567365764160676508) - 1 + NDIM*(NDIM*(NDIM*(-.0023611709677855117884) +
  .11415390023857325268) + (-.63833920076702389094)) + .74849988504685208004,
  NDIM*(-.36479356986049146661) + 1 -
  (NDIM*(NDIM*(NDIM*(-.0023611709677855117884) + .11415390023857325268) +
  (-.63833920076702389094)) + .74849988504685208004),

            NDIM*(NDIM*.0035417564516782676826 +(-.072609367395893679605)) +
  .10557491625218991012, NDIM*(NDIM*.0021486025550098687713 +
  (-.032268563892953949998)) + .010636783990231217481, .014689102496143490175 -
  (NDIM*(NDIM*.0021486025550098687713 + (-.032268563892953949998)) +
  .010636783990231217481), NDIM*.51134708346467591431 + .45976448120806344646 +
  NDIM*(NDIM*.0035417564516782676826 + (-.072609367395893679605)) +
  .10557491625218991012, .18239678493024573331 -
  (NDIM*(NDIM*.0035417564516782676826 + (-.072609367395893679605)) +
  .10557491625218991012),

            NDIM*(-.04508628929435784076) + .21415883524352793401,
  NDIM*(-.027351546526545644722) + .054941067048711234101, .11937596202570775297
  - (NDIM*(-.027351546526545644722) + .054941067048711234101),
  NDIM*.65089519391920250593 + .14744939829434460168,
  -(NDIM*(-.04508628929435784076) + .21415883524352793401),

            .057693384490973483573, .034999626602143583822,
  -.057693384490973483573, -1.3868627719278281436, -.057693384490973483573,

            0, 0, -.2386668732575008879, 0, 0,

            .015532417276607053264 - NDIM*.0035417564516782676826,
  .0035328099607090870236 - NDIM*.0021486025550098687713,
  -(.0035328099607090870236 - NDIM*.0021486025550098687713),
  .09231719987444221619 + .015532417276607053264 - NDIM*.0035417564516782676826,
  -(.015532417276607053264 - NDIM*.0035417564516782676826),

            .02254314464717892038, .013675773263272822361,
  -.013675773263272822361, -.32544759695960125297, -.02254314464717892038,

            .0017708782258391338413, .0010743012775049343856,
  -.0010743012775049343856, .0017708782258391338413, -.0017708782258391338413,


            .25150011495314791996/(1<<NDIM), -.062875028738286979989/(1<<NDIM),
  -(-.062875028738286979989/(1<<NDIM)), .25150011495314791996/(1<<NDIM),
  -(.25150011495314791996/(1<<NDIM))
  };*/

      CPURuleWt =
        (T*)Host.AllocateMemory((void**)CPURuleWt, sizeof(T) * NSETS * NRULES);
      for (int i = 0; i < NSETS * NRULES; ++i) {
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

      //#if DIM >= 3
      if (NDIM >= 3) {
        indxCnt9[7] = 3;
        indxCnt9[8] = NDIM;
      } else if (NDIM == 2) {
        indxCnt9[7] = 0;
        indxCnt9[8] = NDIM;
      }
      // int indxCnt9[]={0, 1, 1, 1, 1, 2, 2, 3, NDIM};
      //#elif DIM == 2
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
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      q_ct1.memcpy(tmp, array, sizeof(K) * size).wait();
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
    loadDeviceConstantMemory(Structures<T>* constMem, int device = 0)
    {
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      //Device.DeviceInit(device, VERBOSE);

      constMem->_gpuG = sycl::malloc_device<T>(NDIM * NSETS, q_ct1);
      constMem->_cRuleWt = sycl::malloc_device<double>(NRULES * NSETS, q_ct1);
	  constMem->_cGeneratorCount = sycl::malloc_device<size_t>(NSETS, q_ct1);
      constMem->_GPUScale = sycl::malloc_device<T>(NSETS * NRULES, q_ct1);
      constMem->_GPUNorm = sycl::malloc_device<T>(NSETS * NRULES, q_ct1);
      constMem->_gpuGenPos = sycl::malloc_device<int>(PERMUTATIONS_POS_ARRAY_SIZE, q_ct1);
	  constMem->_gpuGenPermVarCount = sycl::malloc_device<int>(FEVAL, q_ct1);
	  constMem->_gpuGenPermGIndex = sycl::malloc_device<int>(FEVAL, q_ct1);
      constMem->_gpuGenPermVarStart =sycl::malloc_device<int>((FEVAL + 1), q_ct1);
      
      q_ct1.memcpy(constMem->_gpuG, cpuG,                       sizeof(T) * NDIM * NSETS).wait();
      q_ct1.memcpy(constMem->_cRuleWt, CPURuleWt,               sizeof(T) * NRULES * NSETS).wait();
      q_ct1.memcpy(constMem->_cGeneratorCount,CPUGeneratorCount,sizeof(size_t) * NSETS).wait();
      q_ct1.memcpy(constMem->_GPUScale, CPUScale,               sizeof(T) * NSETS * NRULES).wait();
      q_ct1.memcpy(constMem->_GPUNorm, CPUNorm,                 sizeof(T) * NSETS * NRULES).wait();
      q_ct1.memcpy(constMem->_gpuGenPos,genPtr,                 sizeof(int) * PERMUTATIONS_POS_ARRAY_SIZE).wait();
      q_ct1.memcpy(constMem->_gpuGenPermVarCount,cpuGenPermVarCount, sizeof(int) * FEVAL).wait();
      q_ct1.memcpy(constMem->_gpuGenPermGIndex, cpuGenPermGIndex, sizeof(int) * FEVAL).wait();
      q_ct1.memcpy(constMem->_gpuGenPermVarStart, cpuGenPermVarStart, sizeof(int) * (FEVAL + 1)).wait();
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
