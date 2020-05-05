
namespace quad{
  template <typename T>
    class QuadRule{
    
    T *cpuG;
    T *CPURuleWt;
    T *CPUScale, *CPUNorm;
    int *cpuGenCount, *indxCnt, *CPUGeneratorCount;
    int *cpuGenPermVarCount, *cpuGenPermGIndex, *cpuGenPermVarStart;
    int *genPtr;
    int KEY, NDIM, VERBOSE;
    size_t fEvalPerRegion;
    DeviceMemory<T> Device;
    HostMemory<T> Host;
	

    void Rule9Generate(){
      T cRule9Wt[]={
	NDIM*(NDIM*(NDIM*(-.0023611709677855117884) + .11415390023857325268) + (-.63833920076702389094)) + .74849988504685208004, NDIM*(NDIM*(NDIM*(-.0014324017033399125142) + .057471507864489725949) + (-.14225104571434243234)) - (-.062875028738286979989), NDIM*(.254591133248959089) - (NDIM*(NDIM*(NDIM*(-.0014324017033399125142) + .057471507864489725949) + (-.14225104571434243234)) - (-.062875028738286979989)), NDIM*(NDIM*(-1.207328566678236261) + .89567365764160676508) - 1 + NDIM*(NDIM*(NDIM*(-.0023611709677855117884) + .11415390023857325268) + (-.63833920076702389094)) + .74849988504685208004, NDIM*(-.36479356986049146661) + 1 - (NDIM*(NDIM*(NDIM*(-.0023611709677855117884) + .11415390023857325268) + (-.63833920076702389094)) + .74849988504685208004),
  
	NDIM*(NDIM*.0035417564516782676826 +(-.072609367395893679605)) + .10557491625218991012, NDIM*(NDIM*.0021486025550098687713 + (-.032268563892953949998)) + .010636783990231217481, .014689102496143490175 - (NDIM*(NDIM*.0021486025550098687713 + (-.032268563892953949998)) + .010636783990231217481), NDIM*.51134708346467591431 + .45976448120806344646 + NDIM*(NDIM*.0035417564516782676826 + (-.072609367395893679605)) + .10557491625218991012, .18239678493024573331 - (NDIM*(NDIM*.0035417564516782676826 + (-.072609367395893679605)) + .10557491625218991012),

	NDIM*(-.04508628929435784076) + .21415883524352793401, NDIM*(-.027351546526545644722) + .054941067048711234101, .11937596202570775297 - (NDIM*(-.027351546526545644722) + .054941067048711234101), NDIM*.65089519391920250593 + .14744939829434460168, -(NDIM*(-.04508628929435784076) + .21415883524352793401),
  
	.057693384490973483573, .034999626602143583822, -.057693384490973483573, -1.3868627719278281436, -.057693384490973483573,

	0, 0, -.2386668732575008879, 0, 0,

	.015532417276607053264 - NDIM*.0035417564516782676826, .0035328099607090870236 - NDIM*.0021486025550098687713, -(.0035328099607090870236 - NDIM*.0021486025550098687713),  .09231719987444221619 + .015532417276607053264 - NDIM*.0035417564516782676826, -(.015532417276607053264 - NDIM*.0035417564516782676826),

	.02254314464717892038, .013675773263272822361, -.013675773263272822361, -.32544759695960125297, -.02254314464717892038,

	.0017708782258391338413, .0010743012775049343856, -.0010743012775049343856, .0017708782258391338413, -.0017708782258391338413,

  
	.25150011495314791996/(1<<NDIM), -.062875028738286979989/(1<<NDIM), -(-.062875028738286979989/(1<<NDIM)), .25150011495314791996/(1<<NDIM), -(.25150011495314791996/(1<<NDIM))
      };
      CPURuleWt = (T *)Host.AllocateMemory((void **)CPURuleWt, sizeof(T) * NSETS * NRULES);
      for(int i = 0; i < NSETS * NRULES; ++i){
		CPURuleWt[i] = cRule9Wt[i];
      }
      
      int CPUGeneratorCount9[]={1, 2*NDIM, 2*NDIM, 2*NDIM, 2*NDIM, 2*NDIM*(NDIM - 1), 4*NDIM*(NDIM - 1), 4*NDIM*(NDIM - 1)*(NDIM - 2)/3, 1 << NDIM};
      int cpuGenCount9[]={1, 2*NDIM, 2*NDIM, 2*NDIM, 2*NDIM, 2*NDIM*(NDIM - 1), 4*NDIM*(NDIM - 1), 4*NDIM*(NDIM - 1)*(NDIM - 2)/3, 1 << NDIM};
      
#if DIM >= 3
      int indxCnt9[]={0, 1, 1, 1, 1, 2, 2, 3, NDIM};
#elif DIM == 2
      //int CPUGeneratorCount9[]={1, 2*NDIM, 2*NDIM, 2*NDIM, 2*NDIM, 2*NDIM*(NDIM - 1), 2*NDIM*(NDIM - 1), 4*NDIM*(NDIM - 1), 0, 1 << NDIM};
      //int cpuGenCount9[]={1, 2*NDIM, 2*NDIM, 2*NDIM, 2*NDIM, 2*NDIM*(NDIM - 1), 4*NDIM*(NDIM - 1), 4*NDIM*(NDIM - 1)*(NDIM - 2)/3, 1 << NDIM};
      int indxCnt9[]={0, 1, 1, 1, 1, 2, 2, 0, NDIM};
#endif
      CPUGeneratorCount = (int *)Host.AllocateMemory((void **)CPUGeneratorCount, sizeof(int) * NSETS);
      for(int i =0; i < NSETS; ++i){
		CPUGeneratorCount[i] = CPUGeneratorCount9[i];
      }

      T cpuRule9G[] = {
		.47795365790226950619, .20302858736911986780,
		.44762735462617812882, .125,
		.34303789878087814570 
      };

      cpuG = (T *)Host.AllocateMemory((void *)cpuG, sizeof(T) * NDIM * NSETS); 
      cpuGenCount = (int *)Host.AllocateMemory((void *)cpuGenCount, sizeof(int) * NSETS);
      indxCnt = (int *)Host.AllocateMemory((void *)indxCnt, sizeof(int) * NSETS);

      for(int iter = 0; iter < 9 ; ++iter){
		cpuGenCount[iter] = cpuGenCount9[iter];
		indxCnt[iter]=indxCnt9[iter];
      }

      for(int i = 0; i < NDIM *NSETS; ++i){
		cpuG[i] = 0.0;
      }

      //Compute Generators in CPU
      //{0, 0, 0, 0,...., 0}
  
      //{a1, 0, 0, 0,...., 0}
      cpuG[NDIM]=cpuRule9G[0];
      //{a2, 0, 0, 0,...., 0}
      cpuG[NDIM*2]=cpuRule9G[1];
      //{a3, 0, 0, 0,...., 0}
      cpuG[NDIM*3]=cpuRule9G[2];
      //{a4, 0, 0, 0,...., 0}
      cpuG[NDIM*4]=cpuRule9G[3];

      //{b, b, 0, 0,...., 0}
      cpuG[NDIM*5]=cpuRule9G[0];
      cpuG[NDIM*5+1]=cpuRule9G[0];


      //{y, d, 0, 0,...., 0}
      cpuG[NDIM*6]=cpuRule9G[0];
      cpuG[NDIM*6+1]=cpuRule9G[1];

      //{e, e, e, 0,...., 0}
      cpuG[NDIM*7]=cpuRule9G[0];
      cpuG[NDIM*7+1]=cpuRule9G[0];
      cpuG[NDIM*7+2]=cpuRule9G[0];
  
      //{l, l, l, ...., l}
      for(int dim = 0; dim < NDIM; ++dim ){
		cpuG[NDIM*8+dim] = cpuRule9G[4];
      }
     
      CPUScale = (T *)Host.AllocateMemory((void *)CPUScale, sizeof(T)*NSETS*NRULES);
      CPUNorm = (T *)Host.AllocateMemory((void *)CPUNorm, sizeof(T)*NSETS*NRULES);

      for(int idx = 0; idx < NSETS; ++idx){
		T *s_weight = &cRule9Wt[idx*NRULES];
		for( int r = 1; r < NRULES - 1; ++r ) {
			T scale = (s_weight[r] == 0) ? 100 :-s_weight[r + 1]/s_weight[r];
			T sum = 0;
			for( int x = 0; x < NSETS; ++x ){
				T *weight = &cRule9Wt[x*NRULES];
				sum += CPUGeneratorCount9[x]*fabs(weight[r + 1] + scale*weight[r]);
			}
			CPUScale[idx*NRULES+r]=scale;
			CPUNorm[idx*NRULES+r]=1/sum;
		}
      }
    }
	
    //@brief Template function to display GPU device array variables
    template <class K>
    void display(K *array, size_t size){
      K *tmp = (K *)malloc(sizeof(K) * size);
      cudaMemcpy(tmp, array, sizeof(K)*size, cudaMemcpyDeviceToHost);
      for(int i = 0 ; i < size; ++i){
		printf("%.20lf \n",(T)tmp[i]);
      }
    }


  public:
    ~QuadRule(){
      //Host.ReleaseMemory(CPUScale);
      //Host.ReleaseMemory(CPUNorm);
      //QuadDebug(Device.ReleaseMemory(GPUScale));
      //QuadDebug(Device.ReleaseMemory(GPUNorm));
    }
    
	void loadDeviceConstantMemory(int device = 0){
      Device.DeviceInit(device, VERBOSE);
     
      QuadDebug(Device.CopyHostToDeviceConstantMemory(gpuG, cpuG, sizeof(T) * NDIM *NSETS));      
      QuadDebug(Device.CopyHostToDeviceConstantMemory(cRuleWt, CPURuleWt, sizeof(T) * NRULES *NSETS));
      QuadDebug(Device.CopyHostToDeviceConstantMemory(cGeneratorCount, CPUGeneratorCount, sizeof(size_t)*NSETS));    
      QuadDebug(Device.CopyHostToDeviceConstantMemory(GPUScale, CPUScale, sizeof(T)* NSETS *NRULES));
      QuadDebug(Device.CopyHostToDeviceConstantMemory(GPUNorm, CPUNorm, sizeof(T) * NSETS *NRULES));
      QuadDebug(Device.CopyHostToDeviceConstantMemory(gpuGenPos, genPtr, sizeof(int)*PERMUTATIONS_POS_ARRAY_SIZE));
      QuadDebug(Device.CopyHostToDeviceConstantMemory(gpuGenPermVarCount, cpuGenPermVarCount, sizeof(int)*FEVAL));  
      QuadDebug(Device.CopyHostToDeviceConstantMemory(gpuGenPermGIndex, cpuGenPermGIndex, sizeof(int) *FEVAL));
      QuadDebug(Device.CopyHostToDeviceConstantMemory(gpuGenPermVarStart, cpuGenPermVarStart, sizeof(int) * (FEVAL+1)));
    }

	
    void Init(int ndim, size_t fEval, int key, int verbose){
      NDIM = ndim;
      KEY = key;
      VERBOSE = verbose;
      fEvalPerRegion = fEval;
      Rule9Generate();
      

      cpuGenPermVarCount = (int *)Host.AllocateMemory((void *)cpuGenPermVarCount, sizeof(int) * fEval);
      cpuGenPermVarStart = (int *)Host.AllocateMemory((void *)cpuGenPermVarStart, sizeof(int) * fEval);
      cpuGenPermGIndex   = (int *)Host.AllocateMemory((void *)cpuGenPermGIndex, sizeof(int) * (fEval+1));

      T *cpuGCopy = 0;
      cpuGCopy = (T *)Host.AllocateMemory((void *)cpuGCopy, sizeof(T) * NDIM *NSETS);
      for(int iter = 0; iter < NDIM*NSETS; ++iter){
		cpuGCopy[iter] = cpuG[iter];
      }
      
      /*size_t countPos = 0;
      for(int gIndex = 0; gIndex< NSETS; ++gIndex){
	countPos += cpuGenCount[gIndex]*indxCnt[gIndex];
	}*/

      genPtr = (int *)Host.AllocateMemory((void *)genPtr, sizeof(int) * PERMUTATIONS_POS_ARRAY_SIZE);
      int genPtrPosIndex = 0, permCnt = 0;


      cpuGenPermVarStart[0]=0;//To make sure length of first permutation is ZERO

      for(int gIndex = 0; gIndex < NSETS; ++gIndex){
		int n = cpuGenCount[gIndex], flag = 1;
		int num_permutation = 0;
		T *g = &cpuGCopy[NDIM*gIndex];
		while(num_permutation < n){
			num_permutation++;
			flag = 1;
		  int genPosCnt = 0;
		  cpuGenPermVarStart[permCnt]=genPtrPosIndex;
		  int isAccess[NDIM];
		  for(int dim = 0; dim < NDIM; ++dim){
			isAccess[dim]=0;
		  }
		for(int i = 0; i < indxCnt[gIndex]; ++i){
			//Find pos of cpuG[i]
			for(int dim = 0; dim < NDIM; ++dim ){	    
			  if(cpuG[NDIM*gIndex+i]==fabs(g[dim])&&!isAccess[dim]){
				++genPosCnt;
				isAccess[dim]=1;
				if(g[dim] < 0)
					genPtr[genPtrPosIndex++] = -(dim+1);
				else
					genPtr[genPtrPosIndex++] = dim+1;
				break;
			  }
			}
		}
		permCnt++;
		cpuGenPermVarCount[permCnt-1]=genPosCnt;
		cpuGenPermGIndex[permCnt-1]=gIndex;
		for(int dim = 0; (dim < NDIM)&&(flag==1); ) {
			g[dim] = -g[dim];
			if( g[dim++] < -0.0000000000000001){
				flag = 0;
				break;
			} 
		}
		
		for(int dim = 1; (dim < NDIM)&&(flag==1); ++dim ) {
			T gd = g[dim];
			if( g[dim - 1] > gd ) {
				size_t i, j = dim, ix = dim, dx = dim - 1;
				for( i = 0; i < --j; ++i ) {
					T tmp = g[i];
					g[i] = g[j];
					g[j] = tmp;
					if( tmp <= gd ) --dx;
					if( g[i] > gd ) ix = i;
				}
				if( g[dx] <= gd ) dx = ix;
					g[dim] = g[dx];
				g[dx] = gd;
				flag = 0;
				break;	  
	    }
		
	    if(flag == 0) break;
	  }
	}
	
  }
      cpuGenPermVarStart[permCnt]=genPtrPosIndex;      
      genPtrPosIndex = 0;
      loadDeviceConstantMemory();
      
    }

  };




}
