# Add source files here
EXECUTABLE      := output 
# Cuda source files (compiled with cudacc)
CUFILES         := cudaCuhre.cu 

CCFLAGS         := -arch=sm_70 -lmpi -lgomp -Xcompiler -fopenmp -std=c++11 -lineinfo
INCLUDE  	:= -I /cm/shared/apps/openmpi/gcc/64/2.0.4/include/
LIBS     	:= -L /cm/shared/apps/openmpi/gcc/64/2.0.4/lib/ 


 

${EXECUTABLE}: ${CUFILES}
	nvcc -o ${EXECUTABLE}  ${CCFLAGS} ${INCLUDE} ${LIBS}  ${CUFILES} 
clean:
	-$(RM)	${EXECUTABLE}\
	*.o *.*log
run:
	./output

