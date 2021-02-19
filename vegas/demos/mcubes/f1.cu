#include <chrono>
#include <iostream>
#include <vector>

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include "vegas/vegas_mcubesT.cuh"
#include <chrono>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(){
    
    using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
    int functionID = 1;
    constexpr int NDIM = 9;
    double lbound[NDIM] = {-1., -1., -1., -1., -1., -1., -1., -1., -1.};
    double rbound[NDIM] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
   
    double _ncall = 1.0E+09;
    int minIters = 15;
    int maxIters = 10; 
    int _skip = 1;
    int chunkSize = 32;
     MilliSeconds dt;
    std::cout <<"ID,\t"
              <<"Integral,\t"
              <<"std,\t"
              <<"chisq,\t"
              <<"ncalls,\t"
              <<"chunkSize,\t"
              <<"iters,\t"
              <<"time\n";
    
        
 
    cout<<"1,\t";
    auto t0 = std::chrono::high_resolution_clock::now();
    vegas_mcubes(functionID, NDIM, lbound, rbound, _ncall, minIters, maxIters, _skip, chunkSize);
    dt = std::chrono::high_resolution_clock::now() - t0;
    cudaDeviceReset();
    std::cout<< _ncall <<",\t"<< chunkSize <<",\t"<< maxIters <<",\t"<< dt.count()<<"\n";
    return 0;
}