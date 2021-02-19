#include <chrono>
#include <iostream>
#include <vector>

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <chrono>
#include "vegas/vegas_mcubes_1DT.cuh"

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(){
    
    using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
    int functionID = 0;
    int NDIM = 6;
    double lbound[6] = {0., 0., 0., 0., 0., 0.};
    double rbound[6] = {10., 10., 10., 10., 10., 10.};
    double _ncall = 2.0E+09;
    int minIters = 10;
    int maxIters = 5; 
    int _skip = 0;
    int chunkSize = 2048;
    MilliSeconds dt;
    
    std::cout <<"ID,\t"
              <<"Integral,\t"
              <<"std,\t"
              <<"chisq,\t"
              <<"ncalls,\t"
              <<"chunkSize,\t"
              <<"iters,\t"
              <<"time\n";
    
    while(chunkSize >= 32){
        
        while(_ncall >= 1e3){
            cout<<"0,\t";
            auto t0 = std::chrono::high_resolution_clock::now();
            vegas_mcubes_1D(functionID, NDIM, lbound, rbound, _ncall, minIters, maxIters, _skip, chunkSize);
            dt = std::chrono::high_resolution_clock::now() - t0;
            cudaDeviceReset();
            
            std::cout<< _ncall <<",\t"
                     << chunkSize <<",\t"
                     << maxIters <<",\t"
                     << dt.count()
                     <<"\n";
            _ncall /= 4;
        }
        
        _ncall = 2.0e09;
        chunkSize = chunkSize/2;
    }
    
    return 0;
}