#ifndef VEGAS_UTILS_CUH
#define VEGAS_UTILS_CUH





__inline__ bool
PrecisionAchieved(double estimate,
                  double errorest,
                  double epsrel,
                  double epsabs)
{
  if (std::abs(errorest / estimate) <= epsrel || errorest <= epsabs) {
    return true;
  } else
    return false;
}

__inline__ int
GetStatus(double estimate,
          double errorest,
          int iteration,
          double epsrel,
          double epsabs)
{
  if (PrecisionAchieved(estimate, errorest, epsrel, epsabs) && iteration >= 5) {
    return 0;
  } else
    return 1;
}

__inline__
int GetChunkSize(const double ncall){
    double small = 1.e7;
    double large = 8.e9;
    
    if(ncall <= small)
        return 32;
    else if(ncall <= large)
        return 2048; 
    else
        return 4096;  
}

/*
  returns true if it can update params for an extended run, updates two params
  returns false if it has increased both params to their maximum allowed values
  this maximum is not configurable by the user, placeholder values are currently placed
 */

bool
AdjustParams(double& ncall, int& totalIters)
{
  if (ncall >= 8.e9 && totalIters >= 100){
    return false;
  }
  else if (ncall >= 8.e9) {
    totalIters += 10;
    return true;
  } else if (ncall >= 1.e9){
        ncall += 1.e9;
    return true;
  }
  else{
    ncall *= 10.;
    return true;  
  }
}

#endif
