#ifndef VERBOSE_UTILS_CUH
#define VERBOSE_UTILS_CUH

std::ofstream GetOutFileVar(std::string filename){
  std::ofstream myfile;
  myfile.open (filename.c_str());
  return myfile;
}

#endif
