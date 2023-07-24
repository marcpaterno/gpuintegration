#ifndef STR_TO_DOUBLE_HH
#define STR_TO_DOUBLE_HH

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

inline std::vector<double>
str_to_doubles(std::string const& txt)
{
  std::istringstream in(txt);
  std::string buffer;
  std::vector<double> vals;
  in >> std::hexfloat;
  while (in >> buffer) {
    vals.push_back(std::stod(buffer));
  };
  return vals;
}

#endif
