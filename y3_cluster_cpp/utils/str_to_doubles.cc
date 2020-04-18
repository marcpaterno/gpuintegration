#include "str_to_doubles.hh"

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

std::vector<double>
cosmosis::str_to_doubles(std::string const& txt)
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

