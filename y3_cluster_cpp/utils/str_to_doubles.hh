#ifndef _COSMOSIS_UTILS_STR_TO_DOUBLE_HH_
#define _COSMOSIS_UTILS_STR_TO_DOUBLE_HH_

#include <string>
#include <vector>

namespace cosmosis {
  // Read hexfloat doubles from a string.
  std::vector<double>
  str_to_doubles(std::string const& txt);
}

#endif