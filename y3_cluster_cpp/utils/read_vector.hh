#ifndef Y3_CLUSTER_CPP_READ_VECTOR_HH
#define Y3_CLUSTER_CPP_READ_VECTOR_HH

// Helper function to read a vector<double> from a file with the given filename.
// This requires that CosmoSIS be set up, so that the environment variable
// COSMOSIS_SRC_DIR is defined.

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template <class XFORM>
inline std::vector<double>
read_vector(const std::string filename, XFORM xform)
{
  std::string const fname =
    std::string(std::getenv("Y3_CLUSTER_CPP_DIR")) + "/data/" + filename;
  std::ifstream file(fname);
  if (!file) {
    std::string errmsg("Failed to open file: ");
    errmsg += fname;
    throw std::runtime_error(errmsg);
  }
  std::string line;
  std::vector<double> res;
  while (std::getline(file, line)) {
    // Skip lines that start with comment character
    if (line.find('#') == 0)
      continue;

    // Read all the numbers on this line
    std::istringstream linestream(line);
    double tmp;
    while (linestream >> tmp)
      res.push_back(xform(tmp));
  }
  return res;
}

inline std::vector<double>
read_vector(const std::string filename)
{
  return read_vector(filename, [](double x) { return x; });
}

#endif
