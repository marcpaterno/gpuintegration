#ifndef Y3_CLUSTER_CPP_UTILS_MAKE_IFSTREAM_HPP
#define Y3_CLUSTER_CPP_UTILS_MAKE_IFSTREAM_HPP

#include <fstream>

namespace y3_cluster {

  // Given the path fragement from the top y3_cluster_cpp directory,
  // find and open the file. If the file is not found, a default-constructed
  // ifstream is returned. If the required environment variable
  // Y3_CLUSTER_CPP_DIR is not set, a std::runtime_error is thrown.
  std::ifstream make_ifstream(std::string const& path_fragment);
}

#endif
