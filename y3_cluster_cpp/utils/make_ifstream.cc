#include "make_ifstream.hh"

#include <cstdlib>
#include <stdexcept>

std::ifstream
y3_cluster::make_ifstream(std::string const& path_fragment)
{
  char const* v = std::getenv("Y3_CLUSTER_CPP_DIR");
  if (v == nullptr)
    throw std::runtime_error(
      "environment variable Y3_CLUSTER_CPP_DIR not defined");
  std::string fullpath = std::string(v);
  fullpath += "/";
  fullpath += path_fragment;
  return std::ifstream(fullpath);
}
