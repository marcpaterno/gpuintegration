#ifndef COSMOSIS_DATABLOCK_HH
#define COSMOSIS_DATABLOCK_HH

// This is a drop-in replacement for the portion of cosmosis::DataBlock
// used in this code.

#include "utils/datablock_status.h"
#include "utils/datablock_types.h"
#include "utils/ndarray.hh"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cosmosis {
  class DataBlock {
  public:
    double
    get_double(char const* name)
    {
      return doubles_[name];
    }

    std::vector<double>
    get_vector(char const* name)
    {
      return vectors_[name];
    }

    cosmosis::ndarray<double>
    get_ndarray(char const* name)
    {
      auto i = ndarrays_.find(name);
      if (i == ndarrays_.end())
        throw std::runtime_error("Missing name in datablock");
      return i->second;
    }

    template <class T>
    T view(std::string const&, char const* name);

  private:
    std::unordered_map<std::string, double> doubles_;
    std::unordered_map<std::string, std::vector<double>> vectors_;
    std::unordered_map<std::string, cosmosis::ndarray<double>> ndarrays_;
  };
}

template <>
inline std::vector<double>
cosmosis::DataBlock::view<std::vector<double>>(std::string const&,
                                               char const* name)
{
  return get_vector(name);
}

#endif