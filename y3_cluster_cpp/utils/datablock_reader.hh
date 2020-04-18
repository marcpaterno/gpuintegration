#ifndef _Y3_CLUSTER_CPP_DATABLOCK_READER_HH
#define _Y3_CLUSTER_CPP_DATABLOCK_READER_HH

#include "utils/datablock.hh"
#include "utils/ndarray.hh"
#include <vector>

inline std::vector<double>
get_vector_double(cosmosis::DataBlock& db, const char*, const char* name)
{
  return db.get_vector(name);
}

inline std::vector<double>
get_vector_double(cosmosis::DataBlock& db,
                  std::string const&,
                  std::string const& name)
{
  return get_vector_double(db, nullptr, name.c_str());
}

// Note: This function is deprecated. Prefer to use the DataBlock member
// template
//    T DataBlock::view<T>(section, name)
// to retrieve a parameter of type T from the given section, and with the given
// name.
template <typename T>
T get_datablock(cosmosis::DataBlock& db,
                const char* section,
                const char* value);

template <>
inline
double
get_datablock<double>(cosmosis::DataBlock& db, const char*, const char* val)
{
  return db.get_double(val);
}

template <>
inline
std::vector<double>
get_datablock<std::vector<double>>(cosmosis::DataBlock& db,
                                   const char*,
                                   const char* val)
{
  return db.get_vector(val);
}

template <>
inline
cosmosis::ndarray<double>
get_datablock<cosmosis::ndarray<double>>(cosmosis::DataBlock& db,
                                         const char*,
                                         const char* val)
{
  return db.get_ndarray(val);
}

#endif // _Y3_CLUSTER_CPP_DATABLOCK_READER_HH
