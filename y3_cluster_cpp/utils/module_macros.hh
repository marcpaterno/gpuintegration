#ifndef Y3_CLUSTER_CPP_MODULE_MACROS_HH
#define Y3_CLUSTER_CPP_MODULE_MACROS_HH
////////////////////////////////////////////////////////////////////////
//
// Defines the macros
// DEFINE_COSMOSIS_SCALAR_INTEGRATION_MODULE(<module_classname>) to be used
// in XXX_module.cc to declare a CosmoSIS module that integrates a scalar-
// valued integrand, and
// DEFINE_COSMOSIS_VECTOR_INTEGRATION_MODULE(<module_classname>) to be used in
// XXX_module.cc to declare a CosmoSIS module that integrates a vector-valued
// integrand.
//
// Note: Libraries that include these symbol definitions cannot be
// linked into a main program as other libraries are.  This is because
// the "one definition" rule would be violated.
//
////////////////////////////////////////////////////////////////////////

#include "CosmoSISScalarIntegrationModule.hh"
#include "CosmoSISVectorIntegrationModule.hh"

// Produce the injected functions
// TODO: Refactor this macros to remove the duplication of code.

#define DEFINE_COSMOSIS_SCALAR_INTEGRATION_MODULE(klass)                       \
  using module_type = y3_cluster::CosmoSISScalarIntegrationModule<klass>;      \
  extern "C" {                                                                 \
  void*                                                                        \
  setup(cosmosis::DataBlock* cfg)                                              \
  {                                                                            \
    return new module_type(*cfg);                                              \
  }                                                                            \
                                                                               \
  DATABLOCK_STATUS                                                             \
  execute(cosmosis::DataBlock* sample, void* module)                           \
  {                                                                            \
    auto mod = static_cast<module_type*>(module);                              \
    mod->execute(*sample);                                                     \
    return DBS_SUCCESS;                                                        \
  }                                                                            \
                                                                               \
  int                                                                          \
  cleanup(void* module)                                                        \
  {                                                                            \
    delete static_cast<module_type*>(module);                                  \
    return 0;                                                                  \
  }                                                                            \
  }

#define DEFINE_COSMOSIS_VECTOR_INTEGRATION_MODULE(klass)                       \
  using module_type = y3_cluster::CosmoSISVectorIntegrationModule<klass>;      \
  extern "C" {                                                                 \
  void*                                                                        \
  setup(cosmosis::DataBlock* cfg)                                              \
  {                                                                            \
    return new module_type(*cfg);                                              \
  }                                                                            \
                                                                               \
  DATABLOCK_STATUS                                                             \
  execute(cosmosis::DataBlock* sample, void* module)                           \
  {                                                                            \
    auto mod = static_cast<module_type*>(module);                              \
    mod->execute(*sample);                                                     \
    return DBS_SUCCESS;                                                        \
  }                                                                            \
                                                                               \
  int                                                                          \
  cleanup(void* module)                                                        \
  {                                                                            \
    delete static_cast<module_type*>(module);                                  \
    return 0;                                                                  \
  }                                                                            \
  }

#define DEFINE_COSMOSIS_ONED_INTEGRATION_MODULE(klass)                         \
  using module_type = y3_cluster::OneDIntegrationModule<klass>;                \
  extern "C" {                                                                 \
  void*                                                                        \
  setup(cosmosis::DataBlock* cfg)                                              \
  {                                                                            \
    return new module_type(*cfg);                                              \
  }                                                                            \
                                                                               \
  DATABLOCK_STATUS                                                             \
  execute(cosmosis::DataBlock* sample, void* module)                           \
  {                                                                            \
    auto mod = static_cast<module_type*>(module);                              \
    mod->execute(*sample);                                                     \
    return DBS_SUCCESS;                                                        \
  }                                                                            \
                                                                               \
  int                                                                          \
  cleanup(void* module)                                                        \
  {                                                                            \
    delete static_cast<module_type*>(module);                                  \
    return 0;                                                                  \
  }                                                                            \
  }

#endif
