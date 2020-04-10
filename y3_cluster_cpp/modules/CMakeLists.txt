add_library(sigma_miscent_y1_scalarintegrand SHARED sigma_miscent_y1_scalarintegrand.cc)
target_link_libraries(sigma_miscent_y1_scalarintegrand PRIVATE models utils ${GSL_LIBRARIES})
target_include_directories(sigma_miscent_y1_scalarintegrand
                           PRIVATE
                           ${EXTERNALS_DIR}
                           ${CMAKE_SOURCE_DIR}
                           ${CUBACPP_DIR}
                           ${EIGEN3_INCLUDE_DIR}
                           ${GSL_INCLUDE_DIR})
