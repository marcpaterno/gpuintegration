add_library(models SHARED
            lc_lt_t.cc
            int_lc_lt_des_t.cc)

target_link_libraries(models
                      PRIVATE
                      utils
                      ${GSL_LIBRARIES}
                      ${CUBA_LIBRARIES})

target_include_directories(models
                           PRIVATE
                           ${CMAKE_SOURCE_DIR}
                           ${CMAKE_SOURCE_DIR}/externals
                           ${CUBA_INCLUDE_DIR}
                           ${CUBACPP_DIR}
                           ${EIGEN3_INCLUDE_DIR}
                           ${GSL_INCLUDE_DIR})
