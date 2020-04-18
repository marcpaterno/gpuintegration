add_library(utils SHARED
            interp_1d.cc
            interp_2d.cc
            make_ifstream.cc
            str_to_doubles.cc)
target_link_libraries(utils PRIVATE ${GSL_LIBRARIES})
target_include_directories(utils PRIVATE
                           ${CMAKE_SOURCE_DIR}
                           ${CUBA_INCLUDE_DIR}
                           ${GSL_INCLUDE_DIR}
                           ${CUBACPP_DIR}
                           ${EIGEN3_INCLUDE_DIR})
