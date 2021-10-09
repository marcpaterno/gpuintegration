cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Cuba_vegas VERSION 0.0.1 LANGUAGES CXX C)
enable_testing()

# These flags are used by everything in the project. Put anything that affects
# the ABI here -- e.g., the standard level.
SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -mtune=native" )

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# Turn on compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUBA REQUIRED)

include_directories(AFTER ${CMAKE_SOURCE_DIR} ${CUBACPP_DIR})
set(CMAKE_SHARED_MODULE_PREFIX "")    # We don't need the leading lib

add_executable(Genz Genz.cc)
target_include_directories(Genz PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBACPP_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz PRIVATE ${CUBA_LIBRARIES})
