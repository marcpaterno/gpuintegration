# - Try to find Cuba headers and libraries.
#
# Usage of this module as follows:
#
#     find_package(CUBA)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  CUBA_DIR  Set this variable to the root installation of
#            CUBA if the module has problems finding
#            the proper installation path.
#
# Variables defined by this module:
#
#  CUBA_FOUND              System has Cuba libs/headers
#  CUBA_LIBRARIES          The Cuba libraries
#  CUBA_INCLUDE_DIR        The location of Cuba headers

find_path(CUBA_DIR
    NAMES cuba.h
)

find_library(CUBA_LIBRARIES
    NAMES cuba
    HINTS ${CUBA_DIR}/lib
)

find_path(CUBA_INCLUDE_DIR
    NAMES cuba.h
    HINTS ${CUBA_DIR}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUBA DEFAULT_MSG
    CUBA_LIBRARIES
    CUBA_INCLUDE_DIR
)

mark_as_advanced(
    CUBA_DIR
    CUBA_LIBRARIES
    CUBA_INCLUDE_DIR
)

