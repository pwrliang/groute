include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message("-- Cloning External Project: Thrust")
get_filename_component(FC_BASE "../cmake"
        REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
        thrust
        GIT_REPOSITORY https://github.com/thrust/thrust.git
        GIT_TAG        1.12.0
)

FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
    FetchContent_Populate(
            thrust
    )
endif()
set(THRUST_INCLUDE_DIR "${thrust_SOURCE_DIR}")
set(CUB_INCLUDE_DIR "${thrust_SOURCE_DIR}/cub")