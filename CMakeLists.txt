cmake_minimum_required (VERSION 3.16)
project (groute LANGUAGES CXX CUDA)

############## DEPENDENCIES ##############
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# ------------------------------------------------------------------------------
# setting default cmake type to Release
# ------------------------------------------------------------------------------

set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
          STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
          "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

message("[groute] will build in type: ${CMAKE_BUILD_TYPE}")

# GFlags
add_subdirectory(deps/gflags)
include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)

# GTest
find_package(GTest QUIET)
if(NOT GTEST_FOUND)
  add_subdirectory(deps/googletest/googletest)
endif()

# Try to find METIS
find_package(METIS QUIET)
if (NOT METIS_FOUND)
  # Otherwise, try to find a local copy
  if (EXISTS "${PROJECT_SOURCE_DIR}/metis/")
    set(GKLIB_PATH ${PROJECT_SOURCE_DIR}/metis/GKlib CACHE PATH "path to GKlib" FORCE)
    add_subdirectory(metis)
    set(METIS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/metis/include)
    set(METIS_LIBRARIES metis)
    set(METIS_FOUND "true")
  else()
    message(WARNING "Compiling without METIS partitioning. To install METIS, run 'sudo apt-get install libmetis-dev' or manually download metis and extract to a subdirectory called 'metis'")

    set(METIS_INCLUDE_DIRS "")
    set(METIS_LIBRARIES "")

  endif()
endif()

if (METIS_FOUND)
  message("-- Found METIS: ${METIS_LIBRARIES}")
  add_definitions(-DHAVE_METIS)
endif()


find_package(CUDAToolkit)

set(CMAKE_CUDA_ARCHITECTURES "60;70") #60;61

############## BUILD ##############

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/deps/thrust)
set(EXTRA_LIBS gflags_static pthread ${METIS_LIBRARIES} CUDA::nvToolsExt CUDA::cuda_driver)
set(CMAKE_CUDA_STANDARD 14)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo -Xcompiler -ggdb -std=c++14 --extended-lambda")
else ()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG -Xcompiler -DNDEBUG -std=c++14 --extended-lambda")
endif ()


include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${METIS_INCLUDE_DIRS})
add_executable(cc src/utils/parser.cu src/utils/utils.cu
        samples/cc/cc_async.cu samples/cc/main.cu)
target_link_libraries(cc PRIVATE ${EXTRA_LIBS})

add_executable(bfs src/utils/parser.cu src/utils/utils.cu src/groute/graphs/csr_graph.cu
        samples/bfs/bfs_async.cu
#        samples/bfs/bfs_async_opt.cu
        samples/bfs/bfs_host.cu
        samples/bfs/main.cu)
target_link_libraries(bfs ${EXTRA_LIBS})

add_executable(sssp src/utils/parser.cu src/utils/utils.cu src/groute/graphs/csr_graph.cu
        samples/sssp/sssp_async.cu
#        samples/sssp/sssp_async_opt.cu
        samples/sssp/sssp_host.cu
        samples/sssp/main.cu)
target_link_libraries(sssp ${EXTRA_LIBS})

add_executable(pr src/utils/parser.cu src/utils/utils.cu src/groute/graphs/csr_graph.cu
        samples/pr/pr_async.cu
#        samples/pr/pr_async_opt.cu
        samples/pr/pr_host.cu
        samples/pr/main.cu)
target_link_libraries(pr ${EXTRA_LIBS} CUDA::cuda_driver)

add_executable(pbf samples/pbf/pbf_async.cu samples/pbf/main.cu)
target_link_libraries(pbf ${EXTRA_LIBS})


# Unit tests
enable_testing()

# GTest directory settings
if(NOT GTEST_FOUND)
  include_directories(
          ${gtest_SOURCE_DIR}/include
          ${gtest_SOURCE_DIR})
  link_directories(${gtest_BINARY_DIR}/src)
endif()

# Groute async component tests
add_executable(async-tests test/async-tests/main.cu
        test/async-tests/router_test.cu
        test/async-tests/circular_worklist_test.cu
        test/async-tests/worklist_test.cu)
target_link_libraries(async-tests gtest dl ${EXTRA_LIBS})

# Groute async component micro-benchmark tests
add_executable(mb-tests test/micro-benchmarks/main.cu
        test/micro-benchmarks/timed_kernel.cu test/micro-benchmarks/high_priority_copy.cu)
target_link_libraries(mb-tests gtest pthread ${EXTRA_LIBS})

add_executable(mc_dl test/micro-benchmarks/main.cu test/async-tests/multi_channel_distributed_worklist_test.cu)
target_link_libraries(mc_dl gtest pthread ${EXTRA_LIBS})

add_executable(nvlink test/micro-benchmarks/nvlink.cu)
target_link_libraries(nvlink ${EXTRA_LIBS})