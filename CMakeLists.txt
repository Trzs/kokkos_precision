cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0028 NEW)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set the project name
project(Kokkos_precision)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
)
FetchContent_MakeAvailable(kokkos)

# add_subdirectory(kokkos)

# add the executable
add_executable(kokkos_precision main.cpp)

target_link_libraries(kokkos_precision Kokkos::kokkos)
