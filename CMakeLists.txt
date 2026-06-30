cmake_minimum_required(VERSION 3.18)
project(topk LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(DEBUG "Enable debug output by default at runtime" ON)

option(TOPK_BUILD_CPU "Build CPU backend" ON)
option(TOPK_BUILD_GPU "Build GPU backend" ON)
option(TOPK_BUILD_NPU "Build NPU backend" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (TOPK_BUILD_GPU)
	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

add_library(common STATIC
	common/src/config.cpp
	common/src/bitonic.cpp
	common/src/random.cpp
	common/src/reporting.cpp
)
target_include_directories(common PUBLIC common/include)
target_compile_features(common PUBLIC cxx_std_20)
target_compile_options(common PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)
if(DEBUG)
	target_compile_definitions(common PRIVATE DEBUG=1)
else()
	target_compile_definitions(common PRIVATE DEBUG=0)
endif()

if (TOPK_BUILD_CPU)
	add_subdirectory(CPU)
endif()
if (TOPK_BUILD_GPU)
	add_subdirectory(GPU)
endif()
if (TOPK_BUILD_NPU)
	add_subdirectory(NPU)
endif()
