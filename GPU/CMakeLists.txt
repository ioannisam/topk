cmake_minimum_required(VERSION 3.18)
project(cuda_topk LANGUAGES CXX CUDA)

option(DEBUG "Enable debug output by default at runtime" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_library(common STATIC
	../common/src/config.cpp
	../common/src/bitonic.cpp
	../common/src/random.cpp
	../common/src/reporting.cpp
)
target_include_directories(common PUBLIC
	../common/include
)

add_executable(gpu_topk)
target_sources(gpu_topk PRIVATE
	src/main.cpp
	src/reporting.cpp
	src/runner.cpp
	src/algorithm.cu
)
target_include_directories(gpu_topk PRIVATE include ../common/include)
target_link_libraries(gpu_topk PRIVATE common)
set_target_properties(gpu_topk PROPERTIES OUTPUT_NAME topk)
set_target_properties(gpu_topk PROPERTIES CUDA_ARCHITECTURES native)

target_compile_options(gpu_topk PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -DNDEBUG -Wall -Wextra>)
target_compile_options(gpu_topk PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

if(DEBUG)
	target_compile_definitions(gpu_topk PRIVATE DEBUG=1)
else()
	target_compile_definitions(gpu_topk PRIVATE DEBUG=0)
endif()
