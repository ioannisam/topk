cmake_minimum_required(VERSION 3.16)
project(cpu_topk LANGUAGES CXX)

option(DEBUG "Enable debug output by default at runtime" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(cpu_topk)
target_sources(cpu_topk PRIVATE
	src/main.cpp
	src/config.cpp
	src/layers.cpp
	src/algorithm.cpp
	src/utils.cpp
	src/reporting.cpp
	src/runner.cpp
)
target_include_directories(cpu_topk PRIVATE include)
set_target_properties(cpu_topk PROPERTIES OUTPUT_NAME topk)
target_compile_options(cpu_topk PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)

if(DEBUG)
	target_compile_definitions(cpu_topk PRIVATE DEBUG=1)
else()
	target_compile_definitions(cpu_topk PRIVATE DEBUG=0)
endif()
