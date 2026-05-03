if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	cmake_minimum_required(VERSION 3.16)
	project(cpu_topk LANGUAGES CXX)

	option(DEBUG "Enable debug output by default at runtime" ON)

	set(CMAKE_CXX_STANDARD 20)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if (NOT TARGET common)
	add_library(common STATIC
		../common/src/config.cpp
		../common/src/bitonic.cpp
		../common/src/random.cpp
		../common/src/reporting.cpp
	)
	target_include_directories(common PUBLIC
		../common/include
	)
	target_compile_features(common PUBLIC cxx_std_20)
endif()

add_executable(cpu_topk)
target_sources(cpu_topk PRIVATE
	src/main.cpp
	src/bitonic.cpp
	src/map_reduce.cpp
	src/reporting.cpp
	src/runner.cpp
)
target_include_directories(cpu_topk PRIVATE include ../common/include)
target_link_libraries(cpu_topk PRIVATE common)
set_target_properties(cpu_topk PROPERTIES OUTPUT_NAME topk)
target_compile_options(cpu_topk PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)

if(DEBUG)
	target_compile_definitions(cpu_topk PRIVATE DEBUG=1)
else()
	target_compile_definitions(cpu_topk PRIVATE DEBUG=0)
endif()
