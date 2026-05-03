if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	cmake_minimum_required(VERSION 3.16)
	project(npu_topk LANGUAGES CXX)

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

add_executable(npu_topk)
target_sources(npu_topk PRIVATE
	src/main.cpp
	src/algorithm.cpp
	src/reporting.cpp
	src/runner.cpp
)
target_include_directories(npu_topk PRIVATE include ../common/include)
target_link_libraries(npu_topk PRIVATE common xrt_coreutil xrt_core uuid)
set_target_properties(npu_topk PROPERTIES OUTPUT_NAME topk)
target_compile_options(npu_topk PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)

if(DEBUG)
	target_compile_definitions(npu_topk PRIVATE DEBUG=1)
else()
	target_compile_definitions(npu_topk PRIVATE DEBUG=0)
endif()
