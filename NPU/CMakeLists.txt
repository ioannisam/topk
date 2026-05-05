add_executable(npu_topk)
target_sources(npu_topk PRIVATE
	src/main.cpp
	src/algorithm.cpp
	src/reporting.cpp
	src/runner.cpp
)
target_include_directories(npu_topk PRIVATE include)
target_link_libraries(npu_topk PRIVATE common xrt_coreutil xrt_core uuid)
set_target_properties(npu_topk PROPERTIES OUTPUT_NAME topk)
target_compile_options(npu_topk PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)

if(DEBUG)
	target_compile_definitions(npu_topk PRIVATE DEBUG=1)
else()
	target_compile_definitions(npu_topk PRIVATE DEBUG=0)
endif()
