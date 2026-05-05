add_executable(gpu_topk)
target_sources(gpu_topk PRIVATE
	src/main.cpp
	src/reporting.cpp
	src/runner.cpp
	src/bitonic.cu
	src/map_reduce.cu
)
target_include_directories(gpu_topk PRIVATE include)
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
