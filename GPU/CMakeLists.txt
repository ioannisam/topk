add_executable(gpu_topk)
target_sources(gpu_topk PRIVATE
	src/main.cpp
	src/reporting.cpp
	src/runner.cpp
	src/bitonic.cu
	src/map_reduce.cu
	src/ground_truth.cu
)
target_include_directories(gpu_topk PRIVATE include)
target_link_libraries(gpu_topk PRIVATE common)
set_target_properties(gpu_topk PROPERTIES OUTPUT_NAME topk)
set_target_properties(gpu_topk PROPERTIES CUDA_ARCHITECTURES native)

target_compile_options(gpu_topk PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -DNDEBUG -Wall -Wextra>)
target_compile_options(gpu_topk PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

add_executable(gpu_roofline)
target_sources(gpu_roofline PRIVATE
	src/roofline_main.cpp
	src/roofline.cu
)
target_include_directories(gpu_roofline PRIVATE include)
target_link_libraries(gpu_roofline PRIVATE common)
set_target_properties(gpu_roofline PROPERTIES OUTPUT_NAME roofline)
set_target_properties(gpu_roofline PROPERTIES CUDA_ARCHITECTURES native)

target_compile_options(gpu_roofline PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -DNDEBUG -Wall -Wextra>)
target_compile_options(gpu_roofline PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
