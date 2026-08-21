add_executable(cpu_topk)
target_sources(cpu_topk PRIVATE
	src/main.cpp
	src/bitonic.cpp
	src/map_reduce.cpp
	src/ground_truth.cpp
	src/reporting.cpp
	src/runner.cpp
)
target_include_directories(cpu_topk PRIVATE include)
target_link_libraries(cpu_topk PRIVATE common)
set_target_properties(cpu_topk PROPERTIES OUTPUT_NAME topk)
target_compile_options(cpu_topk PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)

add_executable(cpu_roofline)
target_sources(cpu_roofline PRIVATE
	src/roofline.cpp
)
target_include_directories(cpu_roofline PRIVATE include)
target_link_libraries(cpu_roofline PRIVATE common)
set_target_properties(cpu_roofline PROPERTIES OUTPUT_NAME roofline)
target_compile_options(cpu_roofline PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)
