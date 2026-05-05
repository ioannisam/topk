add_executable(cpu_topk_ground_truth)
target_sources(cpu_topk_ground_truth PRIVATE
    src/main.cpp
    src/runner.cpp
)
target_include_directories(cpu_topk_ground_truth PRIVATE
    include
    ../../CPU/include
)
target_link_libraries(cpu_topk_ground_truth PRIVATE common)
set_target_properties(cpu_topk_ground_truth PROPERTIES OUTPUT_NAME topk)
target_compile_options(cpu_topk_ground_truth PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)

if(DEBUG)
    target_compile_definitions(cpu_topk_ground_truth PRIVATE DEBUG=1)
else()
    target_compile_definitions(cpu_topk_ground_truth PRIVATE DEBUG=0)
endif()
