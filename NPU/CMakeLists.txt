find_package(XRT REQUIRED)

set(NPU_VENV_DIR "${CMAKE_SOURCE_DIR}/npu_venv")
add_custom_target(npu_hardware ALL
    COMMAND bash -c "source ${NPU_VENV_DIR}/bin/activate && make -C ${CMAKE_CURRENT_SOURCE_DIR} OUT_DIR=${CMAKE_CURRENT_BINARY_DIR} hardware"
    COMMENT "Building NPU Hardware (XCLBINs) via Peano/IRON..."
    VERBATIM
)

add_executable(npu_topk
    src/main.cpp
    src/bitonic.cpp
    src/map_reduce.cpp
    src/ground_truth.cpp
    src/reporting.cpp
    src/runner.cpp
)

set_target_properties(npu_topk PROPERTIES OUTPUT_NAME "topk")

target_include_directories(npu_topk PRIVATE include)
target_compile_options(npu_topk PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)
target_link_libraries(npu_topk PRIVATE common xrt_coreutil uuid)

add_dependencies(npu_topk npu_hardware)

add_executable(npu_roofline
    src/roofline_main.cpp
    src/roofline.cpp
)

set_target_properties(npu_roofline PROPERTIES OUTPUT_NAME "roofline")

target_include_directories(npu_roofline PRIVATE include src)
target_compile_options(npu_roofline PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)
target_link_libraries(npu_roofline PRIVATE common xrt_coreutil uuid)
