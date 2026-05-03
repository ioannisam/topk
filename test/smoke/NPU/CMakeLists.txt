cmake_minimum_required(VERSION 3.20)
project(npu_topk_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/smoke.xclbin
    # Added --aie-generate-xclbin to force the packaging step
    COMMAND aiecc.py --aie-generate-cdo --aie-generate-xclbin --no-compile-host --xclbin-name=smoke.xclbin ${CMAKE_CURRENT_SOURCE_DIR}/smoke.mlir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/smoke.mlir
    COMMENT "Compiling MLIR kernel and packaging to xclbin..."
)

add_custom_target(npu_kernel ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/smoke.xclbin)

add_executable(smoke main.cpp)
add_dependencies(smoke npu_kernel)

target_link_libraries(smoke xrt_coreutil)
