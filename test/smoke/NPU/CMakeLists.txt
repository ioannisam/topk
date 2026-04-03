cmake_minimum_required(VERSION 3.16)
project(npu_runtime_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(npu_runtime_smoke main.cpp)
set_target_properties(npu_runtime_smoke PROPERTIES OUTPUT_NAME smoke)
target_link_libraries(npu_runtime_smoke PRIVATE xrt_coreutil xrt_core)
