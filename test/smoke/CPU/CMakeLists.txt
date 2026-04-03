cmake_minimum_required(VERSION 3.16)
project(cpu_topk_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(cpu_topk_smoke main.cpp)
set_target_properties(cpu_topk_smoke PROPERTIES OUTPUT_NAME smoke)
target_compile_options(cpu_topk_smoke PRIVATE -O3 -march=native -DNDEBUG)
