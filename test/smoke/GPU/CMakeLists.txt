cmake_minimum_required(VERSION 3.18)
project(cuda_topk_smoke LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_topk_smoke main.cu)
set_target_properties(cuda_topk_smoke PROPERTIES OUTPUT_NAME smoke)
set_target_properties(cuda_topk_smoke PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(cuda_topk_smoke PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>)
target_compile_options(cuda_topk_smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
