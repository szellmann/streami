cmake_minimum_required(VERSION 3.26.5)
project(streami LANGUAGES C CXX CUDA)

find_package(umesh REQUIRED)

add_subdirectory(submodules/rafi EXCLUDE_FROM_ALL)
add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.cu field/UMeshField.cu)
target_link_libraries(${PROJECT_NAME} PRIVATE
  rafi
  umesh::umesh
  $<BUILD_INTERFACE:cuBQL_cuda_float3_static>)
target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

add_subdirectory(tools/convert_icon)
