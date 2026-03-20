cmake_minimum_required(VERSION 3.26.5)
project(streami LANGUAGES C CXX CUDA)

find_package(umesh REQUIRED)

add_subdirectory(submodules/rafi EXCLUDE_FROM_ALL)
add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL)

add_subdirectory(common)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE
    main.cu field/UMeshField.cu field/StructuredField.cu)
target_link_libraries(${PROJECT_NAME} PRIVATE
  dvr_course-common
  rafi
  umesh::umesh
  $<BUILD_INTERFACE:cuBQL_cuda_float3_static>)
target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

add_library(${PROJECT_NAME}_library)
target_sources(${PROJECT_NAME}_library PRIVATE
    main.cu field/UMeshField.cu field/StructuredField.cu)
target_link_libraries(${PROJECT_NAME}_library PUBLIC
  dvr_course-common
  rafi
  umesh::umesh
  $<BUILD_INTERFACE:cuBQL_cuda_float3_static>)
target_include_directories(${PROJECT_NAME}_library PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
add_subdirectory(editor)
add_subdirectory(tools/convert_icon)
