cmake_minimum_required(VERSION 3.29.2)
project(streami LANGUAGES C CXX CUDA)
add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.cu)
