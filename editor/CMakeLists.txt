# Copyright 2025-2025 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(streami-editor LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(anari 0.15.0 REQUIRED)

add_executable(${PROJECT_NAME} main.cu)
target_link_libraries(${PROJECT_NAME} PRIVATE
    dvr_course-common anari::anari streami_library)
