# ----------------------------------------------------------------------------
# CMakeLists.txt
#
#  Created on: 22 Jan 2021
#      Author: Kiwon Um
#        Mail: kiwon.um@telecom-paris.fr
#
# Description: CMake configuration
#
# Copyright 2021 Kiwon Um
#
# The copyright to the computer program(s) herein is the property of Kiwon Um,
# Telecom Paris, France. The program(s) may be used and/or copied only with
# the written permission of Kiwon Um or in accordance with the terms and
# conditions stipulated in the agreement/contract under which the program(s)
# have been supplied.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.0)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(tpSph)

add_executable(
  ${PROJECT_NAME}
  src/main.cpp "src/kernel.hpp" "src/CLSrc/gpuenv.cpp" "src/CLHeader/clcomputing.h")

  set(OpenCL_LIBRARY ../../../dep/opencl/lib/x86_64/OpenCL)
  FIND_PACKAGE(OpenCL REQUIRED)


INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS}) 


target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARY})
#target_link_libraries(${PROJECT_NAME} PRIVATE dep/opencl/lib/x86/OpenCL)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENGL_LIBRARIES})

add_subdirectory(dep/glfw)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

find_package(OpenMP)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX ${OPENGL_LIBRARIES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})

add_custom_command(TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR})