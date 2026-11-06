cmake_minimum_required(VERSION 3.12)
project(go2_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

set(UNITREE_SDK2_DIR "" CACHE PATH "Path to a Unitree SDK2 installation")
set(UNITREE_DDS_DIR "/usr/local" CACHE PATH "Path to the Unitree CycloneDDS installation")
find_library(
  UNITREE_SDK2_LIBRARY
  NAMES unitree_sdk2
  HINTS ${UNITREE_SDK2_DIR} ${UNITREE_SDK2_DIR}/lib
)
if(NOT UNITREE_SDK2_LIBRARY)
  message(FATAL_ERROR "Unitree SDK2 library not found; set UNITREE_SDK2_DIR")
endif()

find_library(
  DDSC_LIBRARY
  NAMES ddsc
  HINTS ${UNITREE_DDS_DIR}/lib
  NO_DEFAULT_PATH
)
find_library(
  DDSCXX_LIBRARY
  NAMES ddscxx
  HINTS ${UNITREE_DDS_DIR}/lib
  NO_DEFAULT_PATH
)
if(NOT DDSC_LIBRARY OR NOT DDSCXX_LIBRARY)
  message(FATAL_ERROR "Unitree CycloneDDS libraries not found; set UNITREE_DDS_DIR")
endif()

# Ubuntu's spdlog package may use either its bundled fmt or an external libfmt.
find_library(FMT_LIBRARY NAMES fmt)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.22.0")
endif()

include_directories(
  /usr/include/eigen3
  ${UNITREE_DDS_DIR}/include/ddscxx
  ${UNITREE_DDS_DIR}/include/iceoryx/v2.0.2
  ${UNITREE_SDK2_DIR}/include
  include
  ${ONNXRUNTIME_DIR}/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  ${UNITREE_SDK2_LIBRARY} ${DDSC_LIBRARY} ${DDSCXX_LIBRARY} rt pthread
  libboost_program_options.a libyaml-cpp.a
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
)

if(FMT_LIBRARY)
  link_libraries(${FMT_LIBRARY})
endif()

set(CMAKE_BUILD_RPATH "${UNITREE_DDS_DIR}/lib;${ONNXRUNTIME_DIR}/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_ctrl main.cpp)
