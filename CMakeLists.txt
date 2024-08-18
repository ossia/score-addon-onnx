cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT TARGET score_lib_base)
  include("${SCORE_SOURCE_DIR}/cmake/ScoreExternalAddon.cmake")
endif()

project(score_addon_onnx LANGUAGES CXX)

# Find onnxruntime
find_library(ONNXRUNTIME_LIBRARY onnxruntime)
find_path(ONNXRUNTIME_INCLUDE_DIR
  onnxruntime_cxx_api.h
  PATH_SUFFIXES
    onnxruntime
)

if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNXRUNTIME_LIBRARY not found")
endif()
if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIR not found")
endif()

add_library(onnxruntime INTERFACE)

target_link_libraries(onnxruntime
  INTERFACE
     "${ONNXRUNTIME_LIBRARY}"
)
target_include_directories(onnxruntime
  INTERFACE
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

# General initialization
score_common_setup()

if(0)
# Creation of the library
add_library(score_addon_onnx
    Onnx/Executor.hpp
    Onnx/Executor.cpp
    Onnx/Metadata.hpp
    Onnx/Process.hpp
    Onnx/Process.cpp
    Onnx/CommandFactory.hpp
    Onnx/CommandFactory.cpp


    score_addon_onnx.hpp
    score_addon_onnx.cpp
)

# Scan for commands which creates necessary files
# Second argument must be a CMake list of all headers containing commands

score_generate_command_list_file(
  score_addon_onnx
  "Onnx/CommandFactory.hpp"
)

# Target-specific options
setup_score_plugin(score_addon_onnx)

endif()
avnd_score_plugin_init(BASE_TARGET score_addon_onnx_models)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    OnnxModels/YOLO-blob.hpp
    OnnxModels/YOLO-blob.cpp
  TARGET
    yolo-blob
  MAIN_CLASS
    YOLOBlobDetector
  NAMESPACE
    OnnxModels
)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    OnnxModels/Resnet.hpp
    OnnxModels/Resnet.cpp
  TARGET
    resnet
  MAIN_CLASS
    ResnetDetector
  NAMESPACE
    OnnxModels
)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    OnnxModels/YOLO-pose.hpp
    OnnxModels/YOLO-pose.cpp
  TARGET
    yolo-pose
  MAIN_CLASS
    PoseDetector
  NAMESPACE
    OnnxModels::Yolo
)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    OnnxModels/BlazePose.hpp
    OnnxModels/BlazePose.cpp
  TARGET
    blazepose
  MAIN_CLASS
    BlazePoseDetector
  NAMESPACE
    OnnxModels
)

# target_link_libraries(score_addon_onnx_models PRIVATE score_addon_onnx)
avnd_score_plugin_finalize(BASE_TARGET score_addon_onnx_models PLUGIN_VERSION 1
                           PLUGIN_UUID "24a98f5b-700f-401c-ab89-8c1173bda73f")

# Link

target_link_libraries(score_addon_onnx_models
    PUBLIC
      onnxruntime
 )
