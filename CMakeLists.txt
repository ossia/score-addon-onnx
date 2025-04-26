cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(EMSCRIPTEN)
  return()
endif()

if(NOT TARGET score_lib_base)
  include("${SCORE_SOURCE_DIR}/cmake/ScoreExternalAddon.cmake")
endif()

if(NOT TARGET score_plugin_avnd)
  return()
endif()

project(score_addon_onnx LANGUAGES CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntime.cmake")
if(NOT TARGET onnxruntime::onnxruntime)
  return()
endif()

add_subdirectory("${OSSIA_3RDPARTY_FOLDER}/pybind11" pybind11)
# General initialization
score_common_setup()

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
    YOLO7BlobDetector
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

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    PythonModels/StreamDiffusion.hpp
    PythonModels/StreamDiffusion.cpp
  TARGET
    streamdiffusion_img2img
  MAIN_CLASS
    StreamDiffusionImg2Img
  NAMESPACE
    PythonModels
)
avnd_score_plugin_finalize(BASE_TARGET score_addon_onnx_models PLUGIN_VERSION 1
                           PLUGIN_UUID "24a98f5b-700f-401c-ab89-8c1173bda73f")

# Link
target_link_libraries(score_addon_onnx_models
    PUBLIC
      $<COMPILE_ONLY:onnxruntime::onnxruntime>
      pybind11::pybind11 pybind11::embed
      /home/jcelerier/projets/oss/StreamDiffusion-uv/.venv/lib/python3.12/site-packages/torch/lib/libtorch.so
      /home/jcelerier/projets/oss/StreamDiffusion-uv/.venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
      /home/jcelerier/projets/oss/StreamDiffusion-uv/.venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so
      /home/jcelerier/projets/oss/StreamDiffusion-uv/.venv/lib/python3.12/site-packages/torch/lib/libtorch_python.so
      /home/jcelerier/projets/oss/StreamDiffusion-uv/.venv/lib/python3.12/site-packages/torch/lib/libc10.so
 )
target_include_directories(score_addon_onnx_models
    PRIVATE
    "/home/jcelerier/projets/oss//StreamDiffusion-uv/.venv/lib/python3.12/site-packages/torch/include/"
    "/home/jcelerier/projets/oss//StreamDiffusion-uv/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include"
)

