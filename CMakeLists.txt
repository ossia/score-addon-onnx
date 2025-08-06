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

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rapidlib.cmake")

if(NOT WIN32)
  # disabled due to https://github.com/microsoft/onnxruntime-extensions/issues/985
  set(ENABLE_TESTS 0)
  set(ENABLE_PYTHON 0)
  set(ENABLE_JAVA 0)
  set(ENABLE_MODEL_BENCHMARK 0)
  set(OCOS_ENABLE_VENDOR_IMAGE_CODECS 1)
  set(OCOS_ENABLE_VISION 0)
  set(OCOS_ENABLE_AUDIO 0)
  set(OCOS_ENABLE_RE2_REGEX 0)
  set(OCOS_BUILD_SHARED_LIB 0)
  set(OCOS_BUILD_PRESET token_api_only)
  if(LINUX OR WIN32)
    set(OCOS_USE_CUDA 0)
  endif()
  block()
    set(CMAKE_UNITY_BUILD 0)
    add_subdirectory(3rdparty/onnxruntime-extensions)
  endblock()
endif()

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
    OnnxModels/YOLO-segmentation.hpp
    OnnxModels/YOLO-segmentation.cpp
  TARGET
    yolo-segmentation
  MAIN_CLASS
    YOLO8Segmentation
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
    OnnxModels/ENet.hpp
    OnnxModels/ENet.cpp
  TARGET
    enet
  MAIN_CLASS
    EmotionNetDetector
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
    OnnxModels/TRTPose.hpp
    OnnxModels/TRTPose.cpp
  TARGET
    trtpose
  MAIN_CLASS
    TRTPoseDetector
  NAMESPACE
    OnnxModels::TRTPose
)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    OnnxModels/DepthAnything2.hpp
    OnnxModels/DepthAnything2.cpp
  TARGET
    depthanything2
  MAIN_CLASS
    DepthAnythingV2
  NAMESPACE
    OnnxModels
)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    RapidlibModels/Regressor.hpp
    RapidlibModels/Regressor.cpp
  TARGET
    regressor
  MAIN_CLASS
    Regressor
  NAMESPACE
    RapidlibModels
)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    RapidlibModels/Classifier.hpp
    RapidlibModels/Classifier.cpp
  TARGET
    classifier
  MAIN_CLASS
    Classifier
  NAMESPACE
    RapidlibModels
)


if(NOT WIN32)
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    OnnxModels/FastVLM.hpp
    OnnxModels/FastVLM.cpp
    Onnx/helpers/FastVLM.hpp
    Onnx/helpers/FastVLM.cpp
  TARGET
    fastvlm
  MAIN_CLASS
    FastVLMNode
  NAMESPACE
    OnnxModels
)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    OnnxModels/QwenLLM.hpp
    OnnxModels/QwenLLM.cpp
    Onnx/helpers/QwenLLM.hpp
    Onnx/helpers/QwenLLM.cpp
  TARGET
    qwenllm
  MAIN_CLASS
    QwenLLMNode
  NAMESPACE
    OnnxModels
)
endif()

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    OnnxModels/GenerativeImageGAN.hpp
    OnnxModels/GenerativeImageGAN.cpp
  TARGET
    generative-image-gan
  MAIN_CLASS
    GenerativeImageGAN
  NAMESPACE
    OnnxModels
)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_onnx_models
  SOURCES
    OnnxModels/ImageToImageGAN.hpp
    OnnxModels/ImageToImageGAN.cpp
  TARGET
    image-to-image-gan
  MAIN_CLASS
    ImageToImageGAN
  NAMESPACE
    OnnxModels
)

avnd_score_plugin_finalize(BASE_TARGET score_addon_onnx_models PLUGIN_VERSION 1
                           PLUGIN_UUID "24a98f5b-700f-401c-ab89-8c1173bda73f")

target_sources(score_addon_onnx_models
  PRIVATE
    Onnx/helpers/GAN.hpp
    Onnx/helpers/GAN.cpp
    Onnx/helpers/Utilities.hpp
    Onnx/helpers/Images.hpp
    Onnx/helpers/ModelSpec.hpp
    Onnx/helpers/OnnxBase.hpp
    Onnx/helpers/OnnxContext.hpp
)

# Link
target_link_libraries(score_addon_onnx_models
    PUBLIC
      $<COMPILE_ONLY:onnxruntime::onnxruntime>
      rapidlib
 )
if(NOT WIN32)
  target_link_libraries(score_addon_onnx_models
    PUBLIC
      ocos_operators
      onnxruntime_extensions
  )
endif()
