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
    OnnxModels/PoseDetector.hpp
    OnnxModels/PoseDetector.cpp
  TARGET
    pose-detector
  MAIN_CLASS
    PoseDetector
  NAMESPACE
    OnnxModels
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
    Onnx/helpers/Profile.hpp
    Onnx/helpers/SkeletonFormats.hpp
    Onnx/helpers/BlazeFace.hpp
    Onnx/helpers/FaceMesh.hpp
    Onnx/helpers/MediaPipeHands.hpp
    Onnx/helpers/RTMPose.hpp
    Onnx/helpers/ctx.h
    Onnx/helpers/ctx_overlay.h
    Onnx/helpers/CtxOverlay.hpp
)

# ctx.h is a single-header C library; it does not build as C++. Build the overlay
# shim as its own small C static library so it never touches the main target's
# C++ precompiled header. CTX_RASTERIZER=1 enables the in-memory framebuffer
# rasterizer (otherwise gated behind a windowing backend); -w silences the
# header's benign C-isms (const-casts, an implicit static-inline forward use).
add_library(score_onnx_ctx_overlay STATIC Onnx/helpers/ctx_overlay.c)
target_include_directories(score_onnx_ctx_overlay
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/Onnx/helpers)
target_compile_definitions(score_onnx_ctx_overlay PRIVATE CTX_RASTERIZER=1)
set_target_properties(score_onnx_ctx_overlay PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
  target_compile_options(score_onnx_ctx_overlay
    PRIVATE -w -Wno-implicit-function-declaration)
endif()

# ctx.h bundles its own copy of miniz (the mz_* zlib helpers + miniz_def_*
# allocators) and emits it unconditionally under CTX_IMPLEMENTATION, with no
# macro to suppress it (the core helpers aren't even behind MINIZ_NO_ZLIB_APIS).
# score_lib_base already links a miniz, so the two definitions clash at link
# (duplicate mz_adler32, mz_crc32, ...). The overlay rasterizer (CTX_VT=0,
# CTX_AUDIO=0, no image write) never calls miniz; it is only pulled in because
# onnx_overlay_* shares the translation unit. Rename the eight exported miniz
# symbols at preprocess time so ctx's copy is private to this TU: both its
# definitions and any internal references move together, and score's copy no
# longer collides. Pure -D, so it works identically on every toolchain in the
# CI matrix (clang/gcc/emscripten/FreeBSD -D, MSVC /D) -- unlike an objcopy or
# strip post-step. If a ctx update emits a new miniz symbol, the link error
# names it; add it here.
target_compile_definitions(score_onnx_ctx_overlay PRIVATE
  mz_adler32=ctxshim_mz_adler32
  mz_crc32=ctxshim_mz_crc32
  mz_free=ctxshim_mz_free
  mz_version=ctxshim_mz_version
  mz_error=ctxshim_mz_error
  miniz_def_alloc_func=ctxshim_miniz_def_alloc_func
  miniz_def_free_func=ctxshim_miniz_def_free_func
  miniz_def_realloc_func=ctxshim_miniz_def_realloc_func)

target_link_libraries(score_addon_onnx_models PRIVATE score_onnx_ctx_overlay)

# Image-processing hot path (per-pixel bilinear sample / normalize / warp /
# letterbox / Lanczos resize). Built as its own static lib forced to -O3 even in
# Debug builds, because these tight float loops run 5-15x slower at -Og/-O0 and
# would otherwise dominate the frame in a developer build. Header-only inlining
# would pull them back into the -Og host TU, so they live in ImageOps.cpp.
add_library(score_onnx_imageops STATIC Onnx/helpers/ImageOps.cpp)
target_include_directories(score_onnx_imageops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(score_onnx_imageops PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
  target_compile_options(score_onnx_imageops PRIVATE -O3)
else()
  target_compile_options(score_onnx_imageops PRIVATE /O2)
endif()
target_link_libraries(score_addon_onnx_models PRIVATE score_onnx_imageops)

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

# Per-frame pose pipeline profiler (readspec/warp/resize/tensorbuild/infer
# breakdown to stderr). Off by default; enable with -DSCORE_ONNX_PROFILE=ON.
option(SCORE_ONNX_PROFILE "Profile the ONNX pose pipeline per frame" OFF)
if(SCORE_ONNX_PROFILE)
  # Both targets need the define: the frame scope lives in the main target, the
  # warp/resize scopes live in the imageops lib. The profiler accumulator is an
  # inline thread_local in Profile.hpp, so it's a single shared instance across
  # both TUs.
  target_compile_definitions(score_addon_onnx_models PRIVATE ONNX_PROFILE)
  target_compile_definitions(score_onnx_imageops PRIVATE ONNX_PROFILE)
endif()
