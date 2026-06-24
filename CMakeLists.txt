cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

if(EMSCRIPTEN)
  return()
endif()

# C is needed for the ctx_overlay.c helper static library (see below). A score
# build enables C via the top-level score project, but a standalone avnd-addon
# build has no such parent, so enabling it here is required -- otherwise CMake's
# generate step fails with "required internal CMake variable not set"
# (CMAKE_C_COMPILE_OBJECT / CMAKE_C_ARCHIVE_CREATE / CMAKE_C_ARCHIVE_FINISH).
project(score_addon_onnx LANGUAGES C CXX)

# Max/MSP externals link the static CRT (/MT) on Windows, as the official
# max-sdk-base requires. Set it for the whole addon -- before Avendish and the
# object library are created -- so every target shares one runtime; mixing /MT
# and /MD trips MSVC with LNK2038. (CMP0091 is NEW via cmake_minimum_required.)
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Use the Avendish the host (ossia score) provides; otherwise fetch it.
find_package(Avendish QUIET)
if(NOT Avendish_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Avendish
    GIT_REPOSITORY "https://github.com/celtera/avendish"
    GIT_TAG fa0b8836b46723a15d89e06426f225f2431574db)
  FetchContent_Populate(Avendish)
  list(APPEND CMAKE_PREFIX_PATH "${avendish_SOURCE_DIR}")
  find_package(Avendish REQUIRED)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntime.cmake")
if(NOT TARGET onnxruntime::onnxruntime)
  return()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rapidlib.cmake")

# onnxruntime-extensions (tokenizers) are only needed by the LLM/VLM objects
# (QwenLLM, FastVLM). Building it requires the onnxruntime-extensions submodule and
# its own onnxruntime headers, and its CMake fails to configure outside a full score
# tree (it errors "required internal CMake variable not set" right after "Tokenizer
# needed." on the standalone/portability lanes). The bulk of the addon (YOLO*, Pose*,
# Depth*, Resnet, ENet, GAN, rapidlib regressor/classifier) only needs plain
# onnxruntime, so gate the extensions subbuild -- and the two objects that depend on
# it -- behind SCORE_ONNX_ENABLE_EXTENSIONS. It defaults ON in a score build (which
# keeps CUDA + extensions) and OFF standalone, where only plain-onnxruntime objects
# are introspected/built. AVND_ADDON_SCORE is set by Avendish's AvendishAddon.cmake.
option(SCORE_ONNX_ENABLE_EXTENSIONS
  "Build onnxruntime-extensions (tokenizers) and the LLM/VLM objects that need them"
  ${AVND_ADDON_SCORE})

# onnxruntime-extensions does not build on Windows here (issue #985); never enable it
# there even in a score build.
if(WIN32)
  set(SCORE_ONNX_ENABLE_EXTENSIONS 0)
endif()

if(SCORE_ONNX_ENABLE_EXTENSIONS)
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
  set(ONNXRUNTIME_PKG_DIR "${onnxruntime_SOURCE_DIR}")
  block()
    set(CMAKE_UNITY_BUILD 0)
    add_subdirectory(3rdparty/onnxruntime-extensions)
  endblock()
endif()

# General initialization (only available in a score build; standalone has no such command)
if(COMMAND score_common_setup)
  score_common_setup()
endif()

avnd_addon_init(NAME score_addon_onnx_models)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME yolo-blob
  CLASS YOLO7BlobDetector
  NAMESPACE OnnxModels
  SOURCES
    OnnxModels/YOLO-blob.hpp
    OnnxModels/YOLO-blob.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME yolo-segmentation
  CLASS YOLO8Segmentation
  NAMESPACE OnnxModels
  SOURCES
    OnnxModels/YOLO-segmentation.hpp
    OnnxModels/YOLO-segmentation.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME resnet
  CLASS ResnetDetector
  NAMESPACE OnnxModels
  SOURCES
    OnnxModels/Resnet.hpp
    OnnxModels/Resnet.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME enet
  CLASS EmotionNetDetector
  NAMESPACE OnnxModels
  SOURCES
    OnnxModels/ENet.hpp
    OnnxModels/ENet.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME yolo-pose
  CLASS PoseDetector
  NAMESPACE OnnxModels::Yolo
  SOURCES
    OnnxModels/YOLO-pose.hpp
    OnnxModels/YOLO-pose.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME blazepose
  CLASS BlazePoseDetector
  NAMESPACE OnnxModels
  SOURCES
    OnnxModels/BlazePose.hpp
    OnnxModels/BlazePose.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME trtpose
  CLASS TRTPoseDetector
  NAMESPACE OnnxModels::TRTPose
  SOURCES
    OnnxModels/TRTPose.hpp
    OnnxModels/TRTPose.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME pose-detector
  CLASS PoseDetector
  NAMESPACE OnnxModels
  SOURCES
    OnnxModels/PoseDetector.hpp
    OnnxModels/PoseDetector_internal.hpp
    OnnxModels/PoseDetector.cpp
    OnnxModels/PoseDetector_draw.cpp
    OnnxModels/PoseDetector_detect.cpp
    OnnxModels/PoseDetector_landmark.cpp
    OnnxModels/PoseDetector_tracking.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME depthanything2
  CLASS DepthAnythingV2
  NAMESPACE OnnxModels
  SOURCES
    OnnxModels/DepthAnything2.hpp
    OnnxModels/DepthAnything2.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME regressor
  CLASS Regressor
  NAMESPACE RapidlibModels
  SOURCES
    RapidlibModels/Regressor.hpp
    RapidlibModels/Regressor.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME classifier
  CLASS Classifier
  NAMESPACE RapidlibModels
  SOURCES
    RapidlibModels/Classifier.hpp
    RapidlibModels/Classifier.cpp)


# FastVLM and QwenLLM are the only objects that #include the onnxruntime-extensions
# tokenizer headers (Onnx/helpers/TokenIO.hpp etc.), so they can only be built when the
# extensions subbuild above ran. Standalone/portability builds skip both the subbuild
# and these objects; the remaining objects compile against plain onnxruntime.
if(SCORE_ONNX_ENABLE_EXTENSIONS)
avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME fastvlm
  CLASS FastVLMNode
  NAMESPACE OnnxModels
  SOURCES
    OnnxModels/FastVLM.hpp
    OnnxModels/FastVLM.cpp
    Onnx/helpers/FastVLM.hpp
    Onnx/helpers/FastVLM.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME qwenllm
  CLASS QwenLLMNode
  NAMESPACE OnnxModels
  SOURCES
    OnnxModels/QwenLLM.hpp
    OnnxModels/QwenLLM.cpp
    Onnx/helpers/QwenLLM.hpp
    Onnx/helpers/QwenLLM.cpp)
endif()

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME generative-image-gan
  CLASS GenerativeImageGAN
  NAMESPACE OnnxModels
  SOURCES
    OnnxModels/GenerativeImageGAN.hpp
    OnnxModels/GenerativeImageGAN.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME image-to-image-gan
  CLASS ImageToImageGAN
  NAMESPACE OnnxModels
  SOURCES
    OnnxModels/ImageToImageGAN.hpp
    OnnxModels/ImageToImageGAN.cpp)

avnd_addon_finalize(NAME score_addon_onnx_models UUID 24a98f5b-700f-401c-ab89-8c1173bda73f VERSION 1)

target_sources(score_addon_onnx_models
  PRIVATE
    Onnx/helpers/GAN.hpp
    Onnx/helpers/GAN.cpp
    Onnx/helpers/Utilities.hpp
    Onnx/helpers/Images.hpp
    Onnx/helpers/ModelSpec.hpp
    Onnx/helpers/OnnxBase.hpp
    Onnx/helpers/OnnxContext.hpp
    Onnx/helpers/compat/buffer_pool.hpp
    Onnx/helpers/compat/dylib_loader.hpp
    Onnx/helpers/compat/safe_math.hpp
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
#
# The source is referenced through a variable on purpose: score's JIT add-on
# loader gathers compilable sources by *textually* scanning add_library() blocks
# and #include-ing every file it finds into one C++ unity TU. A bare
# "ctx_overlay.c" there would be pulled into that C++ TU and fail to compile (it
# is C-only), and likewise the -O3 ImageOps/TensorToTexture units below would be
# duplicated. Hiding the filenames behind ${...} keeps the real CMake build
# identical while making these helper libraries invisible to the JIT scanner.
set(SCORE_ONNX_CTX_OVERLAY_SRC Onnx/helpers/ctx_overlay.c)
add_library(score_onnx_ctx_overlay STATIC ${SCORE_ONNX_CTX_OVERLAY_SRC})
target_include_directories(score_onnx_ctx_overlay
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/Onnx/helpers)
# ctx.h is a single-header monolith (vector raster + terminal emulator + audio +
# event loop + backends + PNG/zip). We only rasterize a skeleton overlay into a
# framebuffer, so preemptively compile out everything that isn't the rasterizer
# or the (text-drawing) font engine. This is how ctx is meant to be slimmed --
# via CTX_* macros, not by hand-editing the vendored header -- and it strips all
# the POSIX baggage (events/threads/pty/sockets pull <unistd.h>, <sys/select.h>,
# <sys/ioctl.h>, <termios.h>, <pthread.h>, <ucontext.h> ...). Verified to leave
# every rasterizer fast path untouched (O2/O3, compositing, gradient/glyph
# caches, LUTs, AA, fast-rect) -- no perf regression. What ctx still references
# from POSIX in always-compiled glue is tiny: gettimeofday() (clock init) and
# write()/usleep() (never-executed terminal paths). mingw supplies them (its
# Windows builds pass); MSVC and the bare-clang Windows SDK do not, so feed those
# a small self-contained compat shim. (CTX_EVENTS=0 also relies on the
# keymap_get_shifted guard added in ctx.h.)
target_compile_definitions(score_onnx_ctx_overlay PRIVATE
  CTX_RASTERIZER=1
  CTX_EVENTS=0 CTX_THREADS=0 CTX_PTY=0
  CTX_PARSER=0 CTX_FORMATTER=0
  CTX_AUDIO=0 CTX_VT=0
  CTX_SDL=0 CTX_FB=0 CTX_KMS=0 CTX_HEADLESS=0
  CTX_BABL=0 CTX_HARFBUZZ=0 CTX_CURL=0
  CTX_SOCKETS=0 CTX_NET=0
  CTX_TERMINAL_EVENTS=0 CTX_RAW_KB_EVENTS=0
  CTX_PDF=0 CTX_SCREENSHOT=0
  CTX_GET_CONTENTS=0 CTX_LOAD_FILE=0 CTX_FONTS_FROM_FILE=0)
if(WIN32 AND NOT MINGW)
  target_include_directories(score_onnx_ctx_overlay BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Onnx/helpers/msvc-compat)
  # MSVC's <math.h> only defines M_PI etc. when this is set before it is included.
  target_compile_definitions(score_onnx_ctx_overlay PRIVATE _USE_MATH_DEFINES)
endif()
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

# In a score build the base target is the final plugin and links these helpers
# directly (PRIVATE). Standalone, each object is its own static lib that links the
# base PUBLIC, and the per-object back-end executables link the object PUBLIC -- so the
# helper libs must propagate through the base (PUBLIC) for those executables to resolve
# the ImageOps / ctx-overlay symbols the objects reference.
if(AVND_ADDON_SCORE)
  set(SCORE_ONNX_HELPER_LINK PRIVATE)
else()
  set(SCORE_ONNX_HELPER_LINK PUBLIC)
endif()
target_link_libraries(score_addon_onnx_models ${SCORE_ONNX_HELPER_LINK} score_onnx_ctx_overlay)

# Image-processing hot path (per-pixel bilinear sample / normalize / warp /
# letterbox / Lanczos resize). Built as its own static lib forced to -O3 even in
# Debug builds, because these tight float loops run 5-15x slower at -Og/-O0 and
# would otherwise dominate the frame in a developer build. Header-only inlining
# would pull them back into the -Og host TU, so they live in ImageOps.cpp.
# Referenced via a variable so score's JIT add-on scanner (which textually
# harvests add_library() sources into a single C++ unity TU) does not also
# #include these -O3 units -- they are already compiled here as their own lib.
set(SCORE_ONNX_IMAGEOPS_SRC
    Onnx/helpers/ImageOps.cpp
    Onnx/helpers/TensorToTexture.cpp)
add_library(score_onnx_imageops STATIC ${SCORE_ONNX_IMAGEOPS_SRC})
target_include_directories(score_onnx_imageops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(score_onnx_imageops PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
  target_compile_options(score_onnx_imageops PRIVATE -O3)
else()
  target_compile_options(score_onnx_imageops PRIVATE /O2)
endif()
target_link_libraries(score_addon_onnx_models ${SCORE_ONNX_HELPER_LINK} score_onnx_imageops)

# Link. In a score build the host links the onnxruntime shared lib itself (and
# installs it, see cmake/onnxruntime.cmake), so the plugin only needs its headers
# (COMPILE_ONLY) -- avoids a second copy on the link line. Standalone, each back-end
# executable must actually link libonnxruntime, so attach it as a normal PUBLIC dep
# that propagates through the base to the per-object back-ends.
if(AVND_ADDON_SCORE)
  target_link_libraries(score_addon_onnx_models
      PUBLIC
        $<COMPILE_ONLY:onnxruntime::onnxruntime>
        rapidlib
   )
else()
  target_link_libraries(score_addon_onnx_models
      PUBLIC
        onnxruntime::onnxruntime
        rapidlib
   )
endif()
if(SCORE_ONNX_ENABLE_EXTENSIONS)
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
