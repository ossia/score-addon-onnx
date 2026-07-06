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
#
# Only for the standalone build: inside a score build the Max backend is not
# emitted, and score links the dynamic CRT (/MD). Forcing /MT here would make
# score_addon_onnx_models.lib mismatch the rest of score (LNK2038:
# MT_StaticRelease vs MD_DynamicRelease), so defer to score's runtime instead.
if(MSVC AND NOT TARGET score_lib_base)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Windows-wide preprocessor defines for the whole addon, before any target, so
# every object and helper TU is covered (a score build sets these globally; a
# standalone avnd-addon build does not). No effect off Windows.
#  - NOMINMAX: <windows.h> (pulled in by onnxruntime, ctx.h and our dylib_loader)
#    otherwise defines min()/max() as macros, mangling every std::min / std::max /
#    std::numeric_limits<T>::max() in the onnx C++ objects ("C2589: '(' illegal
#    token on right side of '::'").
#  - _USE_MATH_DEFINES: MSVC's <cmath>/<math.h> only define M_PI etc. when this is
#    set before they are included; ROI.hpp / Trt.hpp / PoseDetector_tracking.cpp
#    use M_PI ("C2065: 'M_PI': undeclared identifier"). (ctx_overlay sets it on its
#    own C target too; this covers the C++ objects.)
if(WIN32)
  add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)
endif()

set(AVND_USES_EXCEPTIONS ON CACHE BOOL "" FORCE)

# Use the Avendish the host (ossia score) provides; otherwise fetch it.
find_package(Avendish QUIET)
if(NOT Avendish_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Avendish
    GIT_REPOSITORY "https://github.com/celtera/avendish"
    # avendish master: provides avnd_addon_package + avnd_create_{max_addon,
    # touchdesigner,godot}_package used below to bundle libonnxruntime into the
    # standalone packages.
    GIT_TAG 6660f3d25924c304e8bf42cae76ab766d23f7dda)
  FetchContent_Populate(Avendish)
  list(APPEND CMAKE_PREFIX_PATH "${avendish_SOURCE_DIR}")
  find_package(Avendish REQUIRED)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntime.cmake")
if(NOT TARGET onnxruntime::onnxruntime)
  return()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rapidlib.cmake")

# onnxruntime-extensions provides the BPE tokenizer the LLM/VLM objects (QwenLLM,
# FastVLM) need to turn text into token ids; nothing else in the addon touches it. The
# token_api_only preset wired below points the subbuild at the prebuilt onnxruntime
# package via ONNXRUNTIME_PKG_DIR (set from onnxruntime_SOURCE_DIR by
# cmake/onnxruntime.cmake), so it configures and builds standalone exactly as it does
# inside a score tree -- verified to produce the fastvlm/qwenllm objects on the
# standalone lanes. Default it ON so those two objects build everywhere the tokenizer
# compiles; the rest of the addon (YOLO*, Pose*, Depth*, Resnet, ENet, GAN, rapidlib
# regressor/classifier) only needs plain onnxruntime and is unaffected by this switch.
#
# Windows was previously force-disabled: the upstream extensions did not compile with
# clang + libc++ (the ossia Windows toolchain), as base/file_sys.h built a std::ifstream
# from a std::wstring -- an MSVC-STL-only overload (issue #985, now #1071). The submodule
# now tracks the ossia fork carrying the one-line libc++ fix (filed upstream as
# microsoft/onnxruntime-extensions#1084), so the Windows guard is gone and the tokenizer
# objects build on every non-wasm back-end.
option(SCORE_ONNX_ENABLE_EXTENSIONS
  "Build onnxruntime-extensions (tokenizers) and the LLM/VLM objects that need them"
  ON)

if(SCORE_ONNX_ENABLE_EXTENSIONS)
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

# C++ standard. Avendish sets cxx_std_23 on its per-object targets, but its
# standalone base target (avnd_addon_init) gets none -- so it falls to the
# compiler default (C++17 on the CI gcc-13/14 / Apple Clang lanes). The base
# compiles the shared helper TUs (GAN.cpp -> Images.hpp/Utilities.hpp), which use
# std::span, abbreviated-template auto params and range-for-with-init -- all C++20+
# -- and failed with "'span' is not a member of 'std'" (<span> is empty pre-C++20).
# Require it explicitly here, PUBLIC so the objects + back-ends stay consistent. A
# score build is unaffected (it already builds at >= C++23).
target_compile_features(score_addon_onnx_models PUBLIC cxx_std_23)

# Boost.Container: the decoupled helpers (Images/OnnxBase/Yolo/...) store tensor
# input in boost::container::vector for its default_init no-zero-fill resize.
# Avendish already pulls Boost in (find_dependency), but only links Boost::boost
# onto its per-object targets; the base target compiles the shared helper TUs
# (GAN.cpp -> Images.hpp -> <boost/container/vector.hpp>) directly, so it needs
# Boost on its own include path. A score build supplies Boost globally; standalone
# does not. PUBLIC so it reaches the objects + back-ends, mirroring Eigen below.
#
# Do NOT call find_package(Boost) here: the standalone CI forces
# Boost_NO_BOOST_CMAKE=ON, which routes find_package(Boost) through the removed
# FindBoost module (CMP0167) and fails to locate Boost >= 1.89 even with BOOST_ROOT
# set ("Could NOT find Boost: missing Boost_INCLUDE_DIR"). Reuse whatever Avendish /
# the host already resolved instead.
if(TARGET Boost::boost)
  target_link_libraries(score_addon_onnx_models PUBLIC Boost::boost)
elseif(TARGET Boost::headers)
  target_link_libraries(score_addon_onnx_models PUBLIC Boost::headers)
elseif(Boost_INCLUDE_DIRS)
  target_include_directories(score_addon_onnx_models PUBLIC ${Boost_INCLUDE_DIRS})
elseif(DEFINED BOOST_ROOT)
  target_include_directories(score_addon_onnx_models PUBLIC "${BOOST_ROOT}/include")
endif()

# Eigen: PoseTracker.hpp (the pose_detector object) includes <Eigen/Dense>. A
# score build provides it globally; standalone resolves it here (header-only).
# Attached PUBLIC so it propagates through the base to the objects + back-ends.
if(NOT TARGET Eigen3::Eigen)
  find_package(Eigen3 QUIET)
endif()
if(TARGET Eigen3::Eigen)
  target_link_libraries(score_addon_onnx_models PUBLIC Eigen3::Eigen)
else()
  include(FetchContent)
  FetchContent_Declare(
    Eigen3onnx
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE)
  FetchContent_GetProperties(Eigen3onnx)
  if(NOT eigen3onnx_POPULATED)
    FetchContent_Populate(Eigen3onnx)
  endif()
  target_include_directories(score_addon_onnx_models PUBLIC "${eigen3onnx_SOURCE_DIR}")
endif()

# Explicit per-object back-ends (mirroring score-addon-cv). The image objects take
# a texture input, which has no meaning in Pd -- avendish's Pd binding then fails to
# compile ("fixed_texture_input has no member 'value'") -- so they target the
# texture-capable back-ends only. The pure value/control objects (the RapidLib
# regressor/classifier, the QwenLLM text model) keep Pd. `dump` (the SDK-free target
# the portability smoke builds) and `ossia` are in both. Ignored in a score build.
set(ONNX_TEXTURE_BACKENDS dump ossia max touchdesigner:TOP godot:TEXTURE python)        # texture I/O
set(ONNX_DATA_BACKENDS    dump ossia pd max touchdesigner:CHOP_MESSAGE godot:NODE python) # value I/O

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME yolo_blob
  CLASS YOLO7BlobDetector
  NAMESPACE OnnxModels
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
  SOURCES
    OnnxModels/YOLO-blob.hpp
    OnnxModels/YOLO-blob.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME yolo_segmentation
  CLASS YOLO8Segmentation
  NAMESPACE OnnxModels
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
  SOURCES
    OnnxModels/YOLO-segmentation.hpp
    OnnxModels/YOLO-segmentation.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME resnet
  CLASS ResnetDetector
  NAMESPACE OnnxModels
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
  SOURCES
    OnnxModels/Resnet.hpp
    OnnxModels/Resnet.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME enet
  CLASS EmotionNetDetector
  NAMESPACE OnnxModels
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
  SOURCES
    OnnxModels/ENet.hpp
    OnnxModels/ENet.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME yolo_pose
  CLASS PoseDetector
  NAMESPACE OnnxModels::Yolo
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
  SOURCES
    OnnxModels/YOLO-pose.hpp
    OnnxModels/YOLO-pose.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME blazepose
  CLASS BlazePoseDetector
  NAMESPACE OnnxModels
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
  SOURCES
    OnnxModels/BlazePose.hpp
    OnnxModels/BlazePose.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME trtpose
  CLASS TRTPoseDetector
  NAMESPACE OnnxModels::TRTPose
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
  SOURCES
    OnnxModels/TRTPose.hpp
    OnnxModels/TRTPose.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME pose_detector
  CLASS PoseDetector
  NAMESPACE OnnxModels
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
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
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
  SOURCES
    OnnxModels/DepthAnything2.hpp
    OnnxModels/DepthAnything2.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME regressor
  CLASS Regressor
  NAMESPACE RapidlibModels
  BACKENDS ${ONNX_DATA_BACKENDS}
  SOURCES
    RapidlibModels/Regressor.hpp
    RapidlibModels/Regressor.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME classifier
  CLASS Classifier
  NAMESPACE RapidlibModels
  BACKENDS ${ONNX_DATA_BACKENDS}
  SOURCES
    RapidlibModels/Classifier.hpp
    RapidlibModels/Classifier.cpp)


# FastVLM and QwenLLM are the only objects that #include the onnxruntime-extensions
# tokenizer headers (Onnx/helpers/TokenIO.hpp etc.), so they are built only when the
# extensions subbuild above ran -- now every non-wasm back-end, score and standalone
# alike. The remaining objects compile against plain onnxruntime regardless.
if(SCORE_ONNX_ENABLE_EXTENSIONS)
avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME fastvlm
  CLASS FastVLMNode
  NAMESPACE OnnxModels
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
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
  BACKENDS ${ONNX_DATA_BACKENDS}
  SOURCES
    OnnxModels/QwenLLM.hpp
    OnnxModels/QwenLLM.cpp
    Onnx/helpers/QwenLLM.hpp
    Onnx/helpers/QwenLLM.cpp)
endif()

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME generative_image_gan
  CLASS GenerativeImageGAN
  NAMESPACE OnnxModels
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
  SOURCES
    OnnxModels/GenerativeImageGAN.hpp
    OnnxModels/GenerativeImageGAN.cpp)

avnd_addon_object(
  BASE score_addon_onnx_models
  C_NAME image_to_image_gan
  CLASS ImageToImageGAN
  NAMESPACE OnnxModels
  BACKENDS ${ONNX_TEXTURE_BACKENDS}
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
  # Two ctx.h GCC-isms MSVC can't parse are behind #ifndef-guarded config knobs,
  # so force the portable branches instead of patching the vendored header:
  #   CTX_FORCE_INLINES (default 1) -> CTX_INLINE = inline __attribute__((always_inline))
  #   SQUOZE_USE_BUILTIN_CLZ (default 1) -> __builtin_clz()
  # =0 picks plain `inline` and the manual bit-scan loop. MSVC still inlines at
  # /O2, and the clz fallback is only on the cold squoze text path -- no rasterizer
  # perf impact. (CTX_BRANCH_HINTS already defaults to 0, so __builtin_expect is
  # not compiled; the cpuid asm is behind CTX_X86_64, which is never defined.)
  target_compile_definitions(score_onnx_ctx_overlay PRIVATE
    CTX_FORCE_INLINES=0
    SQUOZE_USE_BUILTIN_CLZ=0)
  # ctx.h uses GCC's __restrict__ spelling (24 sites, the rasterizer composite
  # hot path); MSVC spells it __restrict. Map one to the other so the qualifier
  # (and its alias-optimization hint) survives rather than dropping it.
  target_compile_definitions(score_onnx_ctx_overlay PRIVATE __restrict__=__restrict)
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
# Own static lib, so it needs the C++ standard set independently of the base (same
# C++20+ helper headers, same C++17-default trap on the CI lanes).
target_compile_features(score_onnx_imageops PRIVATE cxx_std_23)
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

# Standalone distributable packages. Bundle the prebuilt libonnxruntime into the
# Max / TouchDesigner / Godot packages so every object can dlopen it from beside
# its own plugin module at runtime (Max support/, TD Plugins/, Godot
# bin/<os>-<arch>/ -- see Onnx/helpers/compat/dylib_loader.hpp get_module_folder()).
# avnd_addon_package reads the per-backend externals avnd_make_* registered and
# assembles one package per backend under <build>/package/<name>/, ready for CI to
# zip + upload. Ignored in a score build (the host bundles onnxruntime itself) and
# when building against an Avendish without the packaging helpers.
if(NOT AVND_ADDON_SCORE AND COMMAND avnd_addon_package)
  avnd_addon_package(
    NAME score-addon-onnx
    SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}"
    PACKAGE_ROOT "${CMAKE_BINARY_DIR}/package"
    BACKENDS max touchdesigner godot
    SUPPORT ${ONNXRUNTIME_SUPPORT_FILES})
endif()
