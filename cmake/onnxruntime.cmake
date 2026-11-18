if(OSSIA_USE_SYSTEM_LIBRARIES)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime)
  find_path(ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
    PATH_SUFFIXES
      onnxruntime
  )

  if(NOT ONNXRUNTIME_LIBRARY)
    message("ONNXRUNTIME_LIBRARY not found")
    return()
  endif()
  if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message("ONNXRUNTIME_INCLUDE_DIR not found")
    return()
  endif()

  add_library(onnxruntime INTERFACE)
  add_library(onnxruntime::onnxruntime ALIAS onnxruntime)

  # Even in the distro case we still dlopen
  # as for instance some distors like debian don't export all the symbols
  # so at least we can fail gracefully
  target_compile_definitions(onnxruntime INTERFACE ORT_API_MANUAL_INIT=1)

  target_link_libraries(onnxruntime
    INTERFACE
       "${ONNXRUNTIME_LIBRARY}"
  )
  target_include_directories(onnxruntime
    INTERFACE
      "${ONNXRUNTIME_INCLUDE_DIR}"
  )
  return()
endif()

# Not found through package manager, let's look it up manually
if(FETCHCONTENT_FULLY_DISCONNECTED)
  return()
endif()

# URLs of the latest release
# The score build wants the GPU (CUDA) package on Windows/Linux x64 for runtime
# acceleration. Standalone / portability builds (pd/max/python/dump/...) only need the
# C++ API to compile and run for introspection -- they have no CUDA toolkit and pulling
# the multi-hundred-MB gpu_cuda13 archive is pure overhead -- so they take the prebuilt
# CPU package, exactly like the celtera ml template. AVND_ADDON_SCORE is set by
# Avendish's AvendishAddon.cmake (1 when building in/against ossia score, 0 standalone).
set(ONNXRUNTIME_VERSION "1.27.1")
if(AVND_ADDON_SCORE)
  set(_ort_gpu 1)
else()
  set(_ort_gpu 0)
endif()
# Emscripten. Microsoft publishes no wasm release asset at all: every asset on
# every onnxruntime release is a native package (win/osx/linux), and the npm
# onnxruntime-web payload is a finished emscripten MODULE (MODULARIZE=1,
# EXPORT_NAME=ortWasmThreaded, --no-entry), not an archive another emcc link can
# consume. The supported way to get onnxruntime into someone else's wasm binary
# is onnxruntime's own `--build_wasm_static_lib`, which bundles every dependency
# (protobuf-lite, onnx, re2, mlas, ...) into a single libonnxruntime.a; see
# https://onnxruntime.ai/docs/build/web.html. Building that from source needs a
# host protoc and 20-60 min, so take the packaged output of exactly that build
# from ossia/sdk, which produces one per release with the same emscripten the
# rest of the wasm build uses.
#
# The hand-published csukuangfj assets used before cannot serve here: they are
# legacy-EH, so they leave __resumeException undefined against score's
# -fwasm-exceptions/-sJSPI link, and whether a given one was configured with
# --enable_wasm_threads varies release to release (v1.27.1 is NON-threaded,
# which the guard below still checks for).
#
# The wasm version is pinned SEPARATELY from the native one. Every plugin that
# needs onnxruntime for wasm fetches this same archive under the same
# FetchContent name, so whichever is configured first provides it for the rest
# and they have to agree on the version: keep this in sync with
# sat-mtl/sherpa-plugins.
if(EMSCRIPTEN)
  set(ONNXRUNTIME_VERSION "1.27.0")
  set(OSSIA_SDK_RELEASE "sdk38" CACHE STRING "ossia/sdk release carrying the wasm onnxruntime")
  set(ONNXRUNTIME_URL "https://github.com/ossia/sdk/releases/download/${OSSIA_SDK_RELEASE}/onnxruntime-${ONNXRUNTIME_VERSION}-wasm.tar.xz")
  include(FetchContent)
  FetchContent_Declare(onnxruntime
    URL "${ONNXRUNTIME_URL}"
  )
  FetchContent_MakeAvailable(onnxruntime)
else()
  # github.com/microsoft/onnxruntime and its release assets are gone; the NuGet
  # packages carry the same binaries (identical sha256) and headers.
  set(_ort_nuget_package "Microsoft.ML.OnnxRuntime")
  set(_ort_nuget_hash "9359e46eba4482ded00e678c98f22b68f51bb411d7934f5516d64050edfa3383")
  set(_ort_lipo_rids "")
  if(WIN32)
    set(_ort_rid "win-x64")
    if(_ort_gpu)
      set(_ort_nuget_package "Microsoft.ML.OnnxRuntime.Gpu.Windows")
      set(_ort_nuget_hash "07ad9d174f19ba47c13bf1989ee0c9a8d4832481e2eb51a16dd659a71162381f")
    endif()
  elseif(APPLE)
    set(_ort_rid "osx-arm64")
    if(CMAKE_OSX_ARCHITECTURES MATCHES ".*x86.*")
      # Last version to support x64 builds
      set(ONNXRUNTIME_VERSION "1.23.2")
      set(_ort_nuget_hash "25fe172f7fcdf34f2b5b02c8b997f6b88abc282956849d4af4335af23d0c4e4a")
      set(_ort_rid "osx-x64")
      if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        set(_ort_lipo_rids osx-x64 osx-arm64)
      endif()
    endif()
  else()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64.*")
      set(_ort_rid "linux-arm64")
    else()
      set(_ort_rid "linux-x64")
      if(_ort_gpu)
        set(_ort_nuget_package "Microsoft.ML.OnnxRuntime.Gpu.Linux")
        set(_ort_nuget_hash "71d5b6f5596c25e2b8a92fe488b98ea4b580dde2adfab079d5651f768a01a512")
      endif()
    endif()
  endif()

  string(TOLOWER "${_ort_nuget_package}" _ort_nuget_id)
  set(ONNXRUNTIME_URL "https://api.nuget.org/v3-flatcontainer/${_ort_nuget_id}/${ONNXRUNTIME_VERSION}/${_ort_nuget_id}.${ONNXRUNTIME_VERSION}.nupkg")

  include(FetchContent)
  FetchContent_Declare(onnxruntime
    URL "${ONNXRUNTIME_URL}"
    URL_HASH SHA256=${_ort_nuget_hash}
    DOWNLOAD_NAME "${_ort_nuget_id}.${ONNXRUNTIME_VERSION}.zip"
  )
  FetchContent_MakeAvailable(onnxruntime)

  # Lay the package out as the former release archives: include/, lib/, VERSION_NUMBER.
  set(_ort_native "${onnxruntime_SOURCE_DIR}/runtimes/${_ort_rid}/native")
  set(_ort_lib "${onnxruntime_SOURCE_DIR}/lib")
  if(NOT EXISTS "${onnxruntime_SOURCE_DIR}/VERSION_NUMBER")
    if(EXISTS "${onnxruntime_SOURCE_DIR}/buildTransitive/native/include/onnxruntime_c_api.h")
      set(_ort_headers "${onnxruntime_SOURCE_DIR}/buildTransitive/native/include")
    else()
      set(_ort_headers "${onnxruntime_SOURCE_DIR}/build/native/include")
    endif()
    if(NOT EXISTS "${_ort_native}" OR NOT EXISTS "${_ort_headers}/onnxruntime_c_api.h")
      message(FATAL_ERROR "${_ort_nuget_package} ${ONNXRUNTIME_VERSION} has no ${_ort_rid} runtime")
    endif()

    file(REMOVE_RECURSE "${onnxruntime_SOURCE_DIR}/include" "${_ort_lib}")
    file(COPY "${_ort_headers}/" DESTINATION "${onnxruntime_SOURCE_DIR}/include")
    file(MAKE_DIRECTORY "${_ort_lib}")

    file(GLOB _ort_natives "${_ort_native}/*")
    if(APPLE)
      set(_ort_real "libonnxruntime.${ONNXRUNTIME_VERSION}.dylib")
      list(REMOVE_ITEM _ort_natives "${_ort_native}/libonnxruntime.dylib")
      if(_ort_lipo_rids)
        set(_ort_lipo_inputs "")
        foreach(_rid IN LISTS _ort_lipo_rids)
          list(APPEND _ort_lipo_inputs "${onnxruntime_SOURCE_DIR}/runtimes/${_rid}/native/libonnxruntime.dylib")
        endforeach()
        execute_process(
          COMMAND lipo -create ${_ort_lipo_inputs} -output "${_ort_lib}/${_ort_real}"
          COMMAND_ERROR_IS_FATAL ANY)
      else()
        file(COPY_FILE "${_ort_native}/libonnxruntime.dylib" "${_ort_lib}/${_ort_real}")
      endif()
      file(CREATE_LINK "${_ort_real}" "${_ort_lib}/libonnxruntime.1.dylib" SYMBOLIC)
      file(CREATE_LINK "${_ort_real}" "${_ort_lib}/libonnxruntime.dylib" SYMBOLIC)
    elseif(NOT WIN32)
      set(_ort_real "libonnxruntime.so.${ONNXRUNTIME_VERSION}")
      list(REMOVE_ITEM _ort_natives "${_ort_native}/libonnxruntime.so")
      file(COPY_FILE "${_ort_native}/libonnxruntime.so" "${_ort_lib}/${_ort_real}")
      file(CREATE_LINK "${_ort_real}" "${_ort_lib}/libonnxruntime.so.1" SYMBOLIC)
      file(CREATE_LINK "libonnxruntime.so.1" "${_ort_lib}/libonnxruntime.so" SYMBOLIC)
    endif()
    if(_ort_natives)
      file(COPY ${_ort_natives} DESTINATION "${_ort_lib}")
    endif()
    if(NOT WIN32)
      file(GLOB _ort_libs LIST_DIRECTORIES false "${_ort_lib}/*")
      foreach(_f IN LISTS _ort_libs)
        if(NOT IS_SYMLINK "${_f}")
          file(CHMOD "${_f}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
               GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
        endif()
      endforeach()
    endif()

    file(WRITE "${onnxruntime_SOURCE_DIR}/VERSION_NUMBER" "${ONNXRUNTIME_VERSION}\n")
  endif()
endif()

# Find the .so & header files and put them in CMake variables
# NO_CMAKE_FIND_ROOT_PATH: a cross-compiling toolchain (emscripten) sets
# CMAKE_FIND_ROOT_PATH_MODE_LIBRARY/INCLUDE to ONLY, which would re-root these
# explicit paths into the target sysroot and never find the package we just
# downloaded. It is a no-op on a native build, where no root path is set.
find_library(onnxruntime_LIBRARY
    NAMES onnxruntime
    PATHS "${onnxruntime_SOURCE_DIR}/lib"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(NOT onnxruntime_LIBRARY)
  if(OSSIA_SDK AND LINUX)
    set(onnxruntime_LIBRARY "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so")
  else()
    message(FATAL_ERROR "Could not find onnxruntime library")
  endif()
endif()

# score's wasm binary is linked with -pthread, which implies --shared-memory, and
# wasm-ld refuses shared memory as soon as ONE input object lacks the `atomics`
# target feature. The prebuilt archives are hand-published and have shipped both
# threaded and non-threaded (see the version note above), and a non-threaded one
# only surfaces at the very end of the build, as a link error in the final
# executable -- an hour in, with nothing to show for it. Detect it here instead
# and disable the addon: the browser build loses the onnx objects, everything
# else still links.
#
# An archive of wasm objects says so directly: each entry of a `target_features`
# section is a length-prefixed name whose length byte (0x07 for "atomics") is
# non-printable, so every feature name lands in a run of its own and matching it
# exactly cannot collide with a mangled symbol.
#
# An archive built with LTO holds bitcode instead, where the features are a
# function attribute and no `target_features` section exists yet -- it is
# emitted at codegen. Nothing can be read out of it with strings(1): the
# attribute lives in the bit-packed string table. Ask the compiler instead.
if(EMSCRIPTEN)
  file(STRINGS "${onnxruntime_LIBRARY}" _ort_wasm_atomics
    REGEX "^atomics$"
    LENGTH_MINIMUM 7
    LENGTH_MAXIMUM 7
    LIMIT_COUNT 1)

  if(NOT _ort_wasm_atomics)
    set(_ort_probe "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-atomics-probe")
    file(REMOVE_RECURSE "${_ort_probe}")
    file(MAKE_DIRECTORY "${_ort_probe}")

    execute_process(
      COMMAND "${CMAKE_AR}" t "${onnxruntime_LIBRARY}"
      OUTPUT_VARIABLE _ort_members
      ERROR_QUIET)
    string(REGEX MATCH "[^\r\n]+" _ort_member "${_ort_members}")

    if(_ort_member)
      execute_process(
        COMMAND "${CMAKE_AR}" x "${onnxruntime_LIBRARY}" "${_ort_member}"
        WORKING_DIRECTORY "${_ort_probe}"
        ERROR_QUIET)
      get_filename_component(_ort_member_file "${_ort_member}" NAME)
      execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -x ir -S -emit-llvm
                -o - "${_ort_probe}/${_ort_member_file}"
        OUTPUT_VARIABLE _ort_ir
        ERROR_QUIET)
      if(_ort_ir MATCHES "\"target-features\"=\"[^\"]*\\+atomics")
        set(_ort_wasm_atomics ON)
      endif()
    endif()

    file(REMOVE_RECURSE "${_ort_probe}")
  endif()

  if(NOT _ort_wasm_atomics)
    message(WARNING
      "onnxruntime ${ONNXRUNTIME_VERSION} for wasm was built WITHOUT threads "
      "(no 'atomics' target feature in ${onnxruntime_LIBRARY}); it cannot be "
      "linked into score's -pthread build. Disabling score-addon-onnx. Pin "
      "ONNXRUNTIME_VERSION in cmake/onnxruntime.cmake to a threaded release.")
    return()
  endif()
endif()

if(WIN32)
  find_file(onnxruntime_DLL
    "onnxruntime.dll"
    PATHS "${onnxruntime_SOURCE_DIR}/lib"
    NO_DEFAULT_PATH
  )
  file(GLOB onnxruntime_DLLS "${onnxruntime_SOURCE_DIR}/lib/*.dll")
endif()

# The prebuilt runtime shared library/libraries, to bundle into the standalone
# Max/TouchDesigner/Godot packages (avnd_addon_package SUPPORT). The objects
# dlopen these at startup and find them relative to their own module (see
# Onnx/helpers/compat/dylib_loader.hpp get_module_folder()).
if(EMSCRIPTEN)
  set(ONNXRUNTIME_SUPPORT_FILES "")
elseif(APPLE)
  file(GLOB ONNXRUNTIME_SUPPORT_FILES "${onnxruntime_SOURCE_DIR}/lib/*.dylib")
elseif(WIN32)
  file(GLOB ONNXRUNTIME_SUPPORT_FILES "${onnxruntime_SOURCE_DIR}/lib/*.dll")
else()
  file(GLOB ONNXRUNTIME_SUPPORT_FILES "${onnxruntime_SOURCE_DIR}/lib/*.so*")
endif()

find_path(onnxruntime_INCLUDE_DIRS
    NAMES onnxruntime_cxx_api.h
    PATHS "${onnxruntime_SOURCE_DIR}/include"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
)
if(NOT onnxruntime_INCLUDE_DIRS)
  if(OSSIA_SDK AND LINUX)
    set(onnxruntime_INCLUDE_DIRS "${onnxruntime_SOURCE_DIR}/include")
  else()
    message(FATAL_ERROR "Could not find onnxruntime headers")
  endif()
endif()

# Create an onnxruntime CMake target which will propagate these variables to the targets
# this target is linked to
if(EMSCRIPTEN)
  add_library(onnxruntime STATIC IMPORTED)
else()
  add_library(onnxruntime SHARED IMPORTED)
endif()

# Windows needs special handling because here linking to a library requires two files:
# The .lib and the .dll
if(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${onnxruntime_DLL}"
    IMPORTED_IMPLIB "${onnxruntime_LIBRARY}"
  )
  foreach(_config ${CMAKE_CONFIGURATION_TYPES})
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION_${_config} "${onnxruntime_DLL}"
      IMPORTED_IMPLIB_${_config} "${onnxruntime_LIBRARY}"
    )
  endforeach()
else()
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${onnxruntime_LIBRARY}")
  foreach(_config ${CMAKE_CONFIGURATION_TYPES})
    set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION_${_config} "${onnxruntime_LIBRARY}")
  endforeach()
endif()

# ORT_API_MANUAL_INIT makes the objects reach the C API through a dlopen of the
# prebuilt shared library (OnnxModels/Utils.hpp libonnxruntime). Emscripten links
# the static archive straight into the binary and has nothing to dlopen, so leave
# the definition off there and let initOnnxRuntime() take its "already there"
# path.
if(NOT EMSCRIPTEN)
  target_compile_definitions(onnxruntime INTERFACE ORT_API_MANUAL_INIT=1)
endif()
target_include_directories(onnxruntime INTERFACE "${onnxruntime_INCLUDE_DIRS}")

# Good practice: using an alias with :: in the name ensure that
# we're going to get quick errors if the library is not found
add_library(onnxruntime::onnxruntime ALIAS onnxruntime)

# In a standalone / portability build the dump + standalone back-end executables link
# onnxruntime and are run during the build (introspection); make the prebuilt shared
# library findable through rpath, like the celtera ml template. Score builds bundle
# onnxruntime themselves (see the install() block above) and set their own rpath, so
# only do this for the standalone case. Must be set before the targets are created.
if(NOT AVND_ADDON_SCORE)
  list(APPEND CMAKE_BUILD_RPATH "${onnxruntime_SOURCE_DIR}/lib")
  list(APPEND CMAKE_INSTALL_RPATH "${onnxruntime_SOURCE_DIR}/lib")
endif()


# The TensorRT execution provider in onnxruntime's GPU package links a specific
# TensorRT major (libnvinfer.so.10 / nvinfer_10.dll for 1.27.1). score ships no
# TensorRT, so installing it yields a library that can never load, and onnxruntime
# advertises TensorrtExecutionProvider because the file is present -- selecting it then
# falls back to CPU. Enable only alongside a matching TensorRT runtime.
option(SCORE_ONNX_INSTALL_TENSORRT_PROVIDER
       "Install onnxruntime's TensorRT execution provider" OFF)

if(SCORE_DEPLOYMENT_BUILD AND NOT OSSIA_USE_SYSTEM_LIBRARIES AND NOT SCORE_NO_INSTALL_ONNXRUNTIME)
    if(APPLE)
        file(GLOB ONNXRUNTIME_FILES "${onnxruntime_SOURCE_DIR}/lib/*.dylib")
    elseif(WIN32)
        file(GLOB ONNXRUNTIME_FILES "${onnxruntime_SOURCE_DIR}/lib/*.dll")
    else()
        file(GLOB ONNXRUNTIME_FILES "${onnxruntime_SOURCE_DIR}/lib/*.so*")
    endif()

    if(NOT SCORE_ONNX_INSTALL_TENSORRT_PROVIDER)
      list(FILTER ONNXRUNTIME_FILES EXCLUDE REGEX "onnxruntime_providers_tensorrt")
    endif()

  if(APPLE)
    set(SCORE_BUNDLEUTILITIES_DIRS_LIST "${SCORE_BUNDLEUTILITIES_DIRS_LIST};${onnxruntime_SOURCE_DIR}/lib/" CACHE INTERNAL "")

    set(_ort_real "libonnxruntime.${ONNXRUNTIME_VERSION}.dylib")
    set(_ort_aliases "")
    set(_ort_install "")
    foreach(_f IN LISTS ONNXRUNTIME_FILES)
      get_filename_component(_n "${_f}" NAME)
      if(_n STREQUAL "${_ort_real}")
        list(APPEND _ort_install "${_f}")
      elseif(_n MATCHES "^libonnxruntime(\\.1)?\\.dylib$")
        list(APPEND _ort_aliases "${_n}")
      else()
        list(APPEND _ort_install "${_f}")
      endif()
    endforeach()

    install(
      FILES ${_ort_install}
      DESTINATION "ossia score.app/Contents/Frameworks"
      COMPONENT OssiaScore)

    foreach(_alias IN LISTS _ort_aliases)
      install(CODE "
        set(_dir \"\${CMAKE_INSTALL_PREFIX}/ossia score.app/Contents/Frameworks\")
        file(REMOVE \"\${_dir}/${_alias}\")
        execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
                        \"${_ort_real}\" \"\${_dir}/${_alias}\")
      " COMPONENT OssiaScore)
    endforeach()
  elseif(WIN32)
    install(
      FILES ${ONNXRUNTIME_FILES}
      DESTINATION "${SCORE_BIN_INSTALL_DIR}"
      COMPONENT OssiaScore)
  else()
    install(
      FILES ${ONNXRUNTIME_FILES}
      DESTINATION lib
      COMPONENT OssiaScore)
  endif()
endif()
