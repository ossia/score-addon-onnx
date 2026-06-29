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
set(ONNXRUNTIME_VERSION "1.24.1")
if(AVND_ADDON_SCORE)
  set(_ort_gpu 1)
else()
  set(_ort_gpu 0)
endif()
if(WIN32)
  if(_ort_gpu)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-gpu_cuda13-${ONNXRUNTIME_VERSION}.zip")
  else()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
  endif()
elseif(APPLE)
  if(CMAKE_OSX_ARCHITECTURES MATCHES ".*x86.*")
    # Last version to support x64 builds
    set(ONNXRUNTIME_VERSION "1.23.2")
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
  else()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz")
  endif()
else()
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64.*")
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz")
  elseif(_ort_gpu)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu_cuda13-${ONNXRUNTIME_VERSION}.tgz")
  else()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
  endif()
endif()

# Ask CMake to download it
include(FetchContent)
FetchContent_Declare(onnxruntime
  URL "${ONNXRUNTIME_URL}"
)
FetchContent_MakeAvailable(onnxruntime)

# Find the .so & header files and put them in CMake variables
find_library(onnxruntime_LIBRARY
    NAMES onnxruntime
    PATHS "${onnxruntime_SOURCE_DIR}/lib"
    NO_DEFAULT_PATH
)

if(NOT onnxruntime_LIBRARY)
  if(OSSIA_SDK AND LINUX)
    set(onnxruntime_LIBRARY "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so")
  else()
    message(FATAL_ERROR "Could not find onnxruntime library")
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
if(APPLE)
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
add_library(onnxruntime SHARED IMPORTED)

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

target_compile_definitions(onnxruntime INTERFACE ORT_API_MANUAL_INIT=1)
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


if(SCORE_DEPLOYMENT_BUILD AND NOT OSSIA_USE_SYSTEM_LIBRARIES AND NOT SCORE_NO_INSTALL_ONNXRUNTIME)
    if(APPLE)
        file(GLOB ONNXRUNTIME_FILES "${onnxruntime_SOURCE_DIR}/lib/*.dylib")
    elseif(WIN32)
        file(GLOB ONNXRUNTIME_FILES "${onnxruntime_SOURCE_DIR}/lib/*.dll")
    else()
        file(GLOB ONNXRUNTIME_FILES "${onnxruntime_SOURCE_DIR}/lib/*.so*")
    endif()

  if(APPLE)
    set(SCORE_BUNDLEUTILITIES_DIRS_LIST "${SCORE_BUNDLEUTILITIES_DIRS_LIST};${onnxruntime_SOURCE_DIR}/lib/" CACHE INTERNAL "")

    install(
      FILES ${ONNXRUNTIME_FILES}
      DESTINATION "ossia score.app/Contents/Frameworks"
      COMPONENT OssiaScore)
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
