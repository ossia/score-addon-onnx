cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT TARGET score_lib_base)
  include("${SCORE_SOURCE_DIR}/cmake/ScoreExternalAddon.cmake")
endif()

project(score_addon_onnx LANGUAGES CXX)

find_library(ONNXRUNTIME_LIBRARY onnxruntime)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNXRUNTIME_LIBRARY not found")
endif()
if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIR not found")
endif()

find_package(OpenCV REQUIRED)
# General initialization
score_common_setup()

# Creation of the library
add_library(score_addon_onnx
    Onnx/Executor.hpp
    Onnx/Metadata.hpp
    Onnx/Presenter.hpp
    Onnx/Process.hpp
    Onnx/View.hpp
    Onnx/Layer.hpp
    Onnx/CommandFactory.hpp

    score_addon_onnx.hpp

    Onnx/Executor.cpp
    Onnx/Presenter.cpp
    Onnx/Process.cpp
    Onnx/View.cpp
    Onnx/CommandFactory.cpp

    score_addon_onnx.cpp
)

# Scan for commands which creates necessary files
# Second argument must be a CMake list of all headers containing commands 

score_generate_command_list_file(
  score_addon_onnx
  "Onnx/CommandFactory.hpp" 
)

# Link

target_link_libraries(score_addon_onnx
    PUBLIC
     score_plugin_engine
     "${ONNXRUNTIME_LIBRARY}"

     opencv_dnn
     opencv_imgproc
     opencv_core
     opencv_imgcodecs
     opencv_highgui
 )
target_include_directories(score_addon_onnx PUBLIC "${ONNXRUNTIME_INCLUDE_DIR}")
# Target-specific options
setup_score_plugin(score_addon_onnx)
