add_library(rapidlib STATIC
  "3rdparty/RapidLib/dependencies/bayesfilter/src/BayesianFilter.cpp"
  "3rdparty/RapidLib/dependencies/bayesfilter/src/filter_utilities.cpp"
  "3rdparty/RapidLib/dependencies/libsvm/libsvm.cpp"
  "3rdparty/RapidLib/src/classification.cpp"
  "3rdparty/RapidLib/src/dtw.cpp"
  "3rdparty/RapidLib/src/fastDTW.cpp"
  "3rdparty/RapidLib/src/knnClassification.cpp"
  "3rdparty/RapidLib/src/modelSet.cpp"
  "3rdparty/RapidLib/src/neuralNetwork.cpp"
  "3rdparty/RapidLib/src/rapidStream.cpp"
  "3rdparty/RapidLib/src/regression.cpp"
  "3rdparty/RapidLib/src/searchWindow.cpp"
  "3rdparty/RapidLib/src/seriesClassification.cpp"
  "3rdparty/RapidLib/src/svmClassification.cpp"
  "3rdparty/RapidLib/src/warpPath.cpp"
)
target_include_directories(rapidlib PUBLIC 3rdparty/RapidLib/)
target_compile_definitions(rapidlib PUBLIC RAPIDLIB_DISABLE_JSONCPP)
# This static lib is linked into the per-object shared libraries (classifier /
# regressor pull in modelSet / neuralNetwork / libsvm, whose vtable relocations
# are not PIC). A score build sets CMAKE_POSITION_INDEPENDENT_CODE globally;
# standalone does not, so the .so link fails with "recompile with -fPIC". Mirror
# the other helper static libs (imageops, ctx_overlay) and force PIC here.
set_target_properties(rapidlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
