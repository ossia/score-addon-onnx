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
# RapidLib's sources use range-based for with an initializer (C++20). A score
# build sets the C++ standard globally so this compiles; a standalone avnd-addon
# build has no such parent, so MSVC falls to its /std:c++17 default and hard-errors
# (C7585: "range-based for statement with an initializer requires at least
# '/std:c++20'"). gcc/clang accept it as an extension, so only the Windows/MSVC
# standalone lanes failed. Require it on the target; own static lib, own standard.
target_compile_features(rapidlib PUBLIC cxx_std_20)
# This static lib is linked into the per-object shared libraries (classifier /
# regressor pull in modelSet / neuralNetwork / libsvm, whose vtable relocations
# are not PIC). A score build sets CMAKE_POSITION_INDEPENDENT_CODE globally;
# standalone does not, so the .so link fails with "recompile with -fPIC". Mirror
# the other helper static libs (imageops, ctx_overlay) and force PIC here.
set_target_properties(rapidlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
