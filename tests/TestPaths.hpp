#pragma once
// Where the tests find their files. The bundled fixtures are under
// tests/data (SCORE_ONNX_TEST_DATA_DIR). Real models and photos are too large
// or not licensed to be bundled: their roots come from environment variables
// only, and a test whose files are missing SKIPs. Such tests carry the
// [model] tag (see the ctest registrations in CMakeLists.txt).
//
//   ONNX_TEST_MODELS        the preset pack's models/ (image-processor/, ...)
//   ONNX_TEST_WILD_MODELS   model collections: wild/, wild2/, pinto/, sherpa/
//   ONNX_TEST_VLM_MODELS    VLM snapshots: FastVLM-0.5B-ONNX/, SmolVLM-*, gemma-3-*
//   ONNX_TEST_LIBREONNX     the libreonnx pack
//   ONNX_TEST_IMAGES        test photos: body.jpg, face.png, animal.jpg
//                           (default: the libreonnx pack's test_images/)
//   ONNX_TEST_AILIA         an ailia-models checkout (sample images, GANs)
//   ONNX_TEST_POSE_PACKAGE  score's pose-detector package
//   ONNX_TEST_PINTO_ZOO     a PINTO_model_zoo checkout
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string>

namespace TestPaths
{
inline std::string env(const char* var)
{
  if(const char* v = std::getenv(var); v && *v)
    return v;
  return {};
}

inline std::string models()
{
  return env("ONNX_TEST_MODELS");
}
inline std::string wild()
{
  return env("ONNX_TEST_WILD_MODELS");
}
inline std::string vlm()
{
  return env("ONNX_TEST_VLM_MODELS");
}
inline std::string libreonnx()
{
  return env("ONNX_TEST_LIBREONNX");
}
inline std::string images()
{
  if(auto dir = env("ONNX_TEST_IMAGES"); !dir.empty())
    return dir;
  if(auto pack = libreonnx(); !pack.empty())
    return pack + "/ossia-detection-model-pack/test_images";
  return {};
}
inline std::string ailia()
{
  return env("ONNX_TEST_AILIA");
}
inline std::string posePackage()
{
  return env("ONNX_TEST_POSE_PACKAGE");
}
inline std::string pintoZoo()
{
  return env("ONNX_TEST_PINTO_ZOO");
}
inline std::string gan()
{
  return env("ONNX_TEST_GAN_MODELS");
}
}
