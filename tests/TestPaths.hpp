#pragma once
// Where the node tests find the models and images they need. Each root can
// be moved with an environment variable; the defaults are the author's
// machine. A test whose file is missing SKIPs.
//
//   ONNX_TEST_MODELS        the preset pack's models/ (image-processor/, ...)
//   ONNX_TEST_WILD_MODELS   model collections: wild/, wild2/, pinto/, sherpa/
//   ONNX_TEST_VLM_MODELS    VLM snapshots: FastVLM-0.5B-ONNX/, SmolVLM-*, gemma-3-*
//   ONNX_TEST_LIBREONNX     the libreonnx pack
//   ONNX_TEST_IMAGES        test photos: body.jpg, face.png
//   ONNX_TEST_AILIA         an ailia-models checkout (sample images)
//   ONNX_TEST_POSE_PACKAGE  score's pose-detector package
//   ONNX_TEST_PINTO_ZOO     a PINTO_model_zoo checkout
#include <cstdlib>
#include <string>

namespace TestPaths
{
inline std::string env(const char* var, std::string fallback)
{
  if(const char* v = std::getenv(var); v && *v)
    return v;
  return fallback;
}
inline std::string home()
{
  return env("HOME", "/tmp");
}

inline std::string models()
{
  return env("ONNX_TEST_MODELS", "/mnt/win2/models/models-presets/models");
}
inline std::string wild()
{
  return env("ONNX_TEST_WILD_MODELS", "/mnt/sdd1/models");
}
inline std::string vlm()
{
  return env("ONNX_TEST_VLM_MODELS", "/mnt/win2/models");
}
inline std::string libreonnx()
{
  return env("ONNX_TEST_LIBREONNX", "/mnt/win2/models/libreonnx");
}
inline std::string images()
{
  return env("ONNX_TEST_IMAGES", libreonnx() + "/ossia-detection-model-pack/test_images");
}
inline std::string ailia()
{
  return env("ONNX_TEST_AILIA", home() + "/projets/oss/ailia-models");
}
inline std::string posePackage()
{
  return env("ONNX_TEST_POSE_PACKAGE", home() + "/Documents/ossia/score/packages/pose-detector");
}
inline std::string pintoZoo()
{
  return env("ONNX_TEST_PINTO_ZOO", "/mnt/win2/PINTO_model_zoo");
}
}
