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
//   ONNX_TEST_GAN_MODELS    fbanime-gan/, MobileStyleGAN.pytorch/ exports
//   ONNX_TEST_INSTANTHMR    InstantHMR's instanthmr.onnx (huggingface.co/
//                           momolesang/InstantHMR; SAM licence, not bundled)
//   ONNX_TEST_MHR           MHR body models: bin/mhr_lod<L>.mhrbin
//                           (tools/mhr_export.py) + ref/
//   ONNX_TEST_INSTANTHMR_VIDEO  a folder of frame_NNNN.jpg, one person
//   ONNX_TEST_DUMP_DIR      where the InstantHMR video test writes its parity
//                           dump for the Python reference (off when unset)
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
inline std::string instantHmr()
{
  return env("ONNX_TEST_INSTANTHMR");
}
inline std::string mhr()
{
  return env("ONNX_TEST_MHR");
}
inline std::string instantHmrVideo()
{
  return env("ONNX_TEST_INSTANTHMR_VIDEO");
}
inline std::string dumpDir()
{
  return env("ONNX_TEST_DUMP_DIR");
}
inline std::string mhrBin(int lod)
{
  const auto root = mhr();
  return root.empty() ? root : root + "/bin/mhr_lod" + std::to_string(lod) + ".mhrbin";
}

inline std::string slurp(const std::filesystem::path& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

inline bool haveAll(std::initializer_list<std::string> paths)
{
  for(const auto& p : paths)
    if(!std::filesystem::exists(p))
      return false;
  return true;
}
}
