// Models whose weights live in a separate .onnx_data file, loaded from bytes
// the way every node does (BUG-LEDGER X1). The fixture is y = x + W with the
// 8-float initializer W = 1..8 stored in tests/data/external-data/tiny.onnx_data.
#include <Onnx/helpers/OnnxContext.hpp>
#include <OnnxModels/Utils.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef SCORE_ONNX_TEST_DATA_DIR
#error "SCORE_ONNX_TEST_DATA_DIR must point to tests/data"
#endif

namespace
{
std::string slurp(const std::filesystem::path& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

// Runs the session from a working directory that does NOT contain the model,
// as score normally does.
struct ScopedCwd
{
  std::filesystem::path previous = std::filesystem::current_path();
  explicit ScopedCwd(const std::filesystem::path& p) { std::filesystem::current_path(p); }
  ~ScopedCwd() { std::filesystem::current_path(previous); }
};

std::vector<float> run(Onnx::OnnxRunContext& ctx)
{
  const auto& spec = ctx.readModelSpec();
  std::vector<float> x(8, 0.f);
  std::vector<int64_t> shape{8};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in[1]{
      Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), shape.data(), shape.size())};
  Ort::Value out[1]{Ort::Value{nullptr}};
  ctx.infer(spec, in, out);
  const float* y = out[0].GetTensorData<float>();
  return {y, y + 8};
}
}

TEST_CASE("External data: loading from bytes with the model path", "[onnx][external-data]")
{
  const auto model = std::filesystem::path(SCORE_ONNX_TEST_DATA_DIR) / "external-data" / "tiny.onnx";
  REQUIRE(std::filesystem::exists(model));
  const auto bytes = slurp(model);
  // The API pointer is set up by the first node, or here (ORT_API_MANUAL_INIT):
  // without it this test crashed when run on its own.
  REQUIRE(OnnxModels::initOnnxRuntime());

  ScopedCwd cwd{std::filesystem::temp_directory_path()};

  Onnx::OnnxRunContext ctx{bytes, model.string()};
  CHECK(run(ctx) == std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
}

TEST_CASE("External data: without the model path the weights are not found", "[onnx][external-data]")
{
  const auto model = std::filesystem::path(SCORE_ONNX_TEST_DATA_DIR) / "external-data" / "tiny.onnx";
  REQUIRE(std::filesystem::exists(model));
  const auto bytes = slurp(model);
  // The API pointer is set up by the first node, or here (ORT_API_MANUAL_INIT):
  // without it this test crashed when run on its own.
  REQUIRE(OnnxModels::initOnnxRuntime());

  ScopedCwd cwd{std::filesystem::temp_directory_path()};

  // Documents why the path is needed: bytes alone resolve tiny.onnx_data
  // against the working directory.
  CHECK_THROWS(Onnx::OnnxRunContext{bytes});
}
