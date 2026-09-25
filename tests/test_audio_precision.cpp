// The audio nodes' sessions run in full float32 (BUG-LEDGER A11). CUDA's
// default TF32 math has a 10-bit mantissa: a noise floor near -60 dB, which
// deep-music-enhancer (+60 dB above its cut-off) turned into a howl. On a
// machine without CUDA both paths are exact and the test passes trivially.
#include <Onnx/helpers/OnnxContext.hpp>
#include <OnnxModels/Utils.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace
{
// Worst relative error of a [64,4096] x [4096,256] product against double.
double matmulError(const Onnx::Options& opts)
{
  std::ifstream f(SCORE_ONNX_TEST_DATA_DIR "/audio/matmul.onnx", std::ios::binary);
  const std::string bytes{std::istreambuf_iterator<char>(f), {}};
  REQUIRE(!bytes.empty());
  Onnx::OnnxRunContext ctx(bytes, {}, opts);

  constexpr int64_t M = 64, K = 4096, N = 256;
  std::mt19937 rng{1};
  std::normal_distribution<float> d;
  std::vector<float> a(M * K), b(K * N);
  for(auto& v : a)
    v = d(rng);
  for(auto& v : b)
    v = d(rng);

  const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t sa[2]{M, K}, sb[2]{K, N};
  Ort::Value ins[2]{
      Ort::Value::CreateTensor<float>(mem, a.data(), a.size(), sa, 2),
      Ort::Value::CreateTensor<float>(mem, b.data(), b.size(), sb, 2)};
  const char* in_names[2]{"a", "b"};
  const char* out_names[1]{"c"};
  auto outs = ctx.session.Run(Ort::RunOptions{nullptr}, in_names, ins, 2, out_names, 1);
  const float* c = outs[0].GetTensorData<float>();

  double worst = 0.;
  for(int64_t i = 0; i < M; i += 7)
    for(int64_t j = 0; j < N; j += 5)
    {
      double ref = 0., mag = 0.;
      for(int64_t k = 0; k < K; k++)
      {
        ref += (double)a[i * K + k] * b[k * N + j];
        mag += std::abs((double)a[i * K + k] * b[k * N + j]);
      }
      worst = std::max(worst, std::abs(c[i * N + j] - ref) / mag);
    }
  return worst;
}
}

TEST_CASE("Audio sessions compute in float32 rather than TF32", "[onnx][audio]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  const double precise = matmulError(Onnx::Options::precise());
  INFO("default options (TF32 on CUDA): " << matmulError({}) << ", precise: " << precise);
  CHECK(precise < 1e-6);
}
