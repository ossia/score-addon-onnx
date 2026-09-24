// Sequence Processor Normalize input (BUG-LEDGER S4): embedding heads trained
// on unit-norm CLIP embeddings saturate on raw ones, and the pack had to ship
// a wrapper model with an LpNormalization in front of the head.
#include <OnnxModels/SequenceProcessor.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace
{
std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

struct Seq
{
  OnnxModels::SequenceProcessor node;
  std::string name, bytes;
  explicit Seq(const std::string& path)
      : name{path}
      , bytes{slurp(path)}
  {
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.worker.request = [this](std::unique_ptr<OnnxModels::SeqInferJob> job) {
      if(auto done = OnnxModels::SequenceProcessor::worker::work(std::move(job)))
        done(node);
    };
  }
  std::vector<float> run(std::vector<float> in)
  {
    node.inputs.in.value = std::move(in);
    node();
    return node.outputs.out.value;
  }
};
}

TEST_CASE("Sequence Processor: Normalize L2 and ZScore", "[onnx][sequence]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  // y = x * Param 1: Out shows the fed input.
  Seq s{SCORE_ONNX_TEST_DATA_DIR "/aux/scale.onnx"};
  s.node.inputs.param1.value = 1.f;
  using N = OnnxModels::SeqNormalize;

  CHECK(s.run({3.f, 0.f, 4.f, 0.f}) == std::vector<float>{3.f, 0.f, 4.f, 0.f});

  s.node.inputs.normalize.value = N::L2;
  const auto l2 = s.run({3.f, 0.f, 4.f, 0.f});
  REQUIRE(l2.size() == 4);
  CHECK(l2[0] == Catch::Approx(0.6f));
  CHECK(l2[2] == Catch::Approx(0.8f));

  s.node.inputs.normalize.value = N::ZScore;
  const auto z = s.run({1.f, 2.f, 3.f, 4.f});
  REQUIRE(z.size() == 4);
  // mean 2.5, population sd sqrt(1.25)
  CHECK(z[0] == Catch::Approx(-1.5 / std::sqrt(1.25)));
  CHECK(z[3] == Catch::Approx(1.5 / std::sqrt(1.25)));

  // A constant frame has no spread: only centred.
  CHECK(s.run({2.f, 2.f, 2.f, 2.f}) == std::vector<float>{0.f, 0.f, 0.f, 0.f});
}

TEST_CASE("Sequence Processor: the raw CLIP NSFW head with L2 matches the wrapper", "[onnx][sequence]")
{
  const char* env = std::getenv("ONNX_TEST_WILD_MODELS");
  const std::string raw
      = std::string(env ? env : "/mnt/sdd1/models") + "/wild2/"
        "nsfw_detector__clip-based-nsfw-detector__clip_nsfw_b32.onnx";
  const char* penv = std::getenv("ONNX_TEST_MODELS");
  const std::string wrapped
      = std::string(penv ? penv : "/mnt/win2/models/models-presets/models")
        + "/sequence-processor/clip-vit-b32-nsfw-head-l2norm.onnx";
  if(!std::filesystem::exists(raw) || !std::filesystem::exists(wrapped))
    SKIP("models not found");
  REQUIRE(OnnxModels::initOnnxRuntime());

  Seq head{raw}, wrapper{wrapped};
  head.node.inputs.normalize.value = OnnxModels::SeqNormalize::L2;

  std::mt19937 rng{7};
  std::normal_distribution<float> d;
  for(int k = 0; k < 4; k++)
  {
    // An un-normalised CLIP image embedding: norm around 10.
    std::vector<float> e(512);
    double sq = 0.;
    for(auto& x : e)
    {
      x = d(rng);
      sq += x * x;
    }
    for(auto& x : e)
      x *= (float)(10. / std::sqrt(sq));

    const auto a = head.run(e);
    const auto b = wrapper.run(e);
    REQUIRE(a.size() == 1);
    REQUIRE(b.size() == 1);
    CHECK(a[0] == Catch::Approx(b[0]).margin(1e-4));
  }
}
