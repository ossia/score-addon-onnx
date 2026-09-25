// Sequence Processor: changing Window while running (BUG-LEDGER S1). The
// window mode was only resolved on model reload, so switching it later did
// nothing. RNNoise takes [?,100,42]: Sliding streams 42-value frames into its
// 100-frame window, Passthrough feeds each payload on its own.
#include <tests/TestPaths.hpp>
#include <OnnxModels/SequenceProcessor.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace
{
std::string seqModel()
{
  return TestPaths::models()
         + "/sequence-processor/rnnoise-rnn.onnx";
}

std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

std::vector<float> frame(int k)
{
  std::vector<float> v(42);
  for(int i = 0; i < 42; i++)
    v[i] = 0.1f * (float)((i * 7 + k * 13) % 17) - 0.5f;
  return v;
}

void feed(OnnxModels::SequenceProcessor& p, int k)
{
  p.inputs.in.value = frame(k);
  p();
}

void load(OnnxModels::SequenceProcessor& p, const std::string& bytes, const std::string& name)
{
  p.inputs.model.file.bytes = bytes;
  p.inputs.model.file.filename = name;
}
}

TEST_CASE("Sequence Processor: switching Window takes effect", "[onnx][sequence]")
{
  const auto model = seqModel();
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  const auto bytes = slurp(model);
  using W = OnnxModels::SeqWindowMode;

  // Reference: Sliding from the start, fed frames 1..100 (one full window).
  OnnxModels::SequenceProcessor sliding;
  load(sliding, bytes, model);
  sliding.inputs.window_mode.value = W::Sliding;
  for(int k = 1; k <= 100; k++)
    feed(sliding, k);
  const auto s = sliding.outputs.out.value;
  REQUIRE(!s.empty());

  // Passthrough on frame 0, then switched to Sliding.
  OnnxModels::SequenceProcessor switched;
  load(switched, bytes, model);
  switched.inputs.window_mode.value = W::Passthrough;
  feed(switched, 0);
  const auto p = switched.outputs.out.value;
  REQUIRE(!p.empty());
  REQUIRE(p != s);

  switched.inputs.window_mode.value = W::Sliding;
  // The window restarts on the switch: nothing runs until it has filled...
  for(int k = 1; k < 100; k++)
    feed(switched, k);
  CHECK(switched.outputs.out.value == p);
  // ...and then it runs on the same 100 frames as the reference.
  feed(switched, 100);
  CHECK(switched.outputs.out.value == s);
}
