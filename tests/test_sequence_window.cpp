// Sequence Processor: changing Window while running takes effect without a
// model reload. The fixture takes [?,8,3] (tests/data/sequence): Sliding
// streams 3-value frames into its 8-frame window, Passthrough feeds each
// payload on its own, zero-padded.
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <OnnxModels/SequenceProcessor.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace
{
const std::string model = SCORE_ONNX_TEST_DATA_DIR "/sequence/window8x3.onnx";

void feed(OnnxModels::SequenceProcessor& p, int k)
{
  p.inputs.in.value = {(float)k, 0.5f * (float)k, 1.f - (float)k};
  p();
}

void load(OnnxModels::SequenceProcessor& p, const std::string& bytes)
{
  p.inputs.model.file.bytes = bytes;
  p.inputs.model.file.filename = model;
  inlineWorker(p);
}
}

TEST_CASE("Sequence Processor: switching Window takes effect", "[onnx][sequence]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  const auto bytes = TestPaths::slurp(model);
  using W = OnnxModels::SeqWindowMode;

  // Reference: Sliding from the start, fed frames 1..8 (one full window).
  OnnxModels::SequenceProcessor sliding;
  load(sliding, bytes);
  sliding.inputs.window_mode.value = W::Sliding;
  for(int k = 1; k <= 8; k++)
    feed(sliding, k);
  const auto s = sliding.outputs.out.value;
  REQUIRE(s.size() == 2);

  // Passthrough on frame 0, then switched to Sliding.
  OnnxModels::SequenceProcessor switched;
  load(switched, bytes);
  switched.inputs.window_mode.value = W::Passthrough;
  feed(switched, 0);
  const auto p = switched.outputs.out.value;
  REQUIRE(p.size() == 2);
  REQUIRE(p != s);
  CHECK_FALSE(switched.inputs.model.current_model_invalid);

  switched.inputs.window_mode.value = W::Sliding;
  // The window restarts on the switch: nothing runs until it has filled...
  for(int k = 1; k < 8; k++)
    feed(switched, k);
  CHECK(switched.outputs.out.value == p);
  // ...and then it runs on the same 8 frames as the reference.
  feed(switched, 8);
  CHECK(switched.outputs.out.value == s);
}
