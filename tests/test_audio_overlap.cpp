// Audio Processor Overlap: a block is taken every block/2 or block/4 samples
// and the outputs
// are overlap-added with a Hann window: a model that returns its frame
// unchanged must give the input back.
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <OnnxModels/AudioProcessor.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>
#include <string>
#include <vector>

namespace
{
float signal(int t)
{
  const double w = 2. * std::numbers::pi / 48000.;
  return (float)(0.3 * std::sin(w * 440. * t) + 0.2 * std::sin(w * 1234. * t + 1.));
}

// Runs 1 s of signal through the identity model; returns the output.
std::vector<float> run(OnnxModels::AudioOverlap overlap, int& runs)
{
  const std::string model = SCORE_ONNX_TEST_DATA_DIR "/audio/identity_512.onnx";
  const auto bytes = TestPaths::slurp(model);
  constexpr int frames = 64;

  OnnxModels::AudioProcessor node;
  node.prepare({.rate = 48000., .input_channels = 1, .output_channels = 1, .frames = frames});
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = model;
  node.inputs.model_rate.value = 48000;
  node.inputs.overlap.value = overlap;
  runs = 0;
  inlineWorker(node);

  std::vector<float> in(frames), out(frames), all;
  float* ins[1]{in.data()};
  float* outs[1]{out.data()};
  node.inputs.audio.samples = ins;
  node.inputs.audio.channels = 1;
  node.outputs.audio.samples = outs;
  node.outputs.audio.channels = 1;
  for(int t0 = 0; t0 < 48000; t0 += frames)
  {
    for(int i = 0; i < frames; i++)
      in[i] = signal(t0 + i);
    node(frames);
    all.insert(all.end(), out.begin(), out.end());
    // The RMS summary is refreshed on every model run.
    if(!node.outputs.data.value.empty())
    {
      runs++;
      node.outputs.data.value.clear();
    }
  }
  REQUIRE_FALSE(node.inputs.model.current_model_invalid);
  return all;
}

// The largest error once the output has caught up, at the best latency. A
// latency is dropped as soon as its error exceeds the best one so far.
float reconstructionError(const std::vector<float>& out)
{
  std::vector<float> in(out.size());
  for(std::size_t t = 0; t < in.size(); t++)
    in[t] = signal((int)t);
  float best = 1e9f;
  for(int lag = 0; lag <= 2048; lag++)
  {
    float err = 0.f;
    for(int t = lag + 1024; t < (int)out.size() && err < best; t++)
      err = std::max(err, std::abs(out[t] - in[t - lag]));
    best = std::min(best, err);
  }
  return best;
}
}

TEST_CASE("Audio Processor: overlap-add gives the input back", "[onnx][audio]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  using O = OnnxModels::AudioOverlap;
  int none = 0, half = 0, quarter = 0;

  CHECK(reconstructionError(run(O::None, none)) < 1e-4f);
  CHECK(reconstructionError(run(O::Half, half)) < 1e-4f);
  CHECK(reconstructionError(run(O::ThreeQuarters, quarter)) < 1e-4f);

  // A block every 512, 256 and 128 samples of the 48000 fed.
  CHECK(none >= 90);
  CHECK(half >= 2 * none - 2);
  CHECK(quarter >= 4 * none - 4);
}

// A mono model on a stereo output: both host channels get every block of the
// one model ring.
TEST_CASE("Audio output: a mono model plays on both host channels", "[onnx][audio]")
{
  Onnx::WaveformOutput out;
  out.prepare(1, 48000., 48000., 1024, 256);
  std::vector<float> ramp(1024);
  for(int i = 0; i < 1024; i++)
    ramp[i] = (float)i;
  out.push(ramp.data(), 1, 1024);

  std::vector<float> l(256), r(256), all;
  float* chans[2]{l.data(), r.data()};
  for(int k = 0; k < 3; k++)
  {
    out.pull(chans, 2, 256);
    CHECK(l == r);
    all.insert(all.end(), l.begin(), l.end());
  }
  // Consecutive: no block went to the other side.
  for(std::size_t i = 1; i < all.size(); i++)
    REQUIRE(all[i] >= all[i - 1]);
  CHECK(all.back() > 700.f);
}
