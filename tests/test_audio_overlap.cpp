// Audio Processor Overlap (BUG-LEDGER A4): the hop was always the block, so
// frame-based streaming models could not be run on overlapping blocks. With
// Overlap, a block is taken every block/2 or block/4 samples and the outputs
// are overlap-added with a Hann window: a model that returns its frame
// unchanged must give the input back.
#include <OnnxModels/AudioProcessor.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <fstream>
#include <iterator>
#include <numbers>
#include <string>
#include <vector>

namespace
{
std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

float signal(int t)
{
  const double w = 2. * std::numbers::pi / 48000.;
  return (float)(0.3 * std::sin(w * 440. * t) + 0.2 * std::sin(w * 1234. * t + 1.));
}

// Runs 1 s of signal through the identity model; returns the output.
std::vector<float> run(OnnxModels::AudioOverlap overlap, int& runs)
{
  const std::string model = SCORE_ONNX_TEST_DATA_DIR "/audio/identity_512.onnx";
  const auto bytes = slurp(model);
  constexpr int frames = 64;

  OnnxModels::AudioProcessor node;
  node.prepare({.rate = 48000., .input_channels = 1, .output_channels = 1, .frames = frames});
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = model;
  node.inputs.model_rate.value = 48000;
  node.inputs.overlap.value = overlap;
  runs = 0;
  node.worker.request = [&](std::unique_ptr<OnnxModels::AudioInferJob> job) {
    if(auto done = OnnxModels::AudioProcessor::worker::work(std::move(job)))
      done(node);
  };

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

// The largest error once the output has caught up, at the best latency.
float reconstructionError(const std::vector<float>& out)
{
  float best = 1e9f;
  for(int lag = 0; lag <= 2048; lag++)
  {
    float err = 0.f;
    for(int t = lag + 1024; t < (int)out.size(); t++)
      err = std::max(err, std::abs(out[t] - signal(t - lag)));
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
