// Audio Processor Block input (BUG-LEDGER A7) and the Data outlet of the
// worker path (A8). A model whose length is free ran on 1024-sample blocks
// whatever it needed (Demucs: 343980), and a model on the worker never wrote
// Data, the output's RMS.
#include <OnnxModels/AudioProcessor.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

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

// Feeds `seconds` of a 0.5 sine at 48 kHz; returns how many ticks produced
// a Data message, the last one in `last`.
int run(OnnxModels::AudioProcessor& node, double seconds, std::vector<float>* last = nullptr)
{
  constexpr int frames = 512;
  std::vector<float> in(frames), out(frames);
  float* ins[1]{in.data()};
  float* outs[1]{out.data()};
  node.inputs.audio.samples = ins;
  node.inputs.audio.channels = 1;
  node.outputs.audio.samples = outs;
  node.outputs.audio.channels = 1;
  int runs = 0;
  for(int t0 = 0; t0 < seconds * 48000; t0 += frames)
  {
    for(int i = 0; i < frames; i++)
      in[i] = 0.5f * (float)std::sin(2. * std::numbers::pi * 440. * (t0 + i) / 48000.);
    node.outputs.data.value.clear();
    node(frames);
    if(!node.outputs.data.value.empty())
    {
      runs++;
      if(last)
        *last = node.outputs.data.value;
    }
  }
  return runs;
}

struct Node
{
  OnnxModels::AudioProcessor node;
  std::string name = SCORE_ONNX_TEST_DATA_DIR "/audio/identity_dyn.onnx";
  std::string bytes = slurp(name);
  explicit Node(int block)
  {
    node.prepare({.rate = 48000., .input_channels = 1, .output_channels = 1, .frames = 512});
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.inputs.model_rate.value = 48000;
    node.inputs.block.value = block;
    node.worker.request = [this](std::unique_ptr<OnnxModels::AudioInferJob> job) {
      if(auto done = OnnxModels::AudioProcessor::worker::work(std::move(job)))
        done(node);
    };
  }
};
}

TEST_CASE("Audio Processor: Block sets a free-length model's block", "[onnx][audio]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Node defaults{0}, b2048{2048};
  // One second as 94 ticks of 512: 48128 samples, 47 blocks of 1024, 23 of 2048.
  CHECK(run(defaults.node, 1.) == 47);
  CHECK(run(b2048.node, 1.) == 23);
}

TEST_CASE("Audio Processor: the worker path writes Data", "[onnx][audio]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  // Above 48000 samples the model runs on the worker: 1.5 s is one block.
  Node big{49152};
  std::vector<float> last;
  const int runs = run(big.node, 1.5, &last);
  CHECK(runs == 1);
  REQUIRE(last.size() == 1);
  // RMS of a 0.5 sine.
  CHECK(last[0] == Catch::Approx(0.5 / std::sqrt(2.)).margin(0.01));
}
