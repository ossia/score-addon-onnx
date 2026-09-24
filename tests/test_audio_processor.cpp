// Audio Processor async path (BUG-LEDGER A1): the next block was popped from
// the input ring before checking for a job in flight, so every block that
// became ready while the worker was busy was consumed and dropped.
#include <OnnxModels/AudioAnalyzer.hpp>
#include <OnnxModels/AudioProcessor.hpp>

#include <Onnx/helpers/AudioRate.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <numbers>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace
{
std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}
}

TEST_CASE("Audio Processor: blocks wait while the worker is busy", "[onnx][audio]")
{
  const std::string model = SCORE_ONNX_TEST_DATA_DIR "/aux/audio_big_block.onnx";
  const auto bytes = slurp(model);
  constexpr int frames = 4096; // 12 ticks per 49152-sample block

  OnnxModels::AudioProcessor node;
  node.prepare({.rate = 48000., .input_channels = 1, .output_channels = 1, .frames = frames});
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = model;

  // The worker takes 14 ticks per job: longer than a block period.
  struct Pending
  {
    std::unique_ptr<OnnxModels::AudioInferJob> job;
    int due;
  };
  std::deque<Pending> worker;
  int tick = 0, dispatches = 0;
  node.worker.request = [&](std::unique_ptr<OnnxModels::AudioInferJob> job) {
    dispatches++;
    worker.push_back({std::move(job), tick + 14});
  };

  std::vector<float> in(frames, 0.25f), out(frames);
  float* ins[1]{in.data()};
  float* outs[1]{out.data()};
  node.inputs.audio.samples = ins;
  node.inputs.audio.channels = 1;
  node.outputs.audio.samples = outs;
  node.outputs.audio.channels = 1;

  // 20 blocks of input.
  for(tick = 0; tick < 12 * 20; tick++)
  {
    while(!worker.empty() && worker.front().due <= tick)
    {
      if(auto done = OnnxModels::AudioProcessor::worker::work(std::move(worker.front().job)))
        done(node);
      worker.pop_front();
    }
    node(frames);
  }
  REQUIRE(!node.inputs.model.current_model_invalid);
  // One job per 14 ticks: 17 jobs, the rest still waiting in the ring. The old
  // code dropped the blocks that arrived meanwhile and ran one job per 2
  // blocks, 10 in all.
  CHECK(dispatches >= 16);
}

namespace
{
// Stereo host at 48 kHz, 512-frame ticks of a constant input; returns the
// last output frame of each channel.
struct AudioHarness
{
  OnnxModels::AudioProcessor node;
  std::string name, bytes;
  std::vector<float> l = std::vector<float>(512, 0.1f), r = l, ol = l, orr = l;
  float* ins[2]{l.data(), r.data()};
  float* outs[2]{ol.data(), orr.data()};
  explicit AudioHarness(const std::string& model)
      : name{model}
      , bytes{slurp(model)}
  {
    node.prepare(
        {.rate = 48000., .input_channels = 2, .output_channels = 2, .frames = 512});
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.inputs.audio.samples = ins;
    node.inputs.audio.channels = 2;
    node.outputs.audio.samples = outs;
    node.outputs.audio.channels = 2;
  }
  void run(int ticks)
  {
    for(int t = 0; t < ticks; t++)
      node(512);
  }
};
}

// BUG-LEDGER A2: a [B,S,C,N] output (Demucs: 4 stems x stereo) was pushed as
// one mono stream of 8N samples.
TEST_CASE("Audio Processor: Param 1 picks a stem of a [B,S,C,N] output", "[onnx][audio]")
{
  AudioHarness h{SCORE_ONNX_TEST_DATA_DIR "/aux/audio_stems.onnx"};
  h.node.inputs.param1.value = 0.6f; // stem 2 of 4: mix * 3
  h.run(8);
  REQUIRE(!h.node.inputs.model.current_model_invalid);
  CHECK(std::abs(h.ol.back() - 0.3f) < 1e-4f);
  CHECK(std::abs(h.orr.back() - 0.3f) < 1e-4f);
}

// BUG-LEDGER A6: Params 2..4 did nothing, and every other input was fed float
// zeros. They now drive the model's scalar inputs.
TEST_CASE("Audio Processor: Param 2 drives a scalar input", "[onnx][audio]")
{
  AudioHarness h{SCORE_ONNX_TEST_DATA_DIR "/aux/audio_gain.onnx"};
  h.node.inputs.param2.value = 0.5f;
  h.run(8);
  REQUIRE(!h.node.inputs.model.current_model_invalid);
  CHECK(std::abs(h.ol.back() - 0.05f) < 1e-4f);
}

// BUG-LEDGER A3 / AA3: the rate came from port names, which almost never say.
TEST_CASE("Audio model rate from the file name", "[onnx][audio]")
{
  using Onnx::audio_rate_detail::rateOfName;
  CHECK(rateOfName("demucs__htdemucs_ft_vocals.onnx") == 44100.);
  CHECK(rateOfName("dtln__dtln1.onnx") == 16000.);
  CHECK(rateOfName("silero-vad-v4-16k.onnx") == 16000.);
  CHECK(rateOfName("hifigan__generator_dynamic.onnx") == 22050.);
  CHECK(rateOfName("vocos_24khz.onnx") == 24000.);
  CHECK(rateOfName("audio_processing__clap__clap_audio_laion.onnx") == 48000.);
  CHECK(rateOfName("model.onnx") == 0.);
}

// BUG-LEDGER AA2: CREPE needs frames at zero mean / unit variance; a quiet
// tone always landed on bin 246 (543.6 Hz).
TEST_CASE("Audio Analyzer: CREPE finds a quiet tone's pitch", "[onnx][audio]")
{
  const char* env = std::getenv("ONNX_TEST_MODELS");
  const std::string model = std::string(env ? env : "/mnt/win2/models/models-presets/models")
                            + "/audio-analyzer/crepe-full.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  const auto bytes = slurp(model);
  OnnxModels::AudioAnalyzer a;
  a.prepare({.rate = 16000., .input_channels = 1, .frames = 512});
  a.inputs.model.file.bytes = bytes;
  a.inputs.model.file.filename = model;
  std::vector<float> block(512);
  float* ch[1]{block.data()};
  a.inputs.audio.samples = ch;
  a.inputs.audio.channels = 1;
  for(int k = 0; k < 8; k++)
  {
    for(int i = 0; i < 512; i++)
      block[i] = 0.02f * (float)std::sin(2 * std::numbers::pi * 330. * (k * 512 + i) / 16000.);
    a(512);
  }
  REQUIRE(!a.inputs.model.current_model_invalid);
  // value1 = argmax / (bins - 1); 330.8 Hz is bin 203.
  CHECK(std::abs(a.outputs.value1.value * 359.f - 203.f) <= 2.f);
}
