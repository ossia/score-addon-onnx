// Audio Processor mel frontend (BUG-LEDGER A5): a vocoder's [1,80,T] input
// was fed raw samples as a [1,1,1024] tensor, so ORT threw on the first block
// and the node went silent. It now gets the log-mel spectrogram of the
// incoming audio, and HiFi-GAN resynthesises it.
#include <OnnxModels/AudioProcessor.hpp>

#include <Onnx/helpers/MelFrontend.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
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

double power(const std::vector<float>& x, double hz, double rate)
{
  double re = 0., im = 0.;
  for(std::size_t i = 0; i < x.size(); i++)
  {
    const double a = 2. * std::numbers::pi * hz * (double)i / rate;
    re += x[i] * std::cos(a);
    im += x[i] * std::sin(a);
  }
  return (re * re + im * im) / ((double)x.size() * x.size());
}
}

TEST_CASE("Mel frontend: a sine lands in its mel band", "[onnx][audio]")
{
  Onnx::MelFrontend mel;
  mel.prepare(80, 22050.);
  std::vector<float> in(mel.blockSize()), out(80 * mel.frames);
  for(std::size_t i = 0; i < in.size(); i++)
    in[i] = 0.5f * (float)std::sin(2. * std::numbers::pi * 1000. * (double)i / 22050.);
  mel.compute(in.data(), out.data());

  // Band centres: the loudest band for frame 10 is the one around 1 kHz.
  int best = 0;
  for(int m = 1; m < 80; m++)
    if(out[m * mel.frames + 10] > out[best * mel.frames + 10])
      best = m;
  const double lo = Onnx::MelFrontend::melToHz(Onnx::MelFrontend::hzToMel(8000.) * best / 81.);
  const double hi = Onnx::MelFrontend::melToHz(Onnx::MelFrontend::hzToMel(8000.) * (best + 2) / 81.);
  CHECK(lo < 1000.);
  CHECK(hi > 1000.);
  // Silence is the log floor.
  std::fill(in.begin(), in.end(), 0.f);
  mel.compute(in.data(), out.data());
  CHECK(out[0] < -11.f);
}

TEST_CASE("Mel frontend: matches the numpy reference", "[onnx][audio]")
{
  Onnx::MelFrontend mel;
  mel.prepare(80, 22050.);
  REQUIRE(mel.blockSize() == 32 * 256 + 1024 - 256);
  std::vector<float> in(mel.blockSize()), out(80 * 32), ref(80 * 32);
  const double w = 2. * std::numbers::pi / 22050.;
  for(std::size_t i = 0; i < in.size(); i++)
    in[i] = (float)(0.3 * std::sin(w * 440. * i) + 0.2 * std::sin(w * 3000. * i + 0.5)
                    + 0.05 * std::sin(w * 7000. * i));
  mel.compute(in.data(), out.data());

  std::ifstream f(SCORE_ONNX_TEST_DATA_DIR "/audio/mel_ref_80x32.f32", std::ios::binary);
  REQUIRE(f.read(reinterpret_cast<char*>(ref.data()), ref.size() * sizeof(float)));
  float worst = 0.f;
  for(std::size_t i = 0; i < ref.size(); i++)
    worst = std::max(worst, std::abs(out[i] - ref[i]));
  CHECK(worst < 1e-3f);
}

TEST_CASE("Audio Processor: HiFi-GAN resynthesises a sine", "[onnx][audio]")
{
  const char* env = std::getenv("ONNX_TEST_WILD_MODELS");
  const std::string model = std::string(env ? env : "/mnt/sdd1/models/wild")
                            + "/hifigan__generator_dynamic.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  REQUIRE(OnnxModels::initOnnxRuntime());
  const auto bytes = slurp(model);
  constexpr int frames = 512;
  constexpr double rate = 22050.;

  OnnxModels::AudioProcessor node;
  node.prepare({.rate = rate, .input_channels = 1, .output_channels = 1, .frames = frames});
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = model;
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
  for(int t0 = 0; t0 < 3 * 22050; t0 += frames)
  {
    for(int i = 0; i < frames; i++)
      in[i] = 0.4f * (float)std::sin(2. * std::numbers::pi * 440. * (t0 + i) / rate);
    node(frames);
    all.insert(all.end(), out.begin(), out.end());
  }
  REQUIRE_FALSE(node.inputs.model.current_model_invalid);

  // The last second: loud, and the pitch survives.
  const std::vector<float> tail(all.end() - 22050, all.end());
  double sq = 0.;
  for(float x : tail)
    sq += x * x;
  CHECK(std::sqrt(sq / tail.size()) > 0.01);
  const double p440 = power(tail, 440., rate);
  for(double other : {220., 330., 660., 1000., 2000.})
    CHECK(p440 > 10. * power(tail, other, rate));
}

TEST_CASE("Audio Processor: an STFT-bin input is refused once", "[onnx][audio]")
{
  const char* env = std::getenv("ONNX_TEST_WILD_MODELS");
  const std::string model
      = std::string(env ? env : "/mnt/sdd1/models/wild")
        + "/unet_source_separation__second_voice_bank.best.opt2.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  REQUIRE(OnnxModels::initOnnxRuntime());
  const auto bytes = slurp(model);

  // input1/input2 [1,257,frames]: magnitude and phase, not mel bins.
  OnnxModels::AudioProcessor node;
  node.prepare({.rate = 16000., .input_channels = 1, .output_channels = 1, .frames = 256});
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = model;
  std::vector<float> in(256, 0.1f), out(256, 1.f);
  float* ins[1]{in.data()};
  float* outs[1]{out.data()};
  node.inputs.audio.samples = ins;
  node.inputs.audio.channels = 1;
  node.outputs.audio.samples = outs;
  node.outputs.audio.channels = 1;
  node(256);
  CHECK(node.inputs.model.current_model_invalid);
}
