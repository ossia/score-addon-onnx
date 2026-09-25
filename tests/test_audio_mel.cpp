// Audio Processor vocoders (BUG-LEDGER A5): a vocoder's [1,80,T] input was fed
// raw samples as a [1,1,1024] tensor, so ORT threw on the first block and the
// node went silent. It now gets the log-mel spectrogram of the incoming audio,
// in the style the vocoder expects (HiFi-GAN or Vocos, with the model's
// metadata), and a Vocos spectrum output is turned back into audio.
#include <tests/TestPaths.hpp>
#include <OnnxModels/AudioProcessor.hpp>

#include <Onnx/helpers/MelFrontend.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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

std::vector<float> threeTones(std::size_t n, double sr)
{
  std::vector<float> in(n);
  const double w = 2. * std::numbers::pi / sr;
  for(std::size_t i = 0; i < n; i++)
    in[i] = (float)(0.3 * std::sin(w * 440. * i) + 0.2 * std::sin(w * 3000. * i + 0.5)
                    + 0.05 * std::sin(w * 7000. * i));
  return in;
}

float worstDiff(const Onnx::MelConfig& cfg, const char* ref_file)
{
  Onnx::MelFrontend mel;
  mel.prepare(cfg, 32, /*context*/ 0);
  REQUIRE(mel.blockSize() == 32 * 256 + 1024 - 256);
  const auto in = threeTones(mel.blockSize(), cfg.rate);
  std::vector<float> out(cfg.n_mels * 32), ref(cfg.n_mels * 32), scratch;
  mel.compute(in.data(), out.data(), scratch);

  std::ifstream f(std::string(SCORE_ONNX_TEST_DATA_DIR "/audio/") + ref_file, std::ios::binary);
  REQUIRE(f.read(reinterpret_cast<char*>(ref.data()), ref.size() * sizeof(float)));
  // The DFT runs in float32: bands more than 10 nats below the loudest one
  // are its rounding noise (the reference is float64), not compared.
  const float peak = *std::max_element(ref.begin(), ref.end());
  float worst = 0.f;
  for(std::size_t i = 0; i < ref.size(); i++)
    if(ref[i] > peak - 10.f)
      worst = std::max(worst, std::abs(out[i] - ref[i]));
  return worst;
}

// Runs `signal` through a vocoder with the host at `rate`, then one more
// second of silence; returns the output.
std::vector<float> resynthesise(
    const std::string& model, double rate, const std::vector<float>& signal,
    OnnxModels::AudioMelStyle style = OnnxModels::AudioMelStyle::Auto)
{
  const auto bytes = slurp(model);
  constexpr int frames = 512;
  OnnxModels::AudioProcessor node;
  node.prepare({.rate = rate, .input_channels = 1, .output_channels = 1, .frames = frames});
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = model;
  node.inputs.mel_style.value = style;
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
  for(std::size_t t0 = 0; t0 < signal.size() + rate; t0 += frames)
  {
    for(int i = 0; i < frames; i++)
      in[i] = t0 + i < signal.size() ? signal[t0 + i] : 0.f;
    node(frames);
    all.insert(all.end(), out.begin(), out.end());
  }
  REQUIRE_FALSE(node.inputs.model.current_model_invalid);
  return all;
}

// Loud, the pitch survives, and no click where the blocks
// meet (the largest second difference of any 256-sample stretch stays within
// 2.5x of the median one).
void checkSine(const std::vector<float>& all, std::size_t length, double rate)
{
  // The last second of the sine's output: the output lags by less than a
  // second, and the silence appended after the sine comes later.
  const std::vector<float> tail(
      all.begin() + (std::ptrdiff_t)(length - rate), all.begin() + (std::ptrdiff_t)length);
  double sq = 0.;
  for(float x : tail)
    sq += x * x;
  CHECK(std::sqrt(sq / tail.size()) > 0.01);
  const double p440 = power(tail, 440., rate);
  for(double other : {220., 330., 660., 1000., 2000.})
    CHECK(p440 > 10. * power(tail, other, rate));

  std::vector<double> rough;
  for(std::size_t k = 0; k + 256 <= tail.size(); k += 256)
  {
    double m = 0;
    for(std::size_t i = k + 2; i < k + 256; i++)
      m = std::max(m, (double)std::abs(tail[i] - 2 * tail[i - 1] + tail[i - 2]));
    rough.push_back(m);
  }
  auto sorted = rough;
  std::sort(sorted.begin(), sorted.end());
  const double median = sorted[sorted.size() / 2];
  INFO("roughness median " << median << " max " << sorted.back());
  CHECK(sorted.back() < 2.5 * median);
}

std::vector<float> sine(double rate, double seconds)
{
  std::vector<float> x((std::size_t)(rate * seconds));
  for(std::size_t i = 0; i < x.size(); i++)
    x[i] = 0.4f * (float)std::sin(2. * std::numbers::pi * 440. * (double)i / rate);
  return x;
}

// 16-bit mono PCM WAV (the sherpa test voices).
std::vector<float> readWav(const std::string& path)
{
  const auto bytes = slurp(path);
  std::vector<float> x;
  for(std::size_t i = 44; i + 1 < bytes.size(); i += 2)
    x.push_back((int16_t)((uint8_t)bytes[i] | ((uint8_t)bytes[i + 1] << 8)) / 32768.f);
  return x;
}

// Log-mel distance between a signal and a vocoder's resynthesis of it, in
// nats per band, over the frames where the input is not silent, at the
// output latency that fits best (searched up to 3 s). With `unrelated`, the
// distance one second past that latency instead: the output compared with
// other speech, the score of audio that has nothing to do with the input.
double melDistance(
    const std::vector<float>& x, const std::vector<float>& y, double rate,
    bool unrelated = false)
{
  auto logmel = [&](const std::vector<float>& s) {
    Onnx::MelFrontend mel;
    const int T = (int)((s.size() - 1024) / 256);
    mel.prepare(Onnx::MelConfig::hifigan(80, rate), T, 0);
    std::vector<float> out(80 * T), scratch;
    mel.compute(s.data(), out.data(), scratch);
    return std::pair{out, T};
  };
  const auto [X, Tx] = logmel(x);
  const auto [Y, Ty] = logmel(y);
  auto distance = [&](int lag) {
    double sum = 0.;
    int n = 0;
    for(int t = 0; t < Tx && t + lag < Ty; t++)
    {
      double e = 0.;
      for(int m = 0; m < 80; m++)
        e += X[m * Tx + t];
      if(e / 80. < -6.)
        continue; // silence
      for(int m = 0; m < 80; m++)
        sum += std::abs(X[m * Tx + t] - Y[m * Ty + t + lag]);
      n += 80;
    }
    return n ? sum / n : 1e9;
  };
  double best = 1e9;
  int bestLag = 0;
  for(int lag = 0; lag < (int)(3 * rate / 256); lag++)
    if(const double d = distance(lag); d < best)
    {
      best = d;
      bestLag = lag;
    }
  return unrelated ? distance(bestLag + (int)(rate / 256)) : best;
}

std::string wild(const char* sub)
{
  return TestPaths::wild() + "/" + sub;
}
}

TEST_CASE("Mel frontend: a sine lands in its mel band", "[onnx][audio]")
{
  Onnx::MelFrontend mel;
  mel.prepare(Onnx::MelConfig::hifigan(80, 22050.), 32, 0);
  std::vector<float> in(mel.blockSize()), out(80 * 32), scratch;
  for(std::size_t i = 0; i < in.size(); i++)
    in[i] = 0.5f * (float)std::sin(2. * std::numbers::pi * 1000. * (double)i / 22050.);
  mel.compute(in.data(), out.data(), scratch);

  // The loudest band for frame 10 is the one around 1 kHz.
  int best = 0;
  for(int m = 1; m < 80; m++)
    if(out[m * 32 + 10] > out[best * 32 + 10])
      best = m;
  const double top = Onnx::MelFrontend::hzToMel(8000.);
  CHECK(Onnx::MelFrontend::melToHz(top * best / 81.) < 1000.);
  CHECK(Onnx::MelFrontend::melToHz(top * (best + 2) / 81.) > 1000.);
  // Silence is the log floor.
  std::fill(in.begin(), in.end(), 0.f);
  mel.compute(in.data(), out.data(), scratch);
  CHECK(out[0] < -11.f);
}

TEST_CASE("Mel frontend: matches the numpy references", "[onnx][audio]")
{
  CHECK(worstDiff(Onnx::MelConfig::hifigan(80, 22050.), "mel_ref_80x32.f32") < 1e-3f);
  CHECK(worstDiff(Onnx::MelConfig::vocos(100, 24000.), "mel_ref_vocos_100x32.f32") < 1e-3f);
}

TEST_CASE("Mel frontend: silence is the log floor of each style", "[onnx][audio]")
{
  // HiFi-GAN adds 1e-9 under the square root, Vocos does not: from silence,
  // Vocos' features are exactly its floor, log(1e-7). With the epsilon they
  // would sit near log(3e-5) and Vocos would resynthesise that as noise.
  for(auto cfg : {Onnx::MelConfig::hifigan(80, 22050.), Onnx::MelConfig::vocos(100, 24000.)})
  {
    Onnx::MelFrontend mel;
    mel.prepare(cfg, 4, 0);
    std::vector<float> in(mel.blockSize(), 0.f), out(cfg.n_mels * 4), scratch;
    mel.compute(in.data(), out.data(), scratch);
    for(float v : out)
      REQUIRE(v == (float)std::log(cfg.log_floor));
  }
}

TEST_CASE("Spectrum synthesis: the inverse STFT gives the signal back", "[onnx][audio]")
{
  // Analyse a signal with the same window and hop, synthesise it back.
  constexpr int n_fft = 1024, hop = 256, T = 64;
  const auto x = threeTones((T - 1) * hop + n_fft, 24000.);
  Onnx::SpectrumSynth synth;
  synth.prepare(n_fft, hop);
  const int F = n_fft / 2 + 1;
  std::vector<float> mag(F * T), c(F * T), s(F * T);
  for(int t = 0; t < T; t++)
    for(int f = 0; f < F; f++)
    {
      double re = 0, im = 0;
      for(int n = 0; n < n_fft; n++)
      {
        const double v = x[t * hop + n] * synth.window[n];
        const double a = -2. * std::numbers::pi * f * n / n_fft;
        re += v * std::cos(a);
        im += v * std::sin(a);
      }
      const double m = std::hypot(re, im);
      mag[f * T + t] = (float)m;
      c[f * T + t] = m > 0 ? (float)(re / m) : 1.f;
      s[f * T + t] = m > 0 ? (float)(im / m) : 0.f;
    }
  std::vector<float> acc, out;
  synth.push(mag.data(), c.data(), s.data(), T, 0, T, acc, out);
  REQUIRE(out.size() == (std::size_t)T * hop);
  // Once n_fft - hop samples have overlapped fully, sample i of the output is
  // sample i of the input.
  float worst = 0.f;
  for(int i = n_fft; i < T * hop; i++)
    worst = std::max(worst, std::abs(out[i] - x[i]));
  CHECK(worst < 1e-4f);
}

TEST_CASE("Audio Processor: HiFi-GAN resynthesises a sine", "[onnx][audio]")
{
  const auto model = wild("wild/hifigan__generator_dynamic.onnx");
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  REQUIRE(OnnxModels::initOnnxRuntime());
  const auto x = sine(22050., 3.);
  checkSine(resynthesise(model, 22050., x), x.size(), 22050.);
}

TEST_CASE("Audio Processor: vocoders resynthesise speech", "[onnx][audio]")
{
  const auto voice = readWav(wild(
      "sherpa/tts/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav"));
  if(voice.empty())
    SKIP("test voice not found");
  REQUIRE(OnnxModels::initOnnxRuntime());
  // Host at 24 kHz: the 22 kHz vocoders go through the resampler.
  // hifigan: waveform out; vocos 22 kHz: 80 bins, HiFi-GAN features (Matcha's
  // vocoder); vocos 24 kHz: 100 bins, Vocos features. The Vocos exports return
  // magnitude / cos / sin.
  int ran = 0;
  for(const char* path :
      {"wild/hifigan__generator_dynamic.onnx",
       "sherpa/tts/matcha-icefall-en_US-ljspeech/vocos-22khz-univ.onnx",
       "sherpa/tts/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/vocos_24khz.onnx"})
  {
    const auto model = wild(path);
    if(!std::filesystem::exists(model))
      continue;
    ran++;
    INFO(model);
    const auto out = resynthesise(model, 24000., voice);
    const double d = melDistance(voice, out, 24000.);
    const double unrelated = melDistance(voice, out, 24000., true);
    CHECK(d < 0.8);
    CHECK(d < 0.4 * unrelated);
    // The other feature style is far worse: Auto picked the right one.
    const bool vocos100 = std::string(path).find("24khz") != std::string::npos;
    const auto wrong = vocos100 ? OnnxModels::AudioMelStyle::HiFiGAN
                                : OnnxModels::AudioMelStyle::Vocos;
    CHECK(melDistance(voice, resynthesise(model, 24000., voice, wrong), 24000.) > 2. * d);
  }
  if(!ran)
    SKIP("no vocoder found under " << TestPaths::wild());
}

TEST_CASE("Audio Processor: an STFT-bin input is refused once", "[onnx][audio]")
{
  const auto model = wild("wild/unet_source_separation__second_voice_bank.best.opt2.onnx");
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

