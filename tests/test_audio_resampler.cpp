// The resampler between the host rate and an audio model's rate: flat in its
// passband, no aliasing going down, no imaging going up, and the same output
// whatever the host block size.
#include <Onnx/helpers/AudioIO.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace
{
std::vector<float> sine(double hz, double rate, std::size_t n, float amp = 0.5f)
{
  std::vector<float> x(n);
  for(std::size_t i = 0; i < n; ++i)
    x[i] = amp * (float)std::sin(2. * std::numbers::pi * hz * (double)i / rate);
  return x;
}

std::vector<float> stream(double in_rate, double out_rate, const std::vector<float>& in, std::size_t block)
{
  Onnx::Resampler r;
  r.prepare(in_rate, out_rate, block);
  std::vector<float> out;
  out.reserve((std::size_t)((double)in.size() * out_rate / in_rate) + 64);
  for(std::size_t i = 0; i < in.size(); i += block)
    r.process(in.data() + i, std::min(block, in.size() - i), out);
  return out;
}

float rms(const std::vector<float>& x, std::size_t from, std::size_t to)
{
  double acc = 0.;
  for(std::size_t i = from; i < to; ++i)
    acc += (double)x[i] * x[i];
  return (float)std::sqrt(acc / (double)(to - from));
}
}

TEST_CASE("Resampler: same rate passes the samples through", "[onnx][audio][resampler]")
{
  const auto in = sine(440., 48000., 1000);
  CHECK(stream(48000., 48000., in, 64) == in);
}

TEST_CASE("Resampler: a constant stays constant", "[onnx][audio][resampler]")
{
  for(auto [a, b] : {std::pair{48000., 16000.}, {16000., 48000.}, {44100., 48000.}, {48000., 22050.}})
  {
    const std::vector<float> in(20000, 0.25f);
    const auto out = stream(a, b, in, 256);
    REQUIRE(out.size() > 1000);
    // Past the kernel's latency and before the end.
    for(std::size_t i = 200; i < out.size() - 200; ++i)
      REQUIRE(std::abs(out[i] - 0.25f) < 1e-4f);
  }
}

TEST_CASE("Resampler: no aliasing when going down", "[onnx][audio][resampler]")
{
  // 12 kHz is above the 8 kHz Nyquist of 16 kHz: it must be gone, not folded
  // down to 4 kHz.
  const auto in = sine(12000., 48000., 48000);
  const auto out = stream(48000., 16000., in, 512);
  REQUIRE(out.size() > 15000);
  CHECK(rms(out, 1000, out.size() - 1000) < 1e-3f);

  // 1 kHz passes at its level (0.5 / sqrt(2)).
  const auto low = stream(48000., 16000., sine(1000., 48000., 48000), 512);
  CHECK(std::abs(rms(low, 1000, low.size() - 1000) - 0.5f / std::numbers::sqrt2_v<float>)
        < 2e-3f);
}

TEST_CASE("Resampler: no images when going up", "[onnx][audio][resampler]")
{
  // A 22.05 kHz signal at 48 kHz: a 440 Hz tone must come out as the same
  // tone, sample for sample.
  const auto in = sine(440., 22050., 22050);
  const auto out = stream(22050., 48000., in, 256);
  REQUIRE(out.size() > 40000);
  // Output i is the input's value at i / 48000 s: the streaming latency
  // delays when it is produced, not where it is.
  float err = 0.f;
  for(std::size_t i = 2000; i < out.size() - 2000; ++i)
  {
    const double t = (double)i / 48000.;
    err = std::max(err, std::abs(out[i] - 0.5f * (float)std::sin(2. * std::numbers::pi * 440. * t)));
  }
  CHECK(err < 2e-3f);
}

TEST_CASE("Resampler: the host block size does not change the output", "[onnx][audio][resampler]")
{
  const auto in = sine(3000., 44100., 9000);
  const auto ref = stream(44100., 48000., in, 9000);
  for(std::size_t block : {1u, 7u, 64u, 441u})
  {
    const auto out = stream(44100., 48000., in, block);
    REQUIRE(out.size() == ref.size());
    for(std::size_t i = 0; i < out.size(); ++i)
      REQUIRE(std::abs(out[i] - ref[i]) < 1e-6f);
  }
}

TEST_CASE("Resampler: a whole signal keeps its length and alignment", "[onnx][audio][resampler]")
{
  const auto in = sine(440., 16000., 1600);
  std::vector<float> out;
  Onnx::Resampler::resampleWhole(in.data(), in.size(), 3., out);
  REQUIRE(out.size() == 4800);
  float err = 0.f;
  for(std::size_t i = 300; i < out.size() - 300; ++i)
    err = std::max(
        err, std::abs(out[i] - 0.5f * (float)std::sin(2. * std::numbers::pi * 440. * (double)i / 48000.)));
  CHECK(err < 2e-3f);

  Onnx::Resampler::resampleWhole(in.data(), in.size(), 48000. / 22050., out);
  CHECK(out.size() == (std::size_t)std::llround(1600. * 48000. / 22050.));
}
