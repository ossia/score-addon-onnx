#pragma once
// Log-mel spectrogram of a block of audio, for the neural vocoders the Audio
// Processor runs on a [1,n_mels,T] input (HiFi-GAN, MelGAN, WaveGlow). It
// follows HiFi-GAN's meldataset.mel_spectrogram: a periodic Hann window,
// magnitude sqrt(re² + im² + 1e-9), librosa's Slaney mel filterbank, then
// log(max(x, 1e-5)). Streaming replaces its reflect padding: a block holds
// T * hop new samples after the n_fft - hop samples that precede them, which
// gives exactly T frames. Dependency-free; allocation-free after prepare().
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Onnx
{
struct MelFrontend
{
  int n_mels = 80;
  int n_fft = 1024;
  int hop = 256;
  int frames = 32; // T per block
  double rate = 22050.;

  std::vector<float> window;       // n_fft
  std::vector<float> cosT, sinT;   // [bins * n_fft]
  std::vector<float> filters;      // [n_mels * bins]
  std::vector<float> magnitude;    // bins, one frame

  int bins() const noexcept { return n_fft / 2 + 1; }
  // Samples a block takes, and how many of them are new.
  int64_t blockSize() const noexcept { return (int64_t)frames * hop + (n_fft - hop); }
  int64_t hopSize() const noexcept { return (int64_t)frames * hop; }

  static double hzToMel(double hz)
  {
    // Slaney: linear below 1 kHz, logarithmic above.
    constexpr double f_sp = 200. / 3., min_log_hz = 1000.;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.;
    return hz < min_log_hz ? hz / f_sp
                           : min_log_mel + std::log(hz / min_log_hz) / logstep;
  }
  static double melToHz(double mel)
  {
    constexpr double f_sp = 200. / 3., min_log_hz = 1000.;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.;
    return mel < min_log_mel ? mel * f_sp
                             : min_log_hz * std::exp(logstep * (mel - min_log_mel));
  }

  void prepare(
      int mels, double sample_rate, int nfft = 1024, int hop_size = 256,
      double fmin = 0., double fmax = 8000., int block_frames = 32)
  {
    n_mels = mels;
    rate = sample_rate;
    n_fft = nfft;
    hop = hop_size;
    frames = block_frames;
    const int F = bins();
    constexpr double pi = 3.14159265358979323846;

    window.resize(n_fft);
    for(int i = 0; i < n_fft; ++i) // periodic, as torch.hann_window
      window[i] = (float)(0.5 - 0.5 * std::cos(2. * pi * i / n_fft));

    cosT.resize((std::size_t)F * n_fft);
    sinT.resize((std::size_t)F * n_fft);
    for(int f = 0; f < F; ++f)
      for(int n = 0; n < n_fft; ++n)
      {
        const double ang = -2. * pi * (double)((int64_t)f * n % n_fft) / n_fft;
        cosT[(std::size_t)f * n_fft + n] = (float)std::cos(ang);
        sinT[(std::size_t)f * n_fft + n] = (float)std::sin(ang);
      }

    // librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax), norm="slaney".
    std::vector<double> pts(n_mels + 2);
    const double mlo = hzToMel(fmin), mhi = hzToMel(fmax);
    for(int m = 0; m < n_mels + 2; ++m)
      pts[m] = melToHz(mlo + (mhi - mlo) * m / (n_mels + 1));
    filters.assign((std::size_t)n_mels * F, 0.f);
    for(int m = 0; m < n_mels; ++m)
    {
      const double lo = pts[m], mid = pts[m + 1], hi = pts[m + 2];
      const double enorm = 2. / (hi - lo);
      for(int f = 0; f < F; ++f)
      {
        const double hz = (double)f * sample_rate / n_fft;
        const double up = (hz - lo) / (mid - lo);
        const double down = (hi - hz) / (hi - mid);
        const double w = std::max(0., std::min(up, down));
        filters[(std::size_t)m * F + f] = (float)(w * enorm);
      }
    }
    magnitude.resize(F);
  }

  // `in` holds blockSize() samples; `out` receives [n_mels, frames].
  void compute(const float* in, float* out)
  {
    const int F = bins();
    for(int t = 0; t < frames; ++t)
    {
      const float* x = in + (std::size_t)t * hop;
      for(int f = 0; f < F; ++f)
      {
        double re = 0., im = 0.;
        const float* ct = &cosT[(std::size_t)f * n_fft];
        const float* st = &sinT[(std::size_t)f * n_fft];
        for(int k = 0; k < n_fft; ++k)
        {
          const double s = (double)x[k] * window[k];
          re += s * ct[k];
          im += s * st[k];
        }
        magnitude[f] = (float)std::sqrt(re * re + im * im + 1e-9);
      }
      for(int m = 0; m < n_mels; ++m)
      {
        const float* fb = &filters[(std::size_t)m * F];
        double acc = 0.;
        for(int f = 0; f < F; ++f)
          acc += (double)fb[f] * magnitude[f];
        out[(std::size_t)m * frames + t] = (float)std::log(std::max(acc, 1e-5));
      }
    }
  }
};
}
