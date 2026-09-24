#pragma once
// Log-mel spectrogram of a block of audio, for the neural vocoders the Audio
// Processor runs on a [1,n_mels,T] input, and the inverse STFT for vocoders
// that return a spectrum (Vocos exports: magnitude, cos and sin of the phase).
//
// Two feature styles cover the common vocoders:
// - HiFi-GAN (meldataset.mel_spectrogram; also MelGAN, WaveGlow, Matcha's
//   Vocos): librosa's Slaney mel scale and filterbank normalisation,
//   0 .. 8 kHz, magnitude sqrt(re² + im² + 1e-9), log(max(x, 1e-5)).
// - Vocos (vocos-mel-24khz, torchaudio MelSpectrogram): HTK mel scale, no
//   filterbank normalisation, 0 .. Nyquist, the plain magnitude,
//   log(max(x, 1e-7)). The epsilon matters: HiFi-GAN's 1e-9 under the square
//   root lifts every silent bin to ~3e-5, far above Vocos' floor, and Vocos
//   then resynthesises that floor as noise.
//
// Streaming: a block holds `frames` new frames plus `context` frames on each
// side, which the vocoder sees but whose output is dropped. A vocoder picks
// its own phase from a wide receptive field: with too little context two
// blocks settle on different phases and the waveform jumps where they meet
// (HiFi-GAN: a click every block at 8 frames, none at 16). The
// n_fft - hop samples of the last frame's window replace the reflect padding.
// Dependency-free. compute() and push() are const and run on the worker.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Onnx
{
struct MelConfig
{
  double rate = 22050.;
  int n_fft = 1024;
  int hop = 256;
  int n_mels = 80;
  double fmin = 0.;
  double fmax = 8000.;
  bool htk = false;         // HTK mel scale, else Slaney
  bool slaney_norm = true;  // area-normalised filters
  double log_floor = 1e-5;
  double mag_eps = 1e-9;    // magnitude = sqrt(re² + im² + mag_eps)

  static MelConfig hifigan(int mels, double rate)
  {
    return {.rate = rate, .n_mels = mels};
  }
  static MelConfig vocos(int mels, double rate)
  {
    return {
        .rate = rate, .n_mels = mels, .fmax = rate / 2., .htk = true,
        .slaney_norm = false, .log_floor = 1e-7, .mag_eps = 0.};
  }
};

namespace mel_detail
{
inline constexpr double pi = 3.14159265358979323846;

// Periodic Hann (torch.hann_window) and the DFT twiddles cos / -sin.
inline void tables(
    int n_fft, std::vector<float>& window, std::vector<float>& cosT,
    std::vector<float>& sinT)
{
  const int F = n_fft / 2 + 1;
  window.resize(n_fft);
  for(int i = 0; i < n_fft; ++i)
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
}
}

struct MelFrontend
{
  MelConfig cfg;
  int frames = 32;  // new frames per block
  int context = 16; // extra frames on each side

  std::vector<float> window;     // n_fft
  std::vector<float> cosT, sinT; // [bins * n_fft]
  std::vector<float> filters;    // [n_mels * bins]

  int bins() const noexcept { return cfg.n_fft / 2 + 1; }
  int blockFrames() const noexcept { return frames + 2 * context; }
  // Samples a block takes, and how many of them are new.
  int64_t blockSize() const noexcept
  {
    return (int64_t)blockFrames() * cfg.hop + (cfg.n_fft - cfg.hop);
  }
  int64_t hopSize() const noexcept { return (int64_t)frames * cfg.hop; }

  static double hzToMel(double hz, bool htk = false)
  {
    if(htk)
      return 2595. * std::log10(1. + hz / 700.);
    // Slaney: linear below 1 kHz, logarithmic above.
    constexpr double f_sp = 200. / 3., min_log_hz = 1000.;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.;
    return hz < min_log_hz ? hz / f_sp
                           : min_log_mel + std::log(hz / min_log_hz) / logstep;
  }
  static double melToHz(double mel, bool htk = false)
  {
    if(htk)
      return 700. * (std::pow(10., mel / 2595.) - 1.);
    constexpr double f_sp = 200. / 3., min_log_hz = 1000.;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.;
    return mel < min_log_mel ? mel * f_sp
                             : min_log_hz * std::exp(logstep * (mel - min_log_mel));
  }

  void prepare(const MelConfig& c, int block_frames = 32, int context_frames = 16)
  {
    cfg = c;
    frames = block_frames;
    context = context_frames;
    const int F = bins();
    mel_detail::tables(cfg.n_fft, window, cosT, sinT);

    // librosa.filters.mel / torchaudio.functional.melscale_fbanks.
    std::vector<double> pts(cfg.n_mels + 2);
    const double mlo = hzToMel(cfg.fmin, cfg.htk), mhi = hzToMel(cfg.fmax, cfg.htk);
    for(int m = 0; m < cfg.n_mels + 2; ++m)
      pts[m] = melToHz(mlo + (mhi - mlo) * m / (cfg.n_mels + 1), cfg.htk);
    filters.assign((std::size_t)cfg.n_mels * F, 0.f);
    for(int m = 0; m < cfg.n_mels; ++m)
    {
      const double lo = pts[m], mid = pts[m + 1], hi = pts[m + 2];
      const double enorm = cfg.slaney_norm ? 2. / (hi - lo) : 1.;
      for(int f = 0; f < F; ++f)
      {
        const double hz = (double)f * cfg.rate / cfg.n_fft;
        const double up = (hz - lo) / (mid - lo);
        const double down = (hi - hz) / (hi - mid);
        const double w = std::max(0., std::min(up, down));
        filters[(std::size_t)m * F + f] = (float)(w * enorm);
      }
    }
  }

  // `in` holds blockSize() samples; `out` receives [n_mels, blockFrames()].
  // `magnitude` is scratch (bins()). Const: one instance serves the worker
  // jobs that share it.
  void compute(const float* in, float* out, std::vector<float>& magnitude) const
  {
    const int F = bins();
    magnitude.resize(F);
    const int T = blockFrames();
    const int n_fft = cfg.n_fft;
    for(int t = 0; t < T; ++t)
    {
      const float* x = in + (std::size_t)t * cfg.hop;
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
        magnitude[f] = (float)std::sqrt(re * re + im * im + cfg.mag_eps);
      }
      for(int m = 0; m < cfg.n_mels; ++m)
      {
        const float* fb = &filters[(std::size_t)m * F];
        double acc = 0.;
        for(int f = 0; f < F; ++f)
          acc += (double)fb[f] * magnitude[f];
        out[(std::size_t)m * T + t] = (float)std::log(std::max(acc, cfg.log_floor));
      }
    }
  }
};

// Inverse STFT of a vocoder's spectrum output, frame by frame, overlap-added
// with the periodic Hann window and divided by the window's squared overlap
// (torch.istft). Each frame completes `hop` samples. The overlap accumulator
// is the caller's (n_fft samples), so the tables can be shared.
struct SpectrumSynth
{
  int n_fft = 1024;
  int hop = 256;
  std::vector<float> window, cosT, sinT;
  float inv_envelope = 1.f;

  void prepare(int nfft, int hop_size)
  {
    n_fft = nfft;
    hop = hop_size;
    mel_detail::tables(n_fft, window, cosT, sinT);
    // Averaged over one hop: constant for the usual n_fft / hop ratios.
    double env = 0.;
    for(int j = 0; j < hop; ++j)
      for(int i = j; i < n_fft; i += hop)
        env += (double)window[i] * window[i];
    env /= hop;
    inv_envelope = env > 0. ? (float)(1. / env) : 1.f;
  }

  // mag, c, s: [F, T] (bin-major), F = n_fft / 2 + 1; synthesises frames
  // [t0, t1) and appends (t1 - t0) * hop samples to `out`.
  void push(
      const float* mag, const float* c, const float* s, int T, int t0, int t1,
      std::vector<float>& acc, std::vector<float>& out) const
  {
    const int F = n_fft / 2 + 1;
    acc.resize(n_fft);
    std::vector<float> frame(n_fft);
    const double inv_n = 1. / n_fft;
    for(int t = t0; t < t1; ++t)
    {
      // irfft: bins 0 and N/2 once, the others twice (conjugate symmetry).
      std::fill(frame.begin(), frame.end(), 0.f);
      for(int f = 0; f < F; ++f)
      {
        const std::size_t idx = (std::size_t)f * T + t;
        const float a = mag[idx] * c[idx];
        const float b = mag[idx] * s[idx];
        const float scale = (f == 0 || f == F - 1) ? 1.f : 2.f;
        const float* ct = &cosT[(std::size_t)f * n_fft];
        const float* st = &sinT[(std::size_t)f * n_fft];
        // Re((a + ib) e^{+iθ}) = a cosθ - b sinθ, with sinT = -sinθ.
        for(int n = 0; n < n_fft; ++n)
          frame[n] += scale * (a * ct[n] + b * st[n]);
      }
      for(int n = 0; n < n_fft; ++n)
        acc[n] += (float)(frame[n] * inv_n) * window[n];
      for(int n = 0; n < hop; ++n)
        out.push_back(acc[n] * inv_envelope);
      std::copy(acc.begin() + hop, acc.end(), acc.begin());
      std::fill(acc.end() - hop, acc.end(), 0.f);
    }
  }
};
}
