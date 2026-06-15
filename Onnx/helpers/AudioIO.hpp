#pragma once
// Audio adapter for the generic ONNX audio nodes (AudioProcessor /
// AudioAnalyzer). The host runs at an arbitrary sample-rate and block size; most
// audio models want a *fixed* sample-rate and a *fixed* number of samples per
// inference (e.g. CREPE 1024 @ 16k, Silero-VAD 512 @ 16k, EnCodec @ 24k,
// DeepFilterNet/RAVE @ 48k, Demucs @ 44.1k). This file provides:
//
//   - ResamplerLin    : streaming linear resampler (host SR -> model SR and back)
//   - AudioRing        : a fixed-capacity ring buffer (block accumulation / hop)
//   - WaveformIO       : ring + resampler glue that fires the model at its
//                        block/hop and builds the [1,C,N] / [1,N] waveform tensor
//   - Stft / Istft     : optional host STFT/iSTFT (since ONNX has no complex
//                        dtype, magnitude [1,F,T] or real/imag [1,2,F,T])
//
// Everything here is dependency-free (only <vector>/<cmath>/<cstdint>/<cstddef>
// and the dependency-free TensorType.hpp) so the pure-logic parts compile and
// run in a standalone test. REAL-TIME SAFE: all buffers are preallocated by
// prepare(); the steady-state push/pop/resample paths never allocate.

#include <Onnx/helpers/TensorType.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Onnx
{

// ---------------------------------------------------------------------------
// Streaming linear resampler. One instance per channel direction. Keeps the
// last input sample across calls so block boundaries don't click. ratio =
// out_rate / in_rate. Output count is not known a-priori (it depends on the
// fractional phase carried over), so process() appends to a caller-owned vector
// that is reserve()d once in prepare().
// ---------------------------------------------------------------------------
struct ResamplerLin
{
  double ratio = 1.0; // out_rate / in_rate
  double pos = 0.0;   // fractional read position into the (prev|cur) stream
  float prev = 0.f;   // last sample of the previous block
  bool primed = false;

  void prepare(double in_rate, double out_rate) noexcept
  {
    ratio = (in_rate > 0.0) ? (out_rate / in_rate) : 1.0;
    reset();
  }

  void reset() noexcept
  {
    pos = 0.0;
    prev = 0.f;
    primed = false;
  }

  bool passthrough() const noexcept
  {
    return std::abs(ratio - 1.0) < 1e-9;
  }

  // Resample `in` (n samples) into `out` (appended). Returns number appended.
  // `out` must have spare capacity (reserved in prepare()); we never shrink it.
  std::size_t process(const float* in, std::size_t n, std::vector<float>& out)
  {
    if(n == 0)
      return 0;
    if(passthrough())
    {
      out.insert(out.end(), in, in + n);
      prev = in[n - 1];
      primed = true;
      return n;
    }

    // The virtual input stream is [prev, in[0], in[1], ... in[n-1]], indexed
    // from -1. We advance `pos` (in input-sample units) by 1/ratio per output
    // sample and read while the right neighbour is still inside this block.
    const double step = 1.0 / ratio;
    const std::size_t before = out.size();
    if(!primed)
    {
      // First-ever block: start exactly on in[0], no carried `prev`.
      pos = 0.0;
      while(pos < (double)(n - 1) + 1e-12)
      {
        const auto i0 = (std::ptrdiff_t)std::floor(pos);
        const double frac = pos - (double)i0;
        const float a = in[i0];
        const float b = (i0 + 1 < (std::ptrdiff_t)n) ? in[i0 + 1] : in[n - 1];
        out.push_back((float)(a + (b - a) * frac));
        pos += step;
      }
      // Next block uses index -1 == this block's in[n-1] (== prev), so the
      // carried phase shifts by n (block-2 index = absolute pos - n).
      pos -= (double)n;
    }
    else
    {
      // pos is measured from index -1 (i.e. `prev`). Read until we'd need a
      // sample at or past n (which belongs to the next block).
      while(pos < (double)(n - 1) + 1e-12)
      {
        const auto idx = pos; // -1 == prev, 0 == in[0], ...
        const auto i0 = (std::ptrdiff_t)std::floor(idx);
        const double frac = idx - (double)i0;
        const float a = (i0 < 0) ? prev : in[i0];
        const float b = (i0 + 1 < 0)        ? prev
                        : (i0 + 1 < (std::ptrdiff_t)n) ? in[i0 + 1]
                                                       : in[n - 1];
        out.push_back((float)(a + (b - a) * frac));
        pos += step;
      }
      pos -= (double)n; // shift origin: next block's prev becomes index -1
    }
    prev = in[n - 1];
    primed = true;
    return out.size() - before;
  }
};

// ---------------------------------------------------------------------------
// Fixed-capacity mono ring buffer. Preallocated; push/pop are allocation-free.
// Used to accumulate resampled input until a full model block is available and
// to queue resampled model output for the host to drain.
// ---------------------------------------------------------------------------
struct AudioRing
{
  std::vector<float> buf;
  std::size_t head = 0; // next write
  std::size_t tail = 0; // next read
  std::size_t count = 0;

  void prepare(std::size_t capacity)
  {
    buf.assign(capacity == 0 ? 1 : capacity, 0.f);
    clear();
  }

  void clear() noexcept
  {
    head = tail = count = 0;
  }

  std::size_t capacity() const noexcept { return buf.size(); }
  std::size_t size() const noexcept { return count; }
  bool empty() const noexcept { return count == 0; }

  // Push n samples; if it would overflow, the oldest samples are dropped
  // (latest-wins; an overrun means the model can't keep up — never reallocate).
  void push(const float* p, std::size_t n) noexcept
  {
    const std::size_t cap = buf.size();
    if(n >= cap)
    {
      // Keep only the most recent `cap` samples.
      p += (n - cap);
      n = cap;
      clear();
    }
    for(std::size_t i = 0; i < n; ++i)
    {
      buf[head] = p[i];
      head = (head + 1 == cap) ? 0 : head + 1;
    }
    count += n;
    if(count > cap)
    {
      const std::size_t drop = count - cap;
      tail = (tail + drop) % cap;
      count = cap;
    }
  }

  // Pop exactly n samples into out (must have n available). Returns false if
  // not enough buffered.
  bool pop(float* out, std::size_t n) noexcept
  {
    if(count < n)
      return false;
    const std::size_t cap = buf.size();
    for(std::size_t i = 0; i < n; ++i)
    {
      out[i] = buf[tail];
      tail = (tail + 1 == cap) ? 0 : tail + 1;
    }
    count -= n;
    return true;
  }

  // Copy the oldest n samples into out without consuming them (for overlapping
  // hop windows). Returns false if not enough buffered.
  bool peek(float* out, std::size_t n) const noexcept
  {
    if(count < n)
      return false;
    const std::size_t cap = buf.size();
    std::size_t t = tail;
    for(std::size_t i = 0; i < n; ++i)
    {
      out[i] = buf[t];
      t = (t + 1 == cap) ? 0 : t + 1;
    }
    return true;
  }

  // Discard the oldest n samples (advance read head); for hop < block_size.
  void drop(std::size_t n) noexcept
  {
    n = (n > count) ? count : n;
    tail = (tail + n) % (buf.empty() ? 1 : buf.size());
    count -= n;
  }
};

// ---------------------------------------------------------------------------
// Waveform tensor descriptor. Most audio models take one of:
//   [1, 1, N]  mono with explicit channel
//   [1, 2, N]  stereo
//   [1, N]     mono without channel dim
//   [N]        bare
// We detect the wanted channel count + layout once from the model's input
// shape; build() interleaves/deinterleaves accordingly.
// ---------------------------------------------------------------------------
enum class WaveLayout : uint8_t
{
  BNC1, // [1,1,N] or [1,C,N]
  BN,   // [1,N]
  N,    // [N]
};

struct WaveformShape
{
  WaveLayout layout = WaveLayout::BNC1;
  int channels = 1; // model-side channel count
  int64_t block = 0; // N (samples per inference); <=0 means dynamic

  // Derive from a model input port shape (positive dims; -1 == dynamic).
  static WaveformShape fromInputShape(const std::vector<int64_t>& s) noexcept
  {
    WaveformShape w;
    switch(s.size())
    {
      case 3: // [B,C,N]
        w.layout = WaveLayout::BNC1;
        w.channels = (s[1] == 2) ? 2 : 1;
        w.block = s[2];
        break;
      case 2: // [B,N]
        w.layout = WaveLayout::BN;
        w.channels = 1;
        w.block = s[1];
        break;
      case 1: // [N]
        w.layout = WaveLayout::N;
        w.channels = 1;
        w.block = s[0];
        break;
      default:
        w.layout = WaveLayout::BNC1;
        w.channels = 1;
        w.block = s.empty() ? 0 : s.back();
        break;
    }
    return w;
  }

  std::vector<int64_t> tensorShape(int64_t n) const
  {
    switch(layout)
    {
      case WaveLayout::BNC1: return {1, channels, n};
      case WaveLayout::BN:   return {1, n};
      case WaveLayout::N:    return {n};
    }
    return {1, channels, n};
  }
};

// ---------------------------------------------------------------------------
// Full input adapter: per-channel resample (host SR -> model SR) into per-
// channel rings; pull a fixed model block when enough is buffered. RT-safe:
// prepare() sizes everything for the worst case; ready()/fill() never allocate.
// ---------------------------------------------------------------------------
struct WaveformInput
{
  WaveformShape shape;
  double host_rate = 48000.0;
  double model_rate = 48000.0;
  int64_t block = 0; // resolved model block (fixed when shape.block>0)
  int64_t hop = 0;   // <= block; defaults to block (no overlap)

  std::vector<ResamplerLin> resamplers; // one per host channel
  std::vector<AudioRing> rings;         // one per model channel (post-mix)
  std::vector<std::vector<float>> rs_scratch; // resample output staging

  void prepare(
      const WaveformShape& s, double in_rate, double out_rate,
      int64_t fixed_block, int64_t hop_size, int host_channels,
      std::size_t max_host_frames)
  {
    shape = s;
    host_rate = in_rate;
    model_rate = out_rate;
    block = fixed_block > 0 ? fixed_block : s.block;
    if(block <= 0)
      block = 1024; // sane default for dynamic-N models
    hop = (hop_size > 0 && hop_size <= block) ? hop_size : block;

    const int mc = s.channels;
    const int hc = host_channels > 0 ? host_channels : 1;

    resamplers.assign(hc, {});
    for(auto& r : resamplers)
      r.prepare(in_rate, out_rate);

    // A host block of `max_host_frames` becomes up to ceil(frames*ratio)+1
    // model-rate samples; ring must hold a full model block plus that margin.
    const double ratio = (in_rate > 0) ? out_rate / in_rate : 1.0;
    const std::size_t per_block
        = (std::size_t)std::ceil((double)max_host_frames * ratio) + 2;
    const std::size_t cap = (std::size_t)block + per_block + 2;

    rings.assign(mc, {});
    for(auto& rg : rings)
      rg.prepare(cap);

    rs_scratch.assign(hc, {});
    for(auto& v : rs_scratch)
    {
      v.clear();
      v.reserve(per_block + 4);
    }
  }

  void reset() noexcept
  {
    for(auto& r : resamplers)
      r.reset();
    for(auto& rg : rings)
      rg.clear();
  }

  // Feed one host block. `chans[c]` points at host_frames samples for channel c.
  // Host channels are mixed/duplicated to the model channel count.
  void push(const float* const* chans, int host_channels, std::size_t frames)
  {
    const int hc = host_channels;
    for(int c = 0; c < hc && c < (int)resamplers.size(); ++c)
    {
      rs_scratch[c].clear();
      resamplers[c].process(chans[c], frames, rs_scratch[c]);
    }
    // Only the first min(hc, resamplers.size()) scratch lanes were filled
    // above; pick from those so a host with MORE channels than we prepared
    // resamplers for can't index past rs_scratch.
    const int filled = std::min<int>(hc, (int)rs_scratch.size());
    if(filled <= 0)
      return;
    const int mc = (int)rings.size();
    for(int m = 0; m < mc; ++m)
    {
      // Fewer-or-equal model channels: take the matching host channel (extra
      // host channels are simply ignored). More model channels: duplicate the
      // matching host channel, falling back to ch0 for the upmixed surplus.
      const int src = (m < filled) ? m : 0;
      rings[m].push(rs_scratch[src].data(), rs_scratch[src].size());
    }
  }

  bool ready() const noexcept
  {
    return !rings.empty() && rings[0].size() >= (std::size_t)block;
  }

  // Build the interleaved/planar waveform buffer for the tensor. Layout is
  // planar [c0_0..c0_{N-1}, c1_0..] (matches [1,C,N]). Consumes `hop` samples,
  // peeks `block` (so overlapping windows reuse samples). Returns N written.
  int64_t fill(std::vector<float>& out)
  {
    if(!ready())
      return 0;
    const int mc = (int)rings.size();
    out.resize((std::size_t)mc * block);
    for(int m = 0; m < mc; ++m)
      rings[m].peek(out.data() + (std::size_t)m * block, (std::size_t)block);
    for(int m = 0; m < mc; ++m)
      rings[m].drop((std::size_t)hop);
    return block;
  }
};

// ---------------------------------------------------------------------------
// Output adapter: model block (model SR) -> resample to host SR -> ring; the
// host drains `frames` per call. RT-safe after prepare().
// ---------------------------------------------------------------------------
struct WaveformOutput
{
  double model_rate = 48000.0;
  double host_rate = 48000.0;
  int channels = 1;
  std::vector<ResamplerLin> resamplers; // one per channel (model->host)
  std::vector<AudioRing> rings;         // one per channel (host rate)
  std::vector<std::vector<float>> rs_scratch;

  void prepare(
      int chans, double in_rate, double out_rate, int64_t model_block,
      std::size_t max_host_frames)
  {
    channels = chans > 0 ? chans : 1;
    model_rate = in_rate;
    host_rate = out_rate;
    resamplers.assign(channels, {});
    for(auto& r : resamplers)
      r.prepare(in_rate, out_rate);

    const double ratio = (in_rate > 0) ? out_rate / in_rate : 1.0;
    const std::size_t per_block
        = (std::size_t)std::ceil((double)(model_block) * ratio) + 2;
    const std::size_t cap = per_block + max_host_frames + 2;
    rings.assign(channels, {});
    for(auto& rg : rings)
      rg.prepare(cap);
    rs_scratch.assign(channels, {});
    for(auto& v : rs_scratch)
    {
      v.clear();
      v.reserve(per_block + 4);
    }
  }

  void reset() noexcept
  {
    for(auto& r : resamplers)
      r.reset();
    for(auto& rg : rings)
      rg.clear();
  }

  // Push a model-rate planar block ([c0..][c1..]) of `n` frames per channel.
  void push(const float* planar, int chans, int64_t n)
  {
    // Bound by what prepare() actually sized: a node that routes a non-audio
    // model here (or pushes before prepare) would otherwise index empty
    // resampler/ring vectors.
    int nc = std::min(chans, channels);
    nc = std::min(nc, (int)resamplers.size());
    nc = std::min(nc, (int)rs_scratch.size());
    nc = std::min(nc, (int)rings.size());
    for(int c = 0; c < nc; ++c)
    {
      rs_scratch[c].clear();
      resamplers[c].process(planar + (std::size_t)c * n, (std::size_t)n,
                            rs_scratch[c]);
      rings[c].push(rs_scratch[c].data(), rs_scratch[c].size());
    }
  }

  // Drain `frames` into the host channels; missing samples are zero-filled.
  void pull(float* const* chans, int host_channels, std::size_t frames)
  {
    for(int c = 0; c < host_channels; ++c)
    {
      const int src = (c < channels) ? c : (channels - 1);
      if(src < 0 || src >= (int)rings.size() || !rings[src].pop(chans[c], frames))
      {
        for(std::size_t i = 0; i < frames; ++i)
          chans[c][i] = 0.f;
      }
    }
  }

  std::size_t available() const noexcept
  {
    return rings.empty() ? 0 : rings[0].size();
  }
};

// ---------------------------------------------------------------------------
// Optional host STFT / iSTFT. ONNX has no complex dtype, so spectrogram models
// take either magnitude [1,F,T] or real/imag interleaved [1,2,F,T]. We provide
// a naive (O(F*N)) DFT good enough for the small n_fft used by most models;
// preallocated window + work buffers keep the steady state allocation-free.
// ---------------------------------------------------------------------------
struct Stft
{
  int n_fft = 512;
  int hop = 128;
  std::vector<float> window; // length n_fft
  std::vector<float> cosT, sinT; // [F * n_fft] precomputed twiddles

  void prepare(int nfft, int hop_size)
  {
    n_fft = nfft > 0 ? nfft : 512;
    hop = hop_size > 0 ? hop_size : n_fft / 4;
    const int F = n_fft / 2 + 1;
    window.assign(n_fft, 0.f);
    for(int i = 0; i < n_fft; ++i) // Hann
      window[i]
          = 0.5f * (1.f - std::cos(2.f * 3.14159265358979323846f * i
                                   / (float)(n_fft - 1)));
    cosT.assign((std::size_t)F * n_fft, 0.f);
    sinT.assign((std::size_t)F * n_fft, 0.f);
    for(int f = 0; f < F; ++f)
      for(int n = 0; n < n_fft; ++n)
      {
        const double ang
            = -2.0 * 3.14159265358979323846 * f * n / (double)n_fft;
        cosT[(std::size_t)f * n_fft + n] = (float)std::cos(ang);
        sinT[(std::size_t)f * n_fft + n] = (float)std::sin(ang);
      }
  }

  int bins() const noexcept { return n_fft / 2 + 1; }
  int frames(std::size_t n) const noexcept
  {
    return (n < (std::size_t)n_fft) ? 0
                                    : 1 + (int)((n - n_fft) / hop);
  }

  // Forward STFT of `in` (n mono samples). Outputs magnitude [F,T] (row-major,
  // bin-major) if `mag` non-null, and/or real/imag [2,F,T] if re/im non-null.
  // Caller sizes the outputs via bins()/frames(). RT-safe (no allocation).
  void forward(
      const float* in, std::size_t n, float* mag, float* re, float* im) const
  {
    const int F = bins();
    const int T = frames(n);
    for(int t = 0; t < T; ++t)
    {
      const std::size_t off = (std::size_t)t * hop;
      for(int f = 0; f < F; ++f)
      {
        double rr = 0.0, ii = 0.0;
        const float* ct = &cosT[(std::size_t)f * n_fft];
        const float* st = &sinT[(std::size_t)f * n_fft];
        for(int k = 0; k < n_fft; ++k)
        {
          const float s = in[off + k] * window[k];
          rr += s * ct[k];
          ii += s * st[k];
        }
        const std::size_t idx = (std::size_t)f * T + t;
        if(re)
          re[idx] = (float)rr;
        if(im)
          im[idx] = (float)ii;
        if(mag)
          mag[idx] = (float)std::sqrt(rr * rr + ii * ii);
      }
    }
  }
};

// Inverse STFT (overlap-add) from real/imag [F,T]. Reconstructs into `out`
// (length >= (T-1)*hop + n_fft, pre-zeroed by caller). Uses the same Hann
// window for synthesis (COLA approx). RT-safe.
struct Istft
{
  int n_fft = 512;
  int hop = 128;
  std::vector<float> window;
  std::vector<float> cosT, sinT; // [F * n_fft]
  std::vector<float> norm;       // window^2 overlap normalisation per sample

  void prepare(int nfft, int hop_size, std::size_t max_out)
  {
    n_fft = nfft > 0 ? nfft : 512;
    hop = hop_size > 0 ? hop_size : n_fft / 4;
    const int F = n_fft / 2 + 1;
    window.assign(n_fft, 0.f);
    for(int i = 0; i < n_fft; ++i)
      window[i]
          = 0.5f * (1.f - std::cos(2.f * 3.14159265358979323846f * i
                                   / (float)(n_fft - 1)));
    cosT.assign((std::size_t)F * n_fft, 0.f);
    sinT.assign((std::size_t)F * n_fft, 0.f);
    for(int f = 0; f < F; ++f)
      for(int nn = 0; nn < n_fft; ++nn)
      {
        const double ang
            = 2.0 * 3.14159265358979323846 * f * nn / (double)n_fft;
        cosT[(std::size_t)f * n_fft + nn] = (float)std::cos(ang);
        sinT[(std::size_t)f * n_fft + nn] = (float)std::sin(ang);
      }
    norm.assign(max_out, 0.f);
  }

  std::size_t outLength(int T) const noexcept
  {
    return (T <= 0) ? 0 : (std::size_t)(T - 1) * hop + n_fft;
  }

  // out and norm scratch must be >= outLength(T) and pre-zeroed.
  void inverse(const float* re, const float* im, int F, int T, float* out)
  {
    const std::size_t len = outLength(T);
    for(std::size_t i = 0; i < len && i < norm.size(); ++i)
      norm[i] = 0.f;
    for(int t = 0; t < T; ++t)
    {
      const std::size_t off = (std::size_t)t * hop;
      for(int k = 0; k < n_fft; ++k)
      {
        double acc = 0.0;
        for(int f = 0; f < F; ++f)
        {
          const std::size_t idx = (std::size_t)f * T + t;
          const float c = cosT[(std::size_t)f * n_fft + k];
          const float s = sinT[(std::size_t)f * n_fft + k];
          // real part of inverse DFT with Hermitian symmetry folded into F bins
          const double scale = (f == 0 || f == F - 1) ? 1.0 : 2.0;
          acc += scale * (re[idx] * c - im[idx] * s);
        }
        acc /= (double)n_fft;
        const float w = window[k];
        out[off + k] += (float)acc * w;
        if(off + k < norm.size())
          norm[off + k] += w * w;
      }
    }
    for(std::size_t i = 0; i < len && i < norm.size(); ++i)
      if(norm[i] > 1e-8f)
        out[i] /= norm[i];
  }
};

} // namespace Onnx
