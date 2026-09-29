#pragma once
// Audio adapter for the generic ONNX audio nodes (AudioProcessor /
// AudioAnalyzer). The host runs at an arbitrary sample-rate and block size; most
// audio models want a *fixed* sample-rate and a *fixed* number of samples per
// inference (e.g. CREPE 1024 @ 16k, Silero-VAD 512 @ 16k, EnCodec @ 24k,
// DeepFilterNet/RAVE @ 48k, Demucs @ 44.1k). This file provides:
//
//   - Resampler        : band-limited resampler (host SR -> model SR and back)
//   - AudioRing        : a fixed-capacity ring buffer (block accumulation / hop)
//   - WaveformIO       : ring + resampler glue that fires the model at its
//                        block/hop and builds the [1,C,N] / [1,N] waveform tensor
//
// Everything here is dependency-free (only <vector>/<cmath>/<cstdint>/<cstddef>
// and the dependency-free TensorType.hpp) so the pure-logic parts compile and
// run in a standalone test. REAL-TIME SAFE: all buffers are preallocated by
// prepare(); the steady-state push/pop/resample paths never allocate.

#include <Onnx/helpers/TensorType.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Onnx
{

// ---------------------------------------------------------------------------
// Band-limited resampler: a Kaiser-windowed sinc, its cutoff below the lower
// of the two Nyquist frequencies (a linear interpolation would alias
// everything above the model's Nyquist into its band when going down, and
// image the model's spectrum when going up). One instance per channel and
// direction; ratio = out_rate / in_rate.
//
// Streaming: process() keeps the last input samples across blocks. Output k
// is the input at time k / out_rate, produced once the `half` input samples
// after that time have arrived (the latency). The output count of a
// call depends on the fractional phase carried over, so process() appends to
// a caller-owned vector reserved once. resampleWhole() converts a complete
// signal instead: no latency, exactly round(n * ratio) samples.
// ---------------------------------------------------------------------------
namespace resampler_detail
{
inline constexpr int zeros = 16;      // zero crossings on each side at cutoff 1
inline constexpr int resolution = 512; // kernel table points per zero crossing
inline constexpr double beta = 8.;    // Kaiser: ~80 dB stopband

inline double besselI0(double x) noexcept
{
  double sum = 1., term = 1.;
  for(int k = 1; k < 64 && term > 1e-12 * sum; ++k)
  {
    term *= (x / (2. * k)) * (x / (2. * k));
    sum += term;
  }
  return sum;
}

// sinc(u) * kaiser(u / zeros) for u in [0, zeros], `resolution` points per
// unit, plus a guard point.
inline const std::vector<float>& kernel()
{
  static const std::vector<float> table = [] {
    std::vector<float> t((std::size_t)zeros * resolution + 2, 0.f);
    const double norm = 1. / besselI0(beta);
    for(std::size_t i = 0; i + 1 < t.size(); ++i)
    {
      const double u = (double)i / resolution;
      const double r = u / zeros;
      const double w = r < 1. ? besselI0(beta * std::sqrt(1. - r * r)) * norm : 0.;
      const double x = 3.14159265358979323846 * u;
      t[i] = (float)(w * (u == 0. ? 1. : std::sin(x) / x));
    }
    return t;
  }();
  return table;
}

inline float kernelAt(const float* k, double u) noexcept
{
  u = std::abs(u) * resolution;
  const auto i = (std::size_t)u;
  if(i >= (std::size_t)zeros * resolution)
    return 0.f;
  const float f = (float)(u - (double)i);
  return k[i] + (k[i + 1] - k[i]) * f;
}

// The cutoff, relative to the input's Nyquist: a little under the lower of
// the two, for the transition band.
inline double cutoff(double ratio) noexcept
{
  return 0.92 * std::min(1., ratio);
}
}

struct Resampler
{
  double ratio = 1.0; // out_rate / in_rate
  double fc = 1.0;    // cutoff, relative to the input's Nyquist
  int half = 0;       // input samples on each side of an output
  double pos = 0.0;   // read position in `history`
  std::vector<float> history;

  void prepare(double in_rate, double out_rate, std::size_t max_block = 0)
  {
    ratio = (in_rate > 0.0 && out_rate > 0.0) ? (out_rate / in_rate) : 1.0;
    fc = resampler_detail::cutoff(ratio);
    half = (int)std::ceil(resampler_detail::zeros / fc);
    (void)resampler_detail::kernel(); // built here, not on the audio thread
    history.clear();
    history.reserve(max_block + 2 * (std::size_t)half + 4);
    reset();
  }

  void reset() noexcept
  {
    // The first outputs read `half` samples of silence before the signal.
    history.assign((std::size_t)half, 0.f);
    pos = (double)half;
  }

  bool passthrough() const noexcept { return std::abs(ratio - 1.0) < 1e-9; }

  // Resample `in` (n samples) into `out` (appended). Returns the number
  // appended. `out` needs spare capacity for about n * ratio + 2 samples.
  std::size_t process(const float* in, std::size_t n, std::vector<float>& out)
  {
    if(n == 0)
      return 0;
    if(passthrough())
    {
      out.insert(out.end(), in, in + n);
      return n;
    }

    history.insert(history.end(), in, in + n);
    const float* k = resampler_detail::kernel().data();
    const float* h = history.data();
    const double step = 1.0 / ratio;
    const std::size_t before = out.size();
    // An output at `pos` reads history[i0 - half + 1 .. i0 + half].
    while((std::size_t)pos + (std::size_t)half < history.size())
    {
      const auto i0 = (std::ptrdiff_t)pos;
      const double frac = pos - (double)i0;
      double acc = 0., wsum = 0.;
      for(int j = -half + 1; j <= half; ++j)
      {
        const float w = resampler_detail::kernelAt(k, ((double)j - frac) * fc);
        acc += (double)w * h[i0 + j];
        wsum += w;
      }
      out.push_back(wsum != 0. ? (float)(acc / wsum) : 0.f);
      pos += step;
    }

    // Drop what no later output reads.
    const auto keep_from = (std::ptrdiff_t)pos - half + 1;
    if(keep_from > 0)
    {
      history.erase(history.begin(), history.begin() + keep_from);
      pos -= (double)keep_from;
    }
    return out.size() - before;
  }

  // A complete signal at another rate: round(n * ratio) samples, aligned on
  // the input (the signal is taken as silent outside of it).
  static void
  resampleWhole(const float* in, std::size_t n, double ratio, std::vector<float>& out)
  {
    out.clear();
    if(n == 0 || !(ratio > 0.))
      return;
    if(std::abs(ratio - 1.0) < 1e-9)
    {
      out.assign(in, in + n);
      return;
    }
    const double fc = resampler_detail::cutoff(ratio);
    const int half = (int)std::ceil(resampler_detail::zeros / fc);
    const float* k = resampler_detail::kernel().data();
    const auto m = (std::size_t)std::llround((double)n * ratio);
    out.resize(m);
    for(std::size_t o = 0; o < m; ++o)
    {
      const double t = (double)o / ratio;
      const auto i0 = (std::ptrdiff_t)t;
      const double frac = t - (double)i0;
      double acc = 0., wsum = 0.;
      for(int j = -half + 1; j <= half; ++j)
      {
        const float w = resampler_detail::kernelAt(k, ((double)j - frac) * fc);
        wsum += w;
        const std::ptrdiff_t i = i0 + j;
        if(i >= 0 && i < (std::ptrdiff_t)n)
          acc += (double)w * in[i];
      }
      out[o] = wsum != 0. ? (float)(acc / wsum) : 0.f;
    }
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
  int stems = 1;     // separated sources in one tensor ([B,S,C,N] outputs)

  // An output: like an input, but a rank-4 [B,S,C,N] (Demucs) holds S stems
  // of C channels. `fallback_channels` stands in for a symbolic channel count.
  static WaveformShape
  fromOutputShape(const std::vector<int64_t>& s, int fallback_channels) noexcept
  {
    if(s.size() != 4)
      return fromInputShape(s);
    WaveformShape w;
    w.layout = WaveLayout::BNC1;
    w.stems = s[1] > 0 ? (int)s[1] : 1;
    w.channels = (s[2] == 1 || s[2] == 2) ? (int)s[2] : std::max(fallback_channels, 1);
    w.block = s[3];
    return w;
  }

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

  // Same, into an existing vector (no allocation once it has the capacity).
  void tensorShapeInto(int64_t n, std::vector<int64_t>& out) const
  {
    switch(layout)
    {
      case WaveLayout::BNC1: out.assign({1, (int64_t)channels, n}); return;
      case WaveLayout::BN:   out.assign({1, n}); return;
      case WaveLayout::N:    out.assign({n}); return;
    }
    out.assign({1, (int64_t)channels, n});
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

  std::vector<Resampler> resamplers; // one per host channel
  std::vector<AudioRing> rings;         // one per model channel (post-mix)
  std::vector<std::vector<float>> rs_scratch; // resample output staging

  void prepare(
      const WaveformShape& s, double in_rate, double out_rate,
      int64_t fixed_block, int64_t hop_size, int host_channels,
      std::size_t max_host_frames, int backlog_blocks = 0)
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
      r.prepare(in_rate, out_rate, max_host_frames);

    // A host block of `max_host_frames` becomes up to ceil(frames*ratio)+1
    // model-rate samples; ring must hold a full model block plus that margin.
    const double ratio = (in_rate > 0) ? out_rate / in_rate : 1.0;
    const std::size_t per_block
        = (std::size_t)std::ceil((double)max_host_frames * ratio) + 2;
    // backlog_blocks: extra whole blocks kept while an async job runs.
    const std::size_t cap
        = (std::size_t)block * (1 + std::max(backlog_blocks, 0)) + per_block + 2;

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
  std::vector<Resampler> resamplers; // one per channel (model->host)
  std::vector<AudioRing> rings;         // one per channel (host rate)
  std::vector<std::vector<float>> rs_scratch;

  // Overlap-add, when the input takes a block every hop < block samples.
  int64_t ola_block = 0;
  int64_t ola_hop = 0;
  std::vector<float> ola_window; // Hann, scaled so the overlaps sum to 1
  std::vector<std::vector<float>> ola_acc; // one block per channel
  std::vector<float> ola_emit;             // planar [C, hop]

  void prepare(
      int chans, double in_rate, double out_rate, int64_t model_block,
      std::size_t max_host_frames, int backlog_blocks = 0)
  {
    channels = chans > 0 ? chans : 1;
    model_rate = in_rate;
    host_rate = out_rate;
    resamplers.assign(channels, {});
    for(auto& r : resamplers)
      r.prepare(in_rate, out_rate, (std::size_t)std::max<int64_t>(model_block, 0));

    const double ratio = (in_rate > 0) ? out_rate / in_rate : 1.0;
    const std::size_t per_block
        = (std::size_t)std::ceil((double)(model_block) * ratio) + 2;
    const std::size_t cap
        = per_block * (1 + std::max(backlog_blocks, 0)) + max_host_frames + 2;
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
    for(auto& a : ola_acc)
      std::fill(a.begin(), a.end(), 0.f);
  }

  // Frames of `block` samples every `hop` (block a multiple of 2 * hop): a
  // periodic Hann window sums to block / (2 * hop) over the overlaps, so it
  // is scaled by the inverse and a model that returns its input gives the
  // input back. hop >= block turns overlap-add off.
  void prepareOverlap(int64_t block, int64_t hop)
  {
    if(hop <= 0 || hop >= block)
    {
      ola_block = ola_hop = 0;
      ola_acc.clear();
      return;
    }
    ola_block = block;
    ola_hop = hop;
    const double gain = 2.0 * (double)hop / (double)block;
    ola_window.resize((std::size_t)block);
    for(int64_t i = 0; i < block; ++i)
      ola_window[i] = (float)(gain * 0.5
                              * (1.0 - std::cos(2.0 * 3.14159265358979323846
                                                * (double)i / (double)block)));
    ola_acc.assign(channels, std::vector<float>((std::size_t)block, 0.f));
    ola_emit.assign((std::size_t)channels * hop, 0.f);
  }

  bool overlapping() const noexcept { return ola_block > 0; }

  // Adds a block-long model frame into the overlap and pushes the `hop`
  // samples it completes. A frame of another length (the model does not
  // return what it was given) is pushed as is.
  void pushOverlap(const float* planar, int chans, int64_t n)
  {
    if(!overlapping() || n != ola_block)
    {
      push(planar, chans, n);
      return;
    }
    const int nc = std::min(chans, (int)ola_acc.size());
    const std::size_t hop = (std::size_t)ola_hop;
    for(int c = 0; c < nc; ++c)
    {
      auto& acc = ola_acc[c];
      const float* in = planar + (std::size_t)c * n;
      for(int64_t i = 0; i < n; ++i)
        acc[i] += ola_window[i] * in[i];
      std::copy_n(acc.begin(), hop, ola_emit.begin() + c * hop);
      std::copy(acc.begin() + hop, acc.end(), acc.begin());
      std::fill(acc.end() - hop, acc.end(), 0.f);
    }
    push(ola_emit.data(), nc, (int64_t)hop);
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
  // Host channels past the model's copy its last channel: popping that ring
  // again would hand each host channel every other block.
  void pull(float* const* chans, int host_channels, std::size_t frames)
  {
    const int popped = std::min(host_channels, std::min(channels, (int)rings.size()));
    for(int c = 0; c < popped; ++c)
      if(!rings[c].pop(chans[c], frames))
        std::fill_n(chans[c], frames, 0.f);
    for(int c = std::max(popped, 0); c < host_channels; ++c)
    {
      if(popped > 0)
        std::copy_n(chans[popped - 1], frames, chans[c]);
      else
        std::fill_n(chans[c], frames, 0.f);
    }
  }

  std::size_t available() const noexcept
  {
    return rings.empty() ? 0 : rings[0].size();
  }
};

} // namespace Onnx
