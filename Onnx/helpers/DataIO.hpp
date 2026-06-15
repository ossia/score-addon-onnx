#pragma once
// DataIO — sequence/vector modality adapter for the generic ONNX node set.
//
// Bridges a flat value-port payload (std::vector<float> + a small shape stamp)
// to/from an ONNX tensor shaped [1,T,F] (time x features) or [1,D] (flat
// vector). It mirrors the role TensorToTexture plays for the image node, but for
// data: it does NOT touch ORT or halp, so it stays dependency-free and
// standalone-testable (the node builds the actual Ort::Value from the buffer +
// shape this adapter produces).
//
// Responsibilities:
//  - resolve the concrete input tensor shape from the model's declared shape
//    (batch forced to 1, dynamic dims -1/0 filled from the payload length),
//  - maintain an optional sliding/accumulating window so models that need a
//    fixed T can be fed one frame at a time (the host buffers the last T frames),
//  - flatten the windowed payload into a contiguous input buffer in [1,T,F]
//    (row-major, time-major) order,
//  - stamp outputs back into a flat vector + (T,F)/D shape metadata.
//
// Real-time safe: after the first reserve()/resize() to the steady-state sizes,
// the hot path (push + flatten + readback) performs no allocation. All buffers
// are members reused across calls.
#include <Onnx/helpers/TensorType.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace Onnx
{

// Lightweight shape descriptor for a data payload moving across a value port.
// A value port only carries a flat vector; this travels alongside it (or is
// recomputed) so consumers know how to read it back as a matrix.
struct DataShape
{
  int64_t T = 0; // time steps (0 == "flat vector", use D)
  int64_t F = 0; // features per step
  int64_t D = 0; // flat length for the [1,D] case (== T*F when matrix)

  bool isMatrix() const noexcept { return T > 0 && F > 0; }
  int64_t flat() const noexcept { return isMatrix() ? (T * F) : D; }
};

// How the host feeds a model that declares a fixed time dimension T while the
// upstream produces one (or a few) frame(s) per tick.
enum class WindowMode : uint8_t
{
  // Pass the incoming payload straight through, reshaped to the model shape.
  // Use when the upstream already provides a full [T,F] block per tick.
  Passthrough,
  // Keep the last T frames in a ring; each tick appends the incoming frame(s)
  // and emits the full [T,F] window once it has filled. Use for streaming
  // single-frame input into a fixed-T model (most RNN/TCN/pose-lift cases).
  Sliding,
};

// Resolve the model's declared input shape into a concrete [1,T,F] / [1,D]
// request. `declared` is spec.inputs[i].shape (may contain -1 / 0 for dynamic
// dims). `payload_len` is the length of the incoming flat vector for this tick.
// `feat_hint` (>0) pins the feature count F when the model leaves it dynamic.
//
// Rules:
//  - batch (dim 0) is always forced to 1,
//  - a concrete positive dim is kept as-is,
//  - a single dynamic dim is filled so the product matches payload_len when
//    possible; otherwise it falls back to feat_hint / payload_len,
//  - rank<=1 declared shapes become [1, payload_len] (flat vector).
struct ResolvedShape
{
  std::vector<int64_t> shape; // concrete tensor shape, batch == 1
  DataShape data;             // T/F/D view of `shape`
};

inline ResolvedShape resolveInputShape(
    const std::vector<int64_t>& declared, int64_t payload_len,
    int64_t feat_hint = 0)
{
  ResolvedShape r;
  payload_len = std::max<int64_t>(payload_len, 0);

  // Flat vector models: [D] or [1,D] (or scalar / empty declared shape).
  if(declared.size() <= 1)
  {
    const int64_t d = (declared.size() == 1 && declared[0] > 0)
                          ? declared[0]
                          : (payload_len > 0 ? payload_len : 1);
    r.shape = {1, d};
    r.data = {.T = 0, .F = 0, .D = d};
    return r;
  }

  r.shape.assign(declared.begin(), declared.end());
  r.shape[0] = 1; // batch always 1

  // Count dynamic dims (<=0) among the non-batch axes and the fixed product.
  int dyn_idx = -1;
  int dyn_count = 0;
  int64_t fixed_prod = 1;
  for(size_t i = 1; i < r.shape.size(); ++i)
  {
    if(r.shape[i] <= 0)
    {
      dyn_count++;
      dyn_idx = (int)i;
    }
    else
    {
      fixed_prod *= r.shape[i];
    }
  }

  if(dyn_count == 1 && fixed_prod > 0)
  {
    // Fill the single dynamic dim so the element count matches the payload.
    int64_t fill = (payload_len > 0 && payload_len % fixed_prod == 0)
                       ? payload_len / fixed_prod
                       : (feat_hint > 0 ? feat_hint : 1);
    r.shape[dyn_idx] = std::max<int64_t>(fill, 1);
  }
  else if(dyn_count >= 1)
  {
    // Multiple dynamic dims: best-effort. Pin the last to feat_hint (or 1) and
    // the rest to 1, except spread the payload over the first dynamic dim.
    for(size_t i = 1; i < r.shape.size(); ++i)
      if(r.shape[i] <= 0)
        r.shape[i] = 1;
    if(dyn_idx >= 0 && feat_hint > 0)
      r.shape[dyn_idx] = feat_hint;
  }

  // Derive the (T,F)/D view from the resolved rank.
  const size_t rank = r.shape.size();
  if(rank == 2)
  {
    r.data = {.T = 0, .F = 0, .D = r.shape[1]};
  }
  else // rank >= 3: treat dims [1..rank-2] as T (collapsed) and last as F
  {
    int64_t t = 1;
    for(size_t i = 1; i + 1 < rank; ++i)
      t *= r.shape[i];
    r.data = {.T = t, .F = r.shape[rank - 1], .D = t * r.shape[rank - 1]};
  }
  return r;
}

// Sliding/accumulating frame window. Holds up to `capacity` frames of `F`
// features each in a flat ring buffer; emits the contiguous [T,F] block in
// chronological order. Real-time safe once `configure()` has sized it.
class FrameWindow
{
public:
  // (Re)configure for T frames of F features. Clears history if dims changed.
  void configure(int64_t T, int64_t F)
  {
    if(T == m_T && F == m_F)
      return;
    m_T = std::max<int64_t>(T, 0);
    m_F = std::max<int64_t>(F, 0);
    m_ring.assign((size_t)(m_T * m_F), 0.f);
    m_flat.assign((size_t)(m_T * m_F), 0.f);
    m_count = 0;
    m_head = 0;
  }

  void reset()
  {
    std::fill(m_ring.begin(), m_ring.end(), 0.f);
    std::fill(m_flat.begin(), m_flat.end(), 0.f);
    m_count = 0;
    m_head = 0;
  }

  int64_t T() const noexcept { return m_T; }
  int64_t F() const noexcept { return m_F; }
  bool filled() const noexcept { return m_count >= m_T; }

  // Append a payload of `len` floats interpreted as floor(len/F) frames of F
  // features (the last partial frame, if any, is ignored). Frames overwrite the
  // oldest entries once full.
  void push(const float* data, int64_t len)
  {
    if(m_F <= 0 || m_T <= 0 || !data)
      return;
    const int64_t nframes = len / m_F;
    for(int64_t k = 0; k < nframes; ++k)
    {
      float* dst = m_ring.data() + (size_t)(m_head * m_F);
      std::copy_n(data + k * m_F, (size_t)m_F, dst);
      m_head = (m_head + 1) % m_T;
      if(m_count < m_T)
        ++m_count;
    }
  }

  // Materialize the window as a contiguous [T,F] block, oldest frame first.
  // Until the ring has filled, the missing (older) frames read as zeros.
  const std::vector<float>& flatten()
  {
    if(m_T <= 0 || m_F <= 0)
      return m_flat;
    // oldest index: when full, it's m_head; before full, it's 0.
    const int64_t start = m_count < m_T ? 0 : m_head;
    for(int64_t i = 0; i < m_T; ++i)
    {
      const int64_t src = (start + i) % m_T;
      std::copy_n(
          m_ring.data() + (size_t)(src * m_F), (size_t)m_F,
          m_flat.data() + (size_t)(i * m_F));
    }
    return m_flat;
  }

private:
  int64_t m_T = 0, m_F = 0;
  int64_t m_count = 0, m_head = 0;
  std::vector<float> m_ring;
  std::vector<float> m_flat;
};

// Output readback: copy a flat output tensor (already decoded to float by the
// node) into `out` and stamp its (T,F)/D shape from the output tensor shape.
// `oshape` is the result tensor's full shape (batch included). Real-time safe:
// `out` is the caller's reused buffer.
inline DataShape readOutput(
    const float* data, int64_t count, const std::vector<int64_t>& oshape,
    std::vector<float>& out)
{
  out.assign(data, data + std::max<int64_t>(count, 0));

  DataShape ds;
  // Strip a leading batch dim of 1 to recover the logical rank.
  size_t b = 0;
  if(!oshape.empty() && oshape[0] == 1)
    b = 1;
  const size_t rank = oshape.size() - b;
  if(rank <= 1)
  {
    ds = {.T = 0, .F = 0, .D = count};
  }
  else
  {
    int64_t t = 1;
    for(size_t i = b; i + 1 < oshape.size(); ++i)
      t *= (oshape[i] > 0 ? oshape[i] : 1);
    const int64_t f = oshape.back() > 0 ? oshape.back() : (t > 0 ? count / t : count);
    ds = {.T = t, .F = f, .D = t * f};
  }
  return ds;
}

// Convenience: full input-build for the node. Given the declared model input
// shape, the incoming payload, and a window, produce the concrete shape + a
// pointer to a contiguous [1,T,F]/[1,D] float buffer ready to wrap as a tensor.
//
//  - Passthrough: the payload itself is reshaped (no buffering); `scratch`
//    holds a copy sized to the resolved element count (zero-padded / truncated).
//  - Sliding: the window is (re)configured to the resolved (T,F), the payload is
//    pushed, and the flattened window is the buffer.
struct InputBuild
{
  std::vector<int64_t> shape; // concrete tensor shape (batch == 1)
  const float* data = nullptr; // points into scratch or the window
  int64_t count = 0;
  bool ready = true; // false == window not yet filled (skip this tick)
};

inline InputBuild buildInput(
    const std::vector<int64_t>& declared, const std::vector<float>& payload,
    WindowMode mode, FrameWindow& window, std::vector<float>& scratch,
    int64_t feat_hint = 0, bool require_full_window = true)
{
  const ResolvedShape rs
      = resolveInputShape(declared, (int64_t)payload.size(), feat_hint);
  InputBuild b;
  b.shape = rs.shape;
  const int64_t need = rs.data.flat();
  b.count = need;

  if(mode == WindowMode::Sliding && rs.data.isMatrix())
  {
    window.configure(rs.data.T, rs.data.F);
    window.push(payload.data(), (int64_t)payload.size());
    if(require_full_window && !window.filled())
    {
      b.ready = false;
      return b;
    }
    b.data = window.flatten().data();
    return b;
  }

  // Passthrough (or flat vector): copy payload into scratch sized to `need`,
  // zero-padding a short payload and truncating a long one.
  scratch.assign((size_t)std::max<int64_t>(need, 0), 0.f);
  const int64_t n = std::min<int64_t>(need, (int64_t)payload.size());
  std::copy_n(payload.data(), (size_t)std::max<int64_t>(n, 0), scratch.data());
  b.data = scratch.data();
  return b;
}

} // namespace Onnx
