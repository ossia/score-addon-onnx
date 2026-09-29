#pragma once
// ossia::safe_isnan / safe_isinf: bit-pattern checks that survive -ffast-math
// (std::isnan is folded to false there). Vendored fallback when standalone.
#include <Onnx/helpers/compat/safe_math.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// One-Euro filter — velocity-adaptive low-pass smoothing for landmarks.
// Smooths jitter when the point is still, stays responsive when it moves fast.
// (Casiez, Roussel, Vogel 2012; the filter MediaPipe uses on its landmarks.)
namespace Onnx
{
struct OneEuroFilter
{
  float min_cutoff = 1.0f; // lower = smoother (more lag)
  float beta = 0.3f;       // higher = more responsive to speed
  float dcutoff = 1.0f;

  bool initialized = false;
  float x_prev = 0.0f;
  float dx_prev = 0.0f;

  static float alpha(float cutoff, float dt)
  {
    const float r = 2.0f * 3.14159265358979f * cutoff * dt;
    return r / (r + 1.0f);
  }

  float filter(float x, float dt)
  {
    if(!initialized)
    {
      initialized = true;
      x_prev = x;
      dx_prev = 0.0f;
      return x;
    }
    const float dx = (x - x_prev) / dt;
    const float a_d = alpha(dcutoff, dt);
    const float dx_hat = a_d * dx + (1.0f - a_d) * dx_prev;
    const float cutoff = min_cutoff + beta * std::fabs(dx_hat);
    const float a = alpha(cutoff, dt);
    const float x_hat = a * x + (1.0f - a) * x_prev;
    x_prev = x_hat;
    dx_prev = dx_hat;
    return x_hat;
  }

  void reset() { initialized = false; }
};

// One filter per scalar component (e.g. x,y,z per keypoint). Reallocating when
// the component count changes resets the temporal state (intentional).
struct PoseSmoother
{
  std::vector<OneEuroFilter> f;
  float min_cutoff = 1.0f;
  float beta = 0.3f;

  void configure(float mc, float b)
  {
    min_cutoff = mc;
    beta = b;
    for(auto& x : f)
    {
      x.min_cutoff = mc;
      x.beta = b;
    }
  }
  void ensure(size_t n)
  {
    if(f.size() != n)
    {
      OneEuroFilter tmpl{min_cutoff, beta, 1.0f};
      f.assign(n, tmpl);
    }
  }
  void reset() { f.clear(); }
};

// ---------------------------------------------------------------------------
// Generic value-vector smoothing (3D joints, camera translation, parametric
// body parameters). Unlike the screen-keypoint paths, which run in FRAME units
// (dt = 1, cutoffs in cycles/frame), these are meant for signals whose One-Euro
// tuning is published in real units — Hz, and beta per unit/second (metres,
// radians) — so callers pass dt in seconds (the node uses a nominal
// kNominalFrameDt; it has no frame clock). Filter state lives in a
// PoseSmoother with one filter per scalar; it is (re)sized — i.e. reset — only
// when the value count changes, so steady state allocates nothing.
inline constexpr float kNominalFrameDt = 1.0f / 30.0f;

namespace detail
{
inline bool finiteValue(float v) noexcept
{
  return !ossia::safe_isnan(v) && !ossia::safe_isinf(v);
}
// Wrap an angle difference to (-pi, pi].
inline float wrapPi(float a) noexcept
{
  constexpr float pi = 3.14159265358979f, two_pi = 2.0f * pi;
  a = std::remainder(a, two_pi); // [-pi, pi]
  return a <= -pi ? a + two_pi : a;
}
}

// Per-entry kind for smoothParams' mask. Values are the mask bytes: 0 and 1
// keep the original "non-zero = angle" meaning of a plain flag mask.
enum ParamKind : std::uint8_t
{
  ParamLinear = 0, // plain One-Euro at the smoother's (min_cutoff, beta)
  ParamAngle = 1,  // radians, filtered on the unwrapped signal (below)
  // Identity-like parameters that should not follow per-frame noise at all
  // (MHR shape): a pure low-pass (beta 0) at kStaticCutoffScale x the
  // smoother's min_cutoff (InstantHMR's tuning: shape 0.3 Hz / beta 0, pose
  // 1 Hz).
  ParamStatic = 2,
};
inline constexpr float kStaticCutoffScale = 0.3f;

// Smooth `values` in place, one One-Euro filter per entry (s.f[i] <-> values[i]).
//
// is_angle (optional; empty = all linear, else one ParamKind per value):
// entries flagged ParamAngle are ANGLES in radians and are filtered on the unwrapped
// signal. The raw sample is first moved to the branch nearest the filter's
// previous output — prev + wrap(raw - prev) with the delta wrapped to
// (-pi, pi] — so a joint rotating through +-pi is a small step, not a 2*pi
// jump that a linear filter would average into a spin the wrong way round.
// The filter state is re-centred
// into (-pi, pi] after each step so it can't drift over a long session, and
// the output is wrapped back to (-pi, pi] like the model's own output.
// Which entries are angles is model-specific (MHR: the root rotation 3:6 and
// the 124 genuine joint rotations 6:130 of its 204 pose params; not the root
// translation, the six *_length/_width_flexible size channels 130:136 hiding
// in the "130 joint angles" block, nor the 68 bone scales), so the caller owns
// the mask.
//
// A non-finite input leaves its value and its filter untouched: one NaN from an
// untrusted model must not poison x_prev forever.
inline void smoothParams(
    PoseSmoother& s, std::span<float> values,
    std::span<const std::uint8_t> is_angle, float dt)
{
  s.ensure(values.size());
  const bool have_mask = is_angle.size() == values.size();
  for(std::size_t i = 0; i < values.size(); ++i)
  {
    float v = values[i];
    if(!detail::finiteValue(v))
      continue;
    auto& f = s.f[i];
    const std::uint8_t kind = have_mask ? is_angle[i] : ParamLinear;
    if(kind == ParamStatic)
    {
      // Re-applied every call: configure() resets every filter to the
      // smoother's shared parameters each frame.
      f.min_cutoff = s.min_cutoff * kStaticCutoffScale;
      f.beta = 0.f;
    }
    if(kind != ParamAngle)
    {
      values[i] = f.filter(v, dt);
      continue;
    }
    if(f.initialized)
      v = f.x_prev + detail::wrapPi(v - f.x_prev);
    float out = f.filter(v, dt);
    const float wrapped = detail::wrapPi(out);
    f.x_prev += wrapped - out; // same angle, re-centred; dx_prev is unaffected
    values[i] = wrapped;
  }
}

// Smooth x,y,z of each keypoint-like struct (fields x,y,z) in place: 3 filters
// per point (s.f[3*i + {0,1,2}]). Confidence is not filtered. Same dt / NaN
// rules as smoothParams.
template <typename KP>
inline void smoothXYZ(PoseSmoother& s, std::span<KP> kps, float dt)
{
  s.ensure(kps.size() * 3);
  for(std::size_t i = 0; i < kps.size(); ++i)
  {
    auto& k = kps[i];
    if(!detail::finiteValue(k.x) || !detail::finiteValue(k.y)
       || !detail::finiteValue(k.z))
      continue;
    k.x = s.f[3 * i + 0].filter(k.x, dt);
    k.y = s.f[3 * i + 1].filter(k.y, dt);
    k.z = s.f[3 * i + 2].filter(k.z, dt);
  }
}

// Smooths the 5 scalars of a ROI rect (cx, cy, w, h, angle) across frames so the
// crop fed to the landmark model stays stable. Angle is unwrapped to avoid ±pi
// discontinuities.
struct RectSmoother
{
  OneEuroFilter f[5];
  bool initialized = false;
  float prev_angle = 0.0f;

  void configure(float min_cutoff, float beta)
  {
    for(auto& x : f)
    {
      x.min_cutoff = min_cutoff;
      x.beta = beta;
    }
  }
  // in/out: {cx, cy, w, h, angle}; returns the smoothed values in the same array
  void smooth(float* v, float dt = 1.0f)
  {
    // A non-finite rect (from an untrusted model) is left as is and does not
    // reach the filters.
    for(int i = 0; i < 5; ++i)
      if(!detail::finiteValue(v[i]))
        return;
    // unwrap angle relative to previous so it never jumps by ~2pi
    v[4] = prev_angle + detail::wrapPi(v[4] - prev_angle);
    for(int i = 0; i < 5; ++i)
      v[i] = f[i].filter(v[i], dt);
    prev_angle = v[4];
    initialized = true;
  }
  void reset()
  {
    for(auto& x : f) x.reset();
    initialized = false;
  }
};
} // namespace Onnx
