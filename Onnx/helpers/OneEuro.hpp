#pragma once
#include <Onnx/helpers/compat/tracking_math.hpp>

#include <vector>

// One-Euro filter — velocity-adaptive low-pass smoothing for landmarks.
// Smooths jitter when the point is still, stays responsive when it moves fast.
// (Casiez, Roussel, Vogel 2012; the filter MediaPipe uses on its landmarks.)
//
// Since the shared filter library landed these are thin wrappers over
// ossia::one_euro_filter. BEHAVIOUR NOTE for anyone diffing against the old
// standalone implementation: the derivative is now taken against the previous
// *raw* input, as in the Casiez reference implementation, where the old code
// differentiated against the previous *filtered* value. The old form smoothed
// the derivative estimate slightly more, so with the same beta the filter now
// reacts marginally faster during speed changes; at the shipped defaults the
// difference is far below the landmark noise floor. dt <= 0 now returns the
// previous output instead of dividing by zero.
namespace Onnx
{
struct OneEuroFilter
{
  float min_cutoff = 1.0f; // lower = smoother (more lag)
  float beta = 0.3f;       // higher = more responsive to speed
  float dcutoff = 1.0f;

  ossia::one_euro_filter<float> impl{};

  float filter(float x, float dt)
  {
    // The tunables are public fields written directly by the smoothers below,
    // so sync them into the shared filter on every call (three float moves).
    impl.min_cutoff = min_cutoff;
    impl.beta = beta;
    impl.d_cutoff = dcutoff;
    return impl(x, dt);
  }

  void reset() { impl.reset(); }
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
    // unwrap angle relative to previous so it never jumps by ~2pi
    v[4] = ossia::unwrap_angle(v[4], prev_angle);
    for(int i = 0; i < 5; ++i)
      v[i] = f[i].filter(v[i], dt);
    prev_angle = v[4];
    initialized = true;
  }
  void reset()
  {
    for(auto& x : f)
      x.reset();
    initialized = false;
  }
};
} // namespace Onnx
