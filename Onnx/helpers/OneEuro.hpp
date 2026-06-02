#pragma once
#include <cmath>
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
} // namespace Onnx
