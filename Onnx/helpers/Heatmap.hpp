#pragma once
// Sub-pixel refinement of keypoint heatmaps, as the reference pipelines do it:
// mmpose's DARK (ViTPose, 2D-FAN) and MediaPipe's RefineLandmarksFromHeatmap
// (BlazePose full / heavy). Pure functions over raw buffers, so the decoders
// and the tests share them.
#include <Onnx/helpers/CoreTypes.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace Onnx::Heatmap
{
// The Gaussian mmpose blurs a heatmap with before DARK: cv2.GaussianBlur's
// kernel for size 11, sigma 0 (= 2), normalized.
inline constexpr int kDarkRadius = 5;
inline constexpr std::array<float, 2 * kDarkRadius + 1> kDarkKernel = [] {
  std::array<float, 2 * kDarkRadius + 1> k{};
  float sum = 0.f;
  for(int i = 0; i < int(k.size()); ++i)
  {
    const float d = float(i - kDarkRadius);
    // constexpr exp: the Taylor series of exp(-d^2/8) converges for d <= 5
    const float x = -d * d / 8.f;
    float term = 1.f, e = 1.f;
    for(int n = 1; n < 40; ++n)
    {
      term *= x / n;
      e += term;
    }
    k[i] = e;
    sum += e;
  }
  for(auto& v : k)
    v /= sum;
  return k;
}();

// mmpose's DARK refinement of the peak (hx, hy) of one hw x hh heatmap: one
// Newton step on the log of the blurred heatmap, scaled back to the raw peak
// value and clipped to [0.001, 50]. mmpose zero-pads before blurring, so the
// map is zero outside. Only the 3x3 neighbourhood of the peak is blurred.
// Returns the offset in cells, each axis in [-1, 1]; 0 on the border cells.
inline Vec2 darkOffset(const float* hm, int hw, int hh, int hx, int hy, float peak_val)
{
  if(hx < 1 || hx >= hw - 1 || hy < 1 || hy >= hh - 1)
    return {0.f, 0.f};
  auto blurred = [&](int x, int y) {
    float acc = 0.f;
    for(int j = -kDarkRadius; j <= kDarkRadius; ++j)
    {
      const int yy = y + j;
      if(yy < 0 || yy >= hh)
        continue;
      float row = 0.f;
      for(int i = -kDarkRadius; i <= kDarkRadius; ++i)
      {
        const int xx = x + i;
        if(xx >= 0 && xx < hw)
          row += kDarkKernel[i + kDarkRadius] * hm[static_cast<size_t>(yy) * hw + xx];
      }
      acc += kDarkKernel[j + kDarkRadius] * row;
    }
    return acc;
  };
  const float peak = blurred(hx, hy);
  const float norm = peak > 1e-12f ? peak_val / peak : 1.f;
  float L[3][3];
  for(int j = 0; j < 3; ++j)
    for(int i = 0; i < 3; ++i)
    {
      const float b = (i == 1 && j == 1) ? peak : blurred(hx + i - 1, hy + j - 1);
      L[j][i] = std::log(std::clamp(b * norm, 0.001f, 50.f));
    }
  const float dx = 0.5f * (L[1][2] - L[1][0]);
  const float dy = 0.5f * (L[2][1] - L[0][1]);
  const float dxx = L[1][2] - 2.f * L[1][1] + L[1][0];
  const float dyy = L[2][1] - 2.f * L[1][1] + L[0][1];
  const float dxy = 0.25f * (L[2][2] - L[0][2] - L[2][0] + L[0][0]);
  const float det = dxx * dyy - dxy * dxy;
  if(std::fabs(det) <= 1e-9f)
    return {0.f, 0.f};
  return {
      std::clamp(-(dyy * dx - dxy * dy) / det, -1.f, 1.f),
      std::clamp(-(dxx * dy - dxy * dx) / det, -1.f, 1.f)};
}

// MediaPipe's RefineLandmarksFromHeatmap (kernel 7, min confidence 0.5) for
// landmark k of a [hh, hw, channels] heatmap of logits: the confidence-
// weighted centre of the 7x7 cells around the regressed point (u, v,
// normalized to the crop), which is sharper than the regression. Leaves
// (u, v) as they are when the point is outside the map or the heatmap is not
// confident there; returns whether it refined.
inline bool refineFromHeatmap(
    const float* hm, int hh, int hw, int channels, int k, float& u, float& v)
{
  const int cc = static_cast<int>(u * hw);
  const int cr = static_cast<int>(v * hh);
  if(cc < 0 || cc >= hw || cr < 0 || cr >= hh)
    return false;
  float sum = 0.f, wc = 0.f, wr = 0.f, mx = 0.f;
  for(int r = std::max(0, cr - 3); r < std::min(hh, cr + 4); ++r)
    for(int c = std::max(0, cc - 3); c < std::min(hw, cc + 4); ++c)
    {
      const float logit = hm[(static_cast<size_t>(r) * hw + c) * channels + k];
      const float conf = 1.0f / (1.0f + std::exp(-logit));
      sum += conf;
      mx = std::max(mx, conf);
      wc += c * conf;
      wr += r * conf;
    }
  if(mx < 0.5f || sum <= 0.f)
    return false;
  u = wc / sum / hw;
  v = wr / sum / hh;
  return true;
}
}
