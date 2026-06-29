#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Qt-free image helpers for the model input path. These replace the QImage
// usage in Images.hpp's tensor-preprocessing functions: wrap a host RGBA8888
// buffer, resize it to the model resolution (Qt's KeepAspectRatioByExpanding +
// SmoothTransformation, i.e. bilinear scale-to-fill + centre-crop), and pack to
// tightly-stored RGB. Mirrors the Qt-free sampler used in score-addon-onnx's
// ctx overlay path and score-addon-librediffusion's lo::rgba_image.
namespace Onnx
{

// A tightly-packed RGBA8888 host image (stride == width*4). The Qt-free
// replacement for QImage in the GAN / draw-overlay paths: the model objects
// memcpy `pixels` straight into their output texture. `empty()` mirrors
// QImage::isNull().
struct ImageData
{
  std::vector<unsigned char> pixels; // RGBA8888, width*height*4 bytes
  int width{};
  int height{};

  bool empty() const noexcept
  {
    return width <= 0 || height <= 0 || pixels.empty();
  }
};

// Resize a source RGBA8888 image to fill dw x dh (both dimensions >= target,
// aspect ratio preserved), then centre-crop to exactly dw x dh. Bilinear,
// edge-clamped. Returns a tightly-packed (stride == dw*4) RGBA8888 buffer.
// Equivalent to QImage::scaled(KeepAspectRatioByExpanding, SmoothTransformation)
// followed by a centred QImage::copy().
inline std::vector<unsigned char> resize_fill_crop_rgba(
    const unsigned char* src, int sw, int sh, int dw, int dh)
{
  std::vector<unsigned char> out(static_cast<std::size_t>(dw) * dh * 4, 0u);
  if(!src || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
    return out;

  // Scale factor that makes the source cover dw x dh in both axes.
  const double s = std::max(
      static_cast<double>(dw) / sw, static_cast<double>(dh) / sh);
  // Top-left of the centred dw x dh crop, in the (virtual) scaled image.
  const double off_x = (sw * s - dw) / 2.0;
  const double off_y = (sh * s - dh) / 2.0;

  const int sw1 = sw - 1, sh1 = sh - 1;
  for(int y = 0; y < dh; ++y)
  {
    const double fy_src = (off_y + y + 0.5) / s - 0.5;
    const int y0 = static_cast<int>(std::floor(fy_src));
    const double wy = fy_src - y0;
    const int y0c = std::clamp(y0, 0, sh1), y1c = std::clamp(y0 + 1, 0, sh1);
    for(int x = 0; x < dw; ++x)
    {
      const double fx_src = (off_x + x + 0.5) / s - 0.5;
      const int x0 = static_cast<int>(std::floor(fx_src));
      const double wx = fx_src - x0;
      const int x0c = std::clamp(x0, 0, sw1), x1c = std::clamp(x0 + 1, 0, sw1);

      const unsigned char* p00 = src + (static_cast<std::size_t>(y0c) * sw + x0c) * 4;
      const unsigned char* p10 = src + (static_cast<std::size_t>(y0c) * sw + x1c) * 4;
      const unsigned char* p01 = src + (static_cast<std::size_t>(y1c) * sw + x0c) * 4;
      const unsigned char* p11 = src + (static_cast<std::size_t>(y1c) * sw + x1c) * 4;
      unsigned char* d = out.data() + (static_cast<std::size_t>(y) * dw + x) * 4;
      for(int c = 0; c < 4; ++c)
      {
        const double top = p00[c] + (p10[c] - p00[c]) * wx;
        const double bot = p01[c] + (p11[c] - p01[c]) * wx;
        d[c] = static_cast<unsigned char>(top + (bot - top) * wy + 0.5);
      }
    }
  }
  return out;
}

// Stretch a source RGBA8888 image to exactly dw x dh, ignoring aspect ratio
// (bilinear, edge-clamped). Equivalent to
// QImage::scaled(dw, dh, Qt::IgnoreAspectRatio, Qt::SmoothTransformation).
inline std::vector<unsigned char> resize_stretch_rgba(
    const unsigned char* src, int sw, int sh, int dw, int dh)
{
  std::vector<unsigned char> out(static_cast<std::size_t>(dw) * dh * 4, 0u);
  if(!src || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
    return out;

  const double sx = static_cast<double>(sw) / dw;
  const double sy = static_cast<double>(sh) / dh;
  const int sw1 = sw - 1, sh1 = sh - 1;
  for(int y = 0; y < dh; ++y)
  {
    const double fy_src = (y + 0.5) * sy - 0.5;
    const int y0 = static_cast<int>(std::floor(fy_src));
    const double wy = fy_src - y0;
    const int y0c = std::clamp(y0, 0, sh1), y1c = std::clamp(y0 + 1, 0, sh1);
    for(int x = 0; x < dw; ++x)
    {
      const double fx_src = (x + 0.5) * sx - 0.5;
      const int x0 = static_cast<int>(std::floor(fx_src));
      const double wx = fx_src - x0;
      const int x0c = std::clamp(x0, 0, sw1), x1c = std::clamp(x0 + 1, 0, sw1);

      const unsigned char* p00 = src + (static_cast<std::size_t>(y0c) * sw + x0c) * 4;
      const unsigned char* p10 = src + (static_cast<std::size_t>(y0c) * sw + x1c) * 4;
      const unsigned char* p01 = src + (static_cast<std::size_t>(y1c) * sw + x0c) * 4;
      const unsigned char* p11 = src + (static_cast<std::size_t>(y1c) * sw + x1c) * 4;
      unsigned char* d = out.data() + (static_cast<std::size_t>(y) * dw + x) * 4;
      for(int c = 0; c < 4; ++c)
      {
        const double top = p00[c] + (p10[c] - p00[c]) * wx;
        const double bot = p01[c] + (p11[c] - p01[c]) * wx;
        d[c] = static_cast<unsigned char>(top + (bot - top) * wy + 0.5);
      }
    }
  }
  return out;
}

// Drop the alpha channel: tightly-packed RGBA8888 -> tightly-packed RGB888.
inline std::vector<unsigned char>
rgba_to_rgb(const unsigned char* rgba, int w, int h)
{
  const std::size_t n = static_cast<std::size_t>(w) * h;
  std::vector<unsigned char> rgb(n * 3);
  for(std::size_t i = 0, j = 0; i < n; ++i, j += 3)
  {
    const unsigned char* p = rgba + i * 4;
    rgb[j] = p[0];
    rgb[j + 1] = p[1];
    rgb[j + 2] = p[2];
  }
  return rgb;
}

}
