#pragma once
// Qt-free image operations for the ONNX pose core: high-quality Lanczos resize
// (via avir LANCIR) and a bilinear affine warp (for rotated ROI crops, which
// LANCIR can't do). Operates on interleaved 8-bit buffers (RGBA by default).
#include <Onnx/helpers/lancir.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Onnx
{
// Non-owning interleaved 8-bit image views. stride is in bytes per row; for
// 8-bit data, "elements" (LANCIR's unit) == bytes.
struct ImageView
{
  const uint8_t* data{};
  int w{}, h{}, channels{4};
  int stride{}; // 0 => tight (w*channels)
  int rowBytes() const { return stride > 0 ? stride : w * channels; }
};
struct MutableImageView
{
  uint8_t* data{};
  int w{}, h{}, channels{4};
  int stride{};
  int rowBytes() const { return stride > 0 ? stride : w * channels; }
};

// Lanczos resize of the whole src into dst (any up/down scale, antialiased).
inline void resize(const ImageView& src, const MutableImageView& dst)
{
  thread_local avir::CLancIR rs;
  avir::CLancIRParams p;
  p.SrcSSize = src.rowBytes(); // bytes == elements for 8-bit
  p.NewSSize = dst.rowBytes();
  rs.resizeImage(
      src.data, src.w, src.h, dst.data, dst.w, dst.h, dst.channels, &p);
}

// 2x3 affine mapping DESTINATION (output) pixel coords -> SOURCE coords:
//   sx = m0*dx + m1*dy + m2 ;  sy = m3*dx + m4*dy + m5
struct Affine
{
  float m0{1}, m1{0}, m2{0}, m3{0}, m4{1}, m5{0};
};

// Build the affine for a rotated ROI: a rect centered at (cx,cy) in source px,
// of size (rw,rh), rotated by `angle` rad, sampled into a dstW x dstH output.
// Matches the crop->image mapping used by the ROI helpers.
inline Affine affineFromRoi(
    float cx, float cy, float rw, float rh, float angle, int dstW, int dstH)
{
  const float c = std::cos(angle), s = std::sin(angle);
  const float ax = rw / dstW, ay = rh / dstH;
  Affine a;
  a.m0 = c * ax;
  a.m1 = -s * ay;
  a.m2 = cx - 0.5f * c * rw + 0.5f * s * rh;
  a.m3 = s * ax;
  a.m4 = c * ay;
  a.m5 = cy - 0.5f * s * rw - 0.5f * c * rh;
  return a;
}

// Sample src into dst through the affine map, bilinear, edge-clamped.
inline void warpAffine(
    const ImageView& src, const MutableImageView& dst, const Affine& a)
{
  const int C = dst.channels;
  const int srow = src.rowBytes(), drow = dst.rowBytes();
  const int sw1 = src.w - 1, sh1 = src.h - 1;
  for(int y = 0; y < dst.h; ++y)
  {
    uint8_t* dp = dst.data + static_cast<size_t>(y) * drow;
    const float bx = a.m1 * y + a.m2, by = a.m4 * y + a.m5;
    for(int x = 0; x < dst.w; ++x, dp += C)
    {
      const float sx = a.m0 * x + bx, sy = a.m3 * x + by;
      const int x0 = static_cast<int>(std::floor(sx));
      const int y0 = static_cast<int>(std::floor(sy));
      const float fx = sx - x0, fy = sy - y0;
      const int x0c = std::clamp(x0, 0, sw1), x1c = std::clamp(x0 + 1, 0, sw1);
      const int y0c = std::clamp(y0, 0, sh1), y1c = std::clamp(y0 + 1, 0, sh1);
      const uint8_t* r0 = src.data + static_cast<size_t>(y0c) * srow;
      const uint8_t* r1 = src.data + static_cast<size_t>(y1c) * srow;
      for(int ch = 0; ch < C; ++ch)
      {
        const float v00 = r0[x0c * src.channels + ch];
        const float v10 = r0[x1c * src.channels + ch];
        const float v01 = r1[x0c * src.channels + ch];
        const float v11 = r1[x1c * src.channels + ch];
        const float top = v00 + (v10 - v00) * fx;
        const float bot = v01 + (v11 - v01) * fx;
        dp[ch] = static_cast<uint8_t>(top + (bot - top) * fy + 0.5f);
      }
    }
  }
}

struct LetterboxInfo
{
  float scale{1};
  int pad_x{}, pad_y{};
};

// Aspect-preserving resize of src into dst (fit), padded with `pad`. center=true
// centers the image (else top-left). Returns scale + pad for un-letterboxing.
inline LetterboxInfo letterbox(
    const ImageView& src, const MutableImageView& dst, uint8_t pad = 0,
    bool center = true)
{
  const int C = dst.channels, drow = dst.rowBytes();
  LetterboxInfo lb;
  lb.scale = std::min(
      static_cast<float>(dst.w) / src.w, static_cast<float>(dst.h) / src.h);
  const int nw = std::max(1, static_cast<int>(std::lround(src.w * lb.scale)));
  const int nh = std::max(1, static_cast<int>(std::lround(src.h * lb.scale)));
  lb.pad_x = center ? (dst.w - nw) / 2 : 0;
  lb.pad_y = center ? (dst.h - nh) / 2 : 0;

  // fill background
  for(int y = 0; y < dst.h; ++y)
  {
    uint8_t* row = dst.data + static_cast<size_t>(y) * drow;
    std::fill(row, row + dst.w * C, pad);
  }
  // resize into the centered sub-rect (strided dst view)
  MutableImageView inner{
      dst.data + static_cast<size_t>(lb.pad_y) * drow + lb.pad_x * C, nw, nh, C,
      drow};
  resize(src, inner);
  return lb;
}
} // namespace Onnx
