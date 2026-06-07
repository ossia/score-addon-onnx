#pragma once
// Qt-free image operations for the ONNX pose core: high-quality Lanczos resize
// (via avir LANCIR) and a bilinear affine warp (for rotated ROI crops, which
// LANCIR can't do). Operates on interleaved 8-bit buffers (RGBA by default).
#include <Onnx/helpers/Profile.hpp>
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
  ONNX_PROF_SCOPE(Resize);
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

// Output tensor layout + channel order for the fused samplers below.
enum class TensorLayout
{
  NchwRgb, // planar, R then G then B  (most landmark / SSD models)
  NchwBgr, // planar, B then G then R  (YOLOX / RTMDet / RTMO)
  NhwcRgb  // interleaved RGB          (some PINTO / NHWC exports)
};

namespace detail
{
// Bilinear RGB sample (alpha ignored), edge-clamped. schan = src channels.
inline void bilinearRGB(
    const uint8_t* data, int srow, int schan, int sw1, int sh1, float sx,
    float sy, float& R, float& G, float& B)
{
  const int x0 = static_cast<int>(std::floor(sx));
  const int y0 = static_cast<int>(std::floor(sy));
  const float fx = sx - x0, fy = sy - y0;
  const int x0c = std::clamp(x0, 0, sw1), x1c = std::clamp(x0 + 1, 0, sw1);
  const int y0c = std::clamp(y0, 0, sh1), y1c = std::clamp(y0 + 1, 0, sh1);
  const uint8_t* r0 = data + static_cast<size_t>(y0c) * srow;
  const uint8_t* r1 = data + static_cast<size_t>(y1c) * srow;
  const auto lerp = [&](int ch) {
    const float v00 = r0[x0c * schan + ch], v10 = r0[x1c * schan + ch];
    const float v01 = r1[x0c * schan + ch], v11 = r1[x1c * schan + ch];
    const float top = v00 + (v10 - v00) * fx, bot = v01 + (v11 - v01) * fx;
    return top + (bot - top) * fy;
  };
  R = lerp(0);
  G = lerp(1);
  B = lerp(2);
}

// out = (channel - mean[c]) * invstd[c], scattered per layout. idx = y*mw+x.
template <TensorLayout L>
inline void writeNorm(
    float* out, int plane, int idx, float R, float G, float B,
    const float m[3], const float is[3])
{
  if constexpr(L == TensorLayout::NchwRgb)
  {
    out[idx] = (R - m[0]) * is[0];
    out[plane + idx] = (G - m[1]) * is[1];
    out[2 * plane + idx] = (B - m[2]) * is[2];
  }
  else if constexpr(L == TensorLayout::NchwBgr)
  {
    out[idx] = (B - m[0]) * is[0];
    out[plane + idx] = (G - m[1]) * is[1];
    out[2 * plane + idx] = (R - m[2]) * is[2];
  }
  else // NhwcRgb
  {
    float* d = out + 3 * idx;
    d[0] = (R - m[0]) * is[0];
    d[1] = (G - m[1]) * is[1];
    d[2] = (B - m[2]) * is[2];
  }
}

template <TensorLayout L>
inline void sampleAffineImpl(
    const ImageView& src, const Affine& a, int mw, int mh, const float m[3],
    const float is[3], float* out)
{
  const int schan = src.channels, srow = src.rowBytes();
  const int sw1 = src.w - 1, sh1 = src.h - 1, plane = mw * mh;
  for(int y = 0; y < mh; ++y)
  {
    const float bx = a.m1 * y + a.m2, by = a.m4 * y + a.m5;
    int idx = y * mw;
    for(int x = 0; x < mw; ++x, ++idx)
    {
      float R, G, B;
      bilinearRGB(
          src.data, srow, schan, sw1, sh1, a.m0 * x + bx, a.m3 * x + by, R, G, B);
      writeNorm<L>(out, plane, idx, R, G, B, m, is);
    }
  }
}

template <TensorLayout L>
inline LetterboxInfo letterboxImpl(
    const ImageView& src, int mw, int mh, bool center, uint8_t pad,
    const float m[3], const float is[3], float* out)
{
  LetterboxInfo lb;
  lb.scale = std::min(
      static_cast<float>(mw) / src.w, static_cast<float>(mh) / src.h);
  const int nw = std::max(1, static_cast<int>(std::lround(src.w * lb.scale)));
  const int nh = std::max(1, static_cast<int>(std::lround(src.h * lb.scale)));
  lb.pad_x = center ? (mw - nw) / 2 : 0;
  lb.pad_y = center ? (mh - nh) / 2 : 0;

  const int schan = src.channels, srow = src.rowBytes();
  const int sw1 = src.w - 1, sh1 = src.h - 1, plane = mw * mh;
  const float inv = 1.0f / lb.scale;
  const float padf = static_cast<float>(pad);
  for(int y = 0; y < mh; ++y)
  {
    const bool yin = (y >= lb.pad_y && y < lb.pad_y + nh);
    const float sy = (y - lb.pad_y) * inv;
    int idx = y * mw;
    for(int x = 0; x < mw; ++x, ++idx)
    {
      float R, G, B;
      if(yin && x >= lb.pad_x && x < lb.pad_x + nw)
        bilinearRGB(
            src.data, srow, schan, sw1, sh1, (x - lb.pad_x) * inv, sy, R, G, B);
      else
        R = G = B = padf;
      writeNorm<L>(out, plane, idx, R, G, B, m, is);
    }
  }
  return lb;
}
} // namespace detail

// Fused affine-sample + normalize into `out` (3*mw*mh floats, caller-owned).
// Samples src (RGBA, alpha ignored) bilinearly through `a` (output px -> src
// px), edge-clamped; out = (sample - mean[c]) * invstd[c]. Replaces the
// warp-into-RGBA-scratch + separate normalize pass.
inline void sampleAffineToTensor(
    TensorLayout L, const ImageView& src, const Affine& a, int mw, int mh,
    const float mean[3], const float invstd[3], float* out,
    prof::Bucket prof_bucket = prof::WarpCrop)
{
  ONNX_PROF_SCOPE_VAR(prof_bucket);
  switch(L)
  {
    case TensorLayout::NchwRgb:
      detail::sampleAffineImpl<TensorLayout::NchwRgb>(src, a, mw, mh, mean, invstd, out);
      break;
    case TensorLayout::NchwBgr:
      detail::sampleAffineImpl<TensorLayout::NchwBgr>(src, a, mw, mh, mean, invstd, out);
      break;
    case TensorLayout::NhwcRgb:
      detail::sampleAffineImpl<TensorLayout::NhwcRgb>(src, a, mw, mh, mean, invstd, out);
      break;
  }
}

// Fused aspect-preserving letterbox + normalize into `out`. Pad band ->
// (pad - mean[c]) * invstd[c]. Returns scale + pad for un-letterboxing.
inline LetterboxInfo letterboxToTensor(
    TensorLayout L, const ImageView& src, int mw, int mh, bool center,
    uint8_t pad, const float mean[3], const float invstd[3], float* out)
{
  ONNX_PROF_SCOPE(WarpDet);
  switch(L)
  {
    case TensorLayout::NchwRgb:
      return detail::letterboxImpl<TensorLayout::NchwRgb>(src, mw, mh, center, pad, mean, invstd, out);
    case TensorLayout::NchwBgr:
      return detail::letterboxImpl<TensorLayout::NchwBgr>(src, mw, mh, center, pad, mean, invstd, out);
    case TensorLayout::NhwcRgb:
      return detail::letterboxImpl<TensorLayout::NhwcRgb>(src, mw, mh, center, pad, mean, invstd, out);
  }
  return {};
}

// Aspect-preserving resize of src into dst (fit), padded with `pad`. center=true
// centers the image (else top-left). Returns scale + pad for un-letterboxing.
// Uses bilinear sampling (cost ~ output pixels), not Lanczos (cost ~ source
// pixels): on large webcam frames downscaled to a small model input this is the
// difference between ~35ms and <1ms. Undersamples on big downscales (aliasing),
// which is acceptable for noisy real-time input.
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
  // bilinear-resize into the centered sub-rect (strided dst view). The affine
  // maps inner-dst px -> src px: sx = x/scale, sy = y/scale.
  MutableImageView inner{
      dst.data + static_cast<size_t>(lb.pad_y) * drow + lb.pad_x * C, nw, nh, C,
      drow};
  const float inv = 1.0f / lb.scale;
  warpAffine(src, inner, Affine{inv, 0.f, 0.f, 0.f, inv, 0.f});
  return lb;
}
} // namespace Onnx
