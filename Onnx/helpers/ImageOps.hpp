#pragma once
// Qt-free image operations for the ONNX pose core: bilinear affine warp + fused
// sample/normalize + (Lanczos) resize, on interleaved 8-bit buffers (RGBA).
//
// The per-pixel hot paths are DEFINED in ImageOps.cpp, which is compiled at -O3
// even in Debug builds (see CMakeLists: score_onnx_imageops). Keeping them out
// of the header means they don't get inlined into a -Og/-O0 host TU, where the
// tight float loops run 5-15x slower.
#include <Onnx/helpers/Profile.hpp>

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

// 2x3 affine mapping DESTINATION (output) pixel coords -> SOURCE coords:
//   sx = m0*dx + m1*dy + m2 ;  sy = m3*dx + m4*dy + m5
struct Affine
{
  float m0{1}, m1{0}, m2{0}, m3{0}, m4{1}, m5{0};
};

// Build the affine for a rotated ROI: a rect centered at (cx,cy) in source px,
// of size (rw,rh), rotated by `angle` rad, sampled into a dstW x dstH output.
// Small and called per-ROI (not per-pixel), so kept inline in the header.
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

// ---- defined in ImageOps.cpp (always compiled -O3) ----

// Lanczos resize of the whole src into dst (any up/down scale, antialiased).
void resize(const ImageView& src, const MutableImageView& dst);

// Sample src into dst through the affine map, bilinear, edge-clamped.
void warpAffine(
    const ImageView& src, const MutableImageView& dst, const Affine& a);

// Fused affine-sample + normalize into `out` (3*mw*mh floats, caller-owned).
// Samples src (RGBA, alpha ignored) bilinearly through `a` (output px -> src
// px), edge-clamped; out = (sample - mean[c]) * invstd[c]. prof_bucket tags the
// profiler (WarpDet for whole-frame, WarpCrop for ROI crops).
void sampleAffineToTensor(
    TensorLayout L, const ImageView& src, const Affine& a, int mw, int mh,
    const float mean[3], const float invstd[3], float* out,
    prof::Bucket prof_bucket = prof::WarpCrop);

// Fused aspect-preserving letterbox + normalize into `out`. Pad band ->
// (pad - mean[c]) * invstd[c]. Returns scale + pad for un-letterboxing.
LetterboxInfo letterboxToTensor(
    TensorLayout L, const ImageView& src, int mw, int mh, bool center,
    uint8_t pad, const float mean[3], const float invstd[3], float* out);

// Aspect-preserving bilinear resize of src into dst (fit), padded with `pad`.
// center=true centers (else top-left). Returns scale + pad for un-letterboxing.
LetterboxInfo letterbox(
    const ImageView& src, const MutableImageView& dst, uint8_t pad = 0,
    bool center = true);
} // namespace Onnx
