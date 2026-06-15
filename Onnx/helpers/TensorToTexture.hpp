#pragma once
// Generic ONNX output tensor -> texture conversion for the ImageProcessor node.
// The dual of ImageOps' input samplers: ImageOps is input-only and 3-channel
// float out, so the tensor->texture (output) path lives here.
//
// Handles rank 2/3/4 results, NCHW/NHWC/gray (channel position read from the
// RESULT shape, independent of the input — this is what fixes the old
// NHWC-mask [1,H,W,1] misread), fp16/uint8/etc. element types (via toFloat),
// and the de-normalization variety from the GAN/depth/SR models. The per-pixel
// hot loops are DEFINED in TensorToTexture.cpp, compiled at -O3 in the
// score_onnx_imageops lib (same rationale as ImageOps.cpp).
#include <Onnx/helpers/ImageModelRole.hpp> // detail::resolveLayout, ImgLayout
#include <Onnx/helpers/TensorType.hpp>

#include <cstdint>
#include <vector>

namespace Onnx
{
// How a raw output value maps to an 8-bit pixel. Superset of the GAN/depth/SR
// denormalizations: clamp[0,1]*255, min-max stretch, inverse-[-1,1], raw clamp,
// and PyTorchGAN's 0.5+255*x.
enum class WriteMode : uint8_t
{
  DirectClamp,     // clamp(x,0,1)*255       — [0,1] outputs
  MinMaxNormalize, // (x-min)/(max-min)*255  — depth / dynamic-range GANs
  Denormalize,     // (x+1)*127.5            — [-1,1] outputs
  Passthrough,     // clamp(x,0,255)         — already-[0,255] outputs
  Half255,         // clamp(0.5+255*x,0,255) — PyTorchGAN
};

// Resolved geometry of an output tensor (channel position from the shape).
struct OutSpec
{
  int rank = 0;
  bool nhwc = false; // interleaved (only meaningful for channels >= 3)
  int channels = 0;
  int w = 0, h = 0;
  bool spatial = false; // false => scalar/vector (ImageToData)
};

inline OutSpec makeOutSpec(const std::vector<int64_t>& shape)
{
  OutSpec s;
  s.rank = static_cast<int>(shape.size());
  const auto li = detail::resolveLayout(shape);
  s.spatial = (li.layout != ImgLayout::Unknown) && (s.rank >= 3);
  if(s.spatial)
  {
    s.nhwc = (li.layout == ImgLayout::NhwcRgb);
    s.channels = li.channels;
    s.w = li.w;
    s.h = li.h;
  }
  return s;
}

// --- defined in TensorToTexture.cpp (compiled -O3) ----

// Convert `count` elements of arbitrary `e` to float. Returns `data` cast in
// place when e == Float (zero copy); otherwise fills `scratch` and returns its
// data(). fp16/bf16 are unpacked in software.
const float* toFloat(
    const void* data, int64_t count, TensorElemType e,
    std::vector<float>& scratch);

// RGB(A) tensor -> RGBA8 destination (4 bytes/pixel, alpha forced 255 unless the
// tensor has a 4th channel). channels==1 is broadcast to gray RGB.
void writeRgb(const float* data, const OutSpec& s, WriteMode m, uint8_t* dst_rgba8);

// Single-channel (channel 0) tensor -> R8 destination (1 byte/pixel).
void writeMask(const float* data, const OutSpec& s, WriteMode m, uint8_t* dst_r8);

// Single-channel tensor -> R32F destination, raw values (precision-preserving;
// e.g. relative/metric depth). No clamping or scaling — downstream normalizes.
void writeMaskF(const float* data, const OutSpec& s, float* dst_r32f);
} // namespace Onnx
