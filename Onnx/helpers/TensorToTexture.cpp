#include <Onnx/helpers/TensorToTexture.hpp>

#include <algorithm>
#include <cmath>

namespace Onnx
{
namespace
{
// Compile-time per-pixel value -> [0,255] mapping.
template <WriteMode M>
inline float mapPixel(float v, float mn, float inv_range) noexcept
{
  if constexpr(M == WriteMode::DirectClamp)
    return v * 255.f;
  else if constexpr(M == WriteMode::MinMaxNormalize || M == WriteMode::AutoRange)
    return (v - mn) * inv_range * 255.f;
  else if constexpr(M == WriteMode::Sigmoid)
    return 255.f / (1.f + std::exp(-v));
  else if constexpr(M == WriteMode::Denormalize)
    return (v + 1.f) * 127.5f;
  else if constexpr(M == WriteMode::Passthrough)
    return v;
  else // Half255
    return 0.5f + 255.f * v;
}

inline uint8_t clamp8(float x) noexcept
{
  // NaN fails both comparisons and casting NaN to uint8 is UB; the !(x>=0)
  // test catches it (NaN >= 0 is false) and maps it to 0.
  if(!(x >= 0.f))
    return 0;
  return static_cast<uint8_t>(x > 255.f ? 255.f : x);
}

// Fetch channel c at pixel index p (0..HW-1) for NCHW or NHWC.
struct Fetch
{
  const float* data;
  int64_t HW;
  int C;
  bool nhwc;
  inline float operator()(int c, int64_t p) const noexcept
  {
    return nhwc ? data[p * C + c] : data[(int64_t)c * HW + p];
  }
};

// The offset and scale of the range-based modes: MinMaxNormalize always
// stretches [lo,hi]; AutoRange keeps a frame that is already within [0,1],
// give or take the rounding of an fp16 export: a -0.01 in one frame must not
// switch the stretch on for that frame only (flicker).
template <WriteMode M>
inline void rangeParams(float lo, float hi, float& mn, float& inv_range) noexcept
{
  if(M == WriteMode::AutoRange && lo >= -0.02f && hi <= 1.02f)
  {
    mn = 0.f;
    inv_range = 1.f;
    return;
  }
  mn = lo;
  inv_range = (hi - lo > 1e-9f) ? 1.f / (hi - lo) : 1.f;
}

template <WriteMode M>
void writeRgbImpl(const float* data, const OutSpec& s, uint8_t* dst)
{
  const int64_t HW = (int64_t)s.w * s.h;
  const int C = s.channels;
  const Fetch at{data, HW, C, s.nhwc && C >= 3};
  const int gi = C >= 2 ? 1 : 0; // gray broadcast when C==1
  const int bi = C >= 3 ? 2 : 0;
  const bool has_a = C >= 4;

  float mn = 0.f, inv_range = 1.f;
  if constexpr(M == WriteMode::MinMaxNormalize || M == WriteMode::AutoRange)
  {
    float lo = at(0, 0), hi = at(0, 0);
    const int cc = std::min(C, 3);
    for(int c = 0; c < cc; ++c)
      for(int64_t p = 0; p < HW; ++p)
      {
        const float v = at(c, p);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
    rangeParams<M>(lo, hi, mn, inv_range);
  }

  for(int64_t p = 0; p < HW; ++p)
  {
    uint8_t* o = dst + p * 4;
    o[0] = clamp8(mapPixel<M>(at(0, p), mn, inv_range));
    o[1] = clamp8(mapPixel<M>(at(gi, p), mn, inv_range));
    o[2] = clamp8(mapPixel<M>(at(bi, p), mn, inv_range));
    o[3] = has_a ? clamp8(mapPixel<M>(at(3, p), mn, inv_range)) : 255;
  }
}

// A 2-channel (background, foreground) output: its foreground value.
// Probabilities are taken as they are; logits go through the 2-way softmax.
// Values within 0.05 of [0,1] are still probabilities: fp16 exports emit
// 1.0002 or -1e-4, and one such pixel must not switch the whole frame to
// softmax(fg - bg) (a mask that flickers in video).
struct Foreground
{
  const float* data;
  int64_t HW;
  bool nhwc;
  bool logits;
  static Foreground make(const float* data, int64_t HW, bool nhwc)
  {
    Foreground f{data, HW, nhwc, false};
    for(int64_t p = 0; p < HW && !f.logits; ++p)
    {
      const float v = f.channel(1, p);
      f.logits = !(v >= -0.05f && v <= 1.05f);
    }
    return f;
  }
  inline float channel(int c, int64_t p) const noexcept
  {
    return nhwc ? data[p * 2 + c] : data[c * HW + p];
  }
  inline float operator()(int64_t p) const noexcept
  {
    return logits ? 1.f / (1.f + std::exp(channel(0, p) - channel(1, p)))
                  : channel(1, p);
  }
};

template <WriteMode M>
void writeMaskImpl(const float* data, const OutSpec& s, uint8_t* dst)
{
  const int64_t HW = (int64_t)s.w * s.h;
  const int C = s.channels;
  // channel 0: for both NCHW and NHWC-with-C==1 the first HW values are plane 0
  // (single channel is contiguous in either layout). For NHWC C>1 take stride.
  // Two channels are background / foreground: take the foreground.
  const bool strided = s.nhwc && C > 1;
  const bool twoClass = C == 2;
  const Foreground fg = twoClass ? Foreground::make(data, HW, s.nhwc) : Foreground{};
  auto value = [&](int64_t p) {
    return twoClass ? fg(p) : (strided ? data[p * C] : data[p]);
  };

  float mn = 0.f, inv_range = 1.f;
  if constexpr(M == WriteMode::MinMaxNormalize || M == WriteMode::AutoRange)
  {
    float lo = value(0), hi = lo;
    for(int64_t p = 0; p < HW; ++p)
    {
      const float v = value(p);
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    rangeParams<M>(lo, hi, mn, inv_range);
  }

  for(int64_t p = 0; p < HW; ++p)
    dst[p] = clamp8(mapPixel<M>(value(p), mn, inv_range));
}

template <typename Impl>
void dispatchMode(WriteMode m, Impl&& impl)
{
  switch(m)
  {
    case WriteMode::DirectClamp:     impl(std::integral_constant<WriteMode, WriteMode::DirectClamp>{}); break;
    case WriteMode::MinMaxNormalize: impl(std::integral_constant<WriteMode, WriteMode::MinMaxNormalize>{}); break;
    case WriteMode::Denormalize:     impl(std::integral_constant<WriteMode, WriteMode::Denormalize>{}); break;
    case WriteMode::Passthrough:     impl(std::integral_constant<WriteMode, WriteMode::Passthrough>{}); break;
    case WriteMode::Half255:         impl(std::integral_constant<WriteMode, WriteMode::Half255>{}); break;
    case WriteMode::AutoRange:       impl(std::integral_constant<WriteMode, WriteMode::AutoRange>{}); break;
    case WriteMode::Sigmoid:         impl(std::integral_constant<WriteMode, WriteMode::Sigmoid>{}); break;
  }
}
} // namespace

const float* toFloat(
    const void* data, int64_t count, TensorElemType e, std::vector<float>& scratch)
{
  if(e == TensorElemType::Float)
    return static_cast<const float*>(data);
  if(count <= 0)
    return scratch.data();
  if((int64_t)scratch.size() < count)
    scratch.resize((std::size_t)count);
  float* o = scratch.data();
  auto convert = [&]<typename T>(auto&& f) {
    const T* p = static_cast<const T*>(data);
    for(int64_t i = 0; i < count; ++i)
      o[i] = f(p[i]);
  };
  auto cast = [](auto v) { return static_cast<float>(v); };
  switch(e)
  {
    case TensorElemType::Float16:  convert.template operator()<uint16_t>(halfToFloat); break;
    case TensorElemType::BFloat16: convert.template operator()<uint16_t>(bfloat16ToFloat); break;
    case TensorElemType::Double:   convert.template operator()<double>(cast); break;
    case TensorElemType::Uint8:    convert.template operator()<uint8_t>(cast); break;
    case TensorElemType::Int8:     convert.template operator()<int8_t>(cast); break;
    case TensorElemType::Uint16:   convert.template operator()<uint16_t>(cast); break;
    case TensorElemType::Int16:    convert.template operator()<int16_t>(cast); break;
    case TensorElemType::Uint32:   convert.template operator()<uint32_t>(cast); break;
    case TensorElemType::Int32:    convert.template operator()<int32_t>(cast); break;
    case TensorElemType::Uint64:   convert.template operator()<uint64_t>(cast); break;
    case TensorElemType::Int64:    convert.template operator()<int64_t>(cast); break;
    case TensorElemType::Bool:
      convert.template operator()<uint8_t>([](uint8_t b) { return b ? 1.f : 0.f; });
      break;
    default:
      std::fill_n(o, count, 0.f);
      break;
  }
  return o;
}

void writeRgb(const float* data, const OutSpec& s, WriteMode m, uint8_t* dst)
{
  if(!s.spatial || s.w <= 0 || s.h <= 0)
    return;
  dispatchMode(m, [&](auto mode) { writeRgbImpl<decltype(mode)::value>(data, s, dst); });
}

void writeMask(const float* data, const OutSpec& s, WriteMode m, uint8_t* dst)
{
  if(!s.spatial || s.w <= 0 || s.h <= 0)
    return;
  dispatchMode(m, [&](auto mode) { writeMaskImpl<decltype(mode)::value>(data, s, dst); });
}

void writeMaskF(const float* data, const OutSpec& s, float* dst)
{
  if(!s.spatial || s.w <= 0 || s.h <= 0)
    return;
  const int64_t HW = (int64_t)s.w * s.h;
  const bool strided = s.nhwc && s.channels > 1;
  const int C = s.channels;
  for(int64_t p = 0; p < HW; ++p)
    dst[p] = strided ? data[p * C] : data[p];
}
} // namespace Onnx
