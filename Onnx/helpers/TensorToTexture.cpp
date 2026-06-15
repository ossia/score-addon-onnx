#include <Onnx/helpers/TensorToTexture.hpp>

#include <algorithm>
#include <cstring>

namespace Onnx
{
namespace
{
// IEEE half (binary16) -> float32. Branchless-ish; handles subnormals/inf/nan.
inline float halfToFloat(uint16_t h) noexcept
{
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if(exp == 0)
  {
    if(mant == 0)
      bits = sign; // +/-0
    else
    {
      // subnormal: normalize
      int e = -1;
      uint32_t m = mant;
      do { m <<= 1; ++e; } while((m & 0x400u) == 0);
      m &= 0x3FFu;
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
    }
  }
  else if(exp == 0x1Fu)
  {
    bits = sign | 0x7F800000u | (mant << 13); // inf / nan
  }
  else
  {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

template <typename T>
void copyAs(const void* src, int64_t n, std::vector<float>& out)
{
  const T* p = static_cast<const T*>(src);
  out.resize(n);
  for(int64_t i = 0; i < n; ++i)
    out[i] = static_cast<float>(p[i]);
}

// Compile-time per-pixel value -> [0,255] mapping.
template <WriteMode M>
inline float mapPixel(float v, float mn, float inv_range) noexcept
{
  if constexpr(M == WriteMode::DirectClamp)
    return v * 255.f;
  else if constexpr(M == WriteMode::MinMaxNormalize)
    return (v - mn) * inv_range * 255.f;
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
  if constexpr(M == WriteMode::MinMaxNormalize)
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
    mn = lo;
    inv_range = (hi - lo > 1e-9f) ? 1.f / (hi - lo) : 1.f;
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

template <WriteMode M>
void writeMaskImpl(const float* data, const OutSpec& s, uint8_t* dst)
{
  const int64_t HW = (int64_t)s.w * s.h;
  // channel 0: for both NCHW and NHWC-with-C==1 the first HW values are plane 0
  // (single channel is contiguous in either layout). For NHWC C>1 take stride.
  const bool strided = s.nhwc && s.channels > 1;
  const int C = s.channels;

  float mn = 0.f, inv_range = 1.f;
  if constexpr(M == WriteMode::MinMaxNormalize)
  {
    float lo = data[0], hi = data[0];
    for(int64_t p = 0; p < HW; ++p)
    {
      const float v = strided ? data[p * C] : data[p];
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    mn = lo;
    inv_range = (hi - lo > 1e-9f) ? 1.f / (hi - lo) : 1.f;
  }

  for(int64_t p = 0; p < HW; ++p)
  {
    const float v = strided ? data[p * C] : data[p];
    dst[p] = clamp8(mapPixel<M>(v, mn, inv_range));
  }
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
  }
}
} // namespace

const float* toFloat(
    const void* data, int64_t count, TensorElemType e, std::vector<float>& scratch)
{
  switch(e)
  {
    case TensorElemType::Float:
      return static_cast<const float*>(data);
    case TensorElemType::Float16:
    {
      const uint16_t* p = static_cast<const uint16_t*>(data);
      scratch.resize(count);
      for(int64_t i = 0; i < count; ++i)
        scratch[i] = halfToFloat(p[i]);
      return scratch.data();
    }
    case TensorElemType::BFloat16:
    {
      const uint16_t* p = static_cast<const uint16_t*>(data);
      scratch.resize(count);
      for(int64_t i = 0; i < count; ++i)
      {
        const uint32_t bits = (uint32_t)p[i] << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        scratch[i] = f;
      }
      return scratch.data();
    }
    case TensorElemType::Double:  copyAs<double>(data, count, scratch); return scratch.data();
    case TensorElemType::Uint8:   copyAs<uint8_t>(data, count, scratch); return scratch.data();
    case TensorElemType::Int8:    copyAs<int8_t>(data, count, scratch); return scratch.data();
    case TensorElemType::Uint16:  copyAs<uint16_t>(data, count, scratch); return scratch.data();
    case TensorElemType::Int16:   copyAs<int16_t>(data, count, scratch); return scratch.data();
    case TensorElemType::Uint32:  copyAs<uint32_t>(data, count, scratch); return scratch.data();
    case TensorElemType::Int32:   copyAs<int32_t>(data, count, scratch); return scratch.data();
    case TensorElemType::Uint64:  copyAs<uint64_t>(data, count, scratch); return scratch.data();
    case TensorElemType::Int64:   copyAs<int64_t>(data, count, scratch); return scratch.data();
    case TensorElemType::Bool:    copyAs<uint8_t>(data, count, scratch); return scratch.data();
    case TensorElemType::Unknown:
    default:
      // Unknown element types (fp8 / int4 / complex / string) have an element
      // size we don't know — reinterpreting `count` of them as float32 would
      // read up to 8x past the buffer. Emit a defined zero plane instead.
      scratch.assign(count > 0 ? (std::size_t)count : 0u, 0.f);
      return scratch.data();
  }
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
