#pragma once
// Dependency-free mirror of the ONNX/ORT tensor element types we handle. Keeping
// this out of <onnxruntime_cxx_api.h> lets ModelSpec and the tensor->texture
// (output) path stay ORT-free and standalone-compilable, in the same spirit as
// CoreTypes.hpp. The ORT enum is mapped to this in OnnxContext (the one place
// that already includes ORT).
#include <bit>
#include <cstdint>
#include <span>

namespace Onnx
{
enum class TensorElemType : uint8_t
{
  Unknown = 0,
  Float,    // float32
  Float16,  // IEEE half
  BFloat16, // bfloat16
  Double,   // float64
  Uint8,
  Int8,
  Uint16,
  Int16,
  Uint32,
  Int32,
  Uint64,
  Int64,
  Bool,
};

// Bytes per element (0 for Unknown). Float16/BFloat16 are 2 bytes.
inline constexpr int elemSize(TensorElemType t) noexcept
{
  switch(t)
  {
    case TensorElemType::Float:
    case TensorElemType::Uint32:
    case TensorElemType::Int32:
      return 4;
    case TensorElemType::Float16:
    case TensorElemType::BFloat16:
    case TensorElemType::Uint16:
    case TensorElemType::Int16:
      return 2;
    case TensorElemType::Double:
    case TensorElemType::Uint64:
    case TensorElemType::Int64:
      return 8;
    case TensorElemType::Uint8:
    case TensorElemType::Int8:
    case TensorElemType::Bool:
      return 1;
    case TensorElemType::Unknown:
    default:
      return 0;
  }
}
// The element count of a shape, dynamic (<= 0) dims counted as 1.
inline constexpr int64_t flatSize(std::span<const int64_t> shape) noexcept
{
  int64_t n = 1;
  for(auto d : shape)
    if(d > 0)
      n *= d;
  return n;
}

// IEEE binary16 <-> binary32, round-to-nearest-even; NaN stays NaN, values past
// the half range become inf, values below half the smallest subnormal become 0.
inline constexpr uint16_t floatToHalf(float f) noexcept
{
  const uint32_t x = std::bit_cast<uint32_t>(f);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t ax = x & 0x7FFFFFFFu;
  if(ax >= 0x7F800000u)
    return uint16_t(sign | 0x7C00u | (ax > 0x7F800000u ? 0x200u : 0u));
  if(ax >= 0x477FF000u) // >= 65520: rounds past the largest finite half
    return uint16_t(sign | 0x7C00u);

  uint32_t h, rem, halfway;
  if(ax < 0x38800000u) // below 2^-14: a subnormal half
  {
    if(ax < 0x33000000u) // below 2^-25
      return uint16_t(sign);
    const uint32_t e = ax >> 23;
    const uint32_t m = (ax & 0x7FFFFFu) | 0x800000u;
    const uint32_t shift = 126 - e;
    h = m >> shift;
    rem = m & ((1u << shift) - 1u);
    halfway = 1u << (shift - 1u);
  }
  else
  {
    h = (((ax >> 23) - 112u) << 10) | ((ax >> 13) & 0x3FFu);
    rem = ax & 0x1FFFu;
    halfway = 0x1000u;
  }
  if(rem > halfway || (rem == halfway && (h & 1u)))
    ++h; // a carry into the exponent is the correctly rounded value
  return uint16_t(sign | h);
}

inline constexpr float halfToFloat(uint16_t h) noexcept
{
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if(exp == 0x1Fu)
    bits = sign | 0x7F800000u | (mant << 13);
  else if(exp != 0)
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  else if(mant == 0)
    bits = sign;
  else
  {
    uint32_t e = 113;
    while((mant & 0x400u) == 0)
    {
      mant <<= 1;
      --e;
    }
    bits = sign | (e << 23) | ((mant & 0x3FFu) << 13);
  }
  return std::bit_cast<float>(bits);
}

// binary32 -> bfloat16, round-to-nearest-even; NaN stays NaN.
inline constexpr uint16_t floatToBFloat16(float f) noexcept
{
  const uint32_t x = std::bit_cast<uint32_t>(f);
  if((x & 0x7FFFFFFFu) > 0x7F800000u)
    return uint16_t((x >> 16) | 0x40u);
  return uint16_t((x + 0x7FFFu + ((x >> 16) & 1u)) >> 16);
}

inline constexpr float bfloat16ToFloat(uint16_t b) noexcept
{
  return std::bit_cast<float>(uint32_t(b) << 16);
}
} // namespace Onnx
