// Element type conversions shared by the nodes (fp16 / bf16 inputs and
// outputs, and every output type read back as float).
#include <Onnx/helpers/TensorToTexture.hpp>

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace Onnx;

TEST_CASE("fp16: every half survives half -> float -> half", "[onnx][tensortype]")
{
  for(uint32_t h = 0; h < 0x10000u; ++h)
  {
    const float f = halfToFloat(uint16_t(h));
    const bool nan = (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu);
    if(nan)
    {
      CHECK(std::isnan(f));
      CHECK(std::isnan(halfToFloat(floatToHalf(f))));
    }
    else
    {
      REQUIRE(floatToHalf(f) == h);
    }
  }
}

TEST_CASE("fp16: round to nearest even, overflow and underflow", "[onnx][tensortype]")
{
  CHECK(floatToHalf(0.f) == 0x0000);
  CHECK(floatToHalf(-0.f) == 0x8000);
  CHECK(floatToHalf(1.f) == 0x3C00);
  CHECK(floatToHalf(-2.f) == 0xC000);
  CHECK(floatToHalf(65504.f) == 0x7BFF);
  CHECK(floatToHalf(65519.f) == 0x7BFF);
  CHECK(floatToHalf(65520.f) == 0x7C00);
  CHECK(floatToHalf(1e10f) == 0x7C00);
  CHECK(floatToHalf(-1e10f) == 0xFC00);
  CHECK(floatToHalf(std::numeric_limits<float>::infinity()) == 0x7C00);
  CHECK((floatToHalf(std::numeric_limits<float>::quiet_NaN()) & 0x7FFFu) > 0x7C00u);

  // 1 + 2^-11 is halfway between 1 and the next half: ties go to the even 1.
  CHECK(floatToHalf(1.f + std::ldexp(1.f, -11)) == 0x3C00);
  // 1 + 3 * 2^-11 is halfway between two odd/even neighbours: up to even.
  CHECK(floatToHalf(1.f + 3.f * std::ldexp(1.f, -11)) == 0x3C02);
  // Just above halfway rounds up.
  CHECK(floatToHalf(std::nextafter(1.f + std::ldexp(1.f, -11), 2.f)) == 0x3C01);

  // Subnormals: the smallest one is 2^-24; half of it ties to 0, just above
  // it rounds to the smallest.
  CHECK(floatToHalf(std::ldexp(1.f, -24)) == 0x0001);
  CHECK(floatToHalf(std::ldexp(1.f, -25)) == 0x0000);
  CHECK(floatToHalf(std::nextafter(std::ldexp(1.f, -25), 1.f)) == 0x0001);
  CHECK(floatToHalf(1e-30f) == 0x0000);
  // The largest subnormal rounding up into the smallest normal.
  CHECK(floatToHalf(std::ldexp(1.f, -14) - std::ldexp(1.f, -26)) == 0x0400);
}

#if defined(__FLT16_MAX__)
TEST_CASE("fp16: matches the compiler's _Float16 conversion", "[onnx][tensortype]")
{
  // Every 2^-7 ulp step of the float range around the half range.
  for(uint32_t x = 0; x < 0x7F800000u; x += 0x81u)
  {
    const float f = std::bit_cast<float>(x);
    const auto ref = std::bit_cast<uint16_t>(static_cast<_Float16>(f));
    REQUIRE(floatToHalf(f) == ref);
    REQUIRE(floatToHalf(-f) == uint16_t(ref | 0x8000u));
  }
}
#endif

TEST_CASE("bf16: round to nearest even, NaN kept", "[onnx][tensortype]")
{
  CHECK(floatToBFloat16(1.f) == 0x3F80);
  CHECK(bfloat16ToFloat(0x3F80) == 1.f);
  // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to even (1).
  CHECK(floatToBFloat16(1.f + std::ldexp(1.f, -8)) == 0x3F80);
  CHECK(floatToBFloat16(1.f + 3.f * std::ldexp(1.f, -8)) == 0x3F82);
  CHECK(std::isnan(bfloat16ToFloat(floatToBFloat16(std::numeric_limits<float>::quiet_NaN()))));
  // A NaN whose payload is only in the low bits must not turn into inf.
  CHECK(std::isnan(bfloat16ToFloat(floatToBFloat16(std::bit_cast<float>(0x7F800001u)))));
  for(uint32_t b = 0; b < 0x10000u; ++b)
  {
    const float f = bfloat16ToFloat(uint16_t(b));
    if(!std::isnan(f))
      REQUIRE(floatToBFloat16(f) == b);
  }
}

TEST_CASE("toFloat: every element type reads back as float", "[onnx][tensortype]")
{
  std::vector<float> scratch;

  const float f32[3]{1.5f, -2.f, 3.f};
  CHECK(toFloat(f32, 3, TensorElemType::Float, scratch) == f32);

  const uint16_t f16[3]{0x3C00, 0xC000, 0x7C00};
  const float* r = toFloat(f16, 3, TensorElemType::Float16, scratch);
  CHECK(r[0] == 1.f);
  CHECK(r[1] == -2.f);
  CHECK(std::isinf(r[2]));

  const uint16_t b16[2]{0x3F80, 0xC000};
  r = toFloat(b16, 2, TensorElemType::BFloat16, scratch);
  CHECK(r[0] == 1.f);
  CHECK(r[1] == -2.f);

  const int64_t i64[2]{-7, 1 << 20};
  r = toFloat(i64, 2, TensorElemType::Int64, scratch);
  CHECK(r[0] == -7.f);
  CHECK(r[1] == float(1 << 20));

  const uint8_t u8[2]{0, 255};
  r = toFloat(u8, 2, TensorElemType::Uint8, scratch);
  CHECK(r[1] == 255.f);

  const int8_t i8[1]{-128};
  CHECK(toFloat(i8, 1, TensorElemType::Int8, scratch)[0] == -128.f);

  const uint8_t b[3]{0, 1, 7};
  r = toFloat(b, 3, TensorElemType::Bool, scratch);
  CHECK(r[0] == 0.f);
  CHECK(r[1] == 1.f);
  CHECK(r[2] == 1.f);

  const double d[1]{0.25};
  CHECK(toFloat(d, 1, TensorElemType::Double, scratch)[0] == 0.25f);

  // An element type of unknown size is never read.
  r = toFloat(nullptr, 4, TensorElemType::Unknown, scratch);
  for(int i = 0; i < 4; ++i)
    CHECK(r[i] == 0.f);
}
