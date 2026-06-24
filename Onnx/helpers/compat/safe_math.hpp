#pragma once
// Vendored, self-contained copy of the two ossia::safe_* helpers used by the
// pose detector (from libossia, ossia/math/safe_math.hpp). Used only for
// standalone builds, where libossia is not on the include path; the score
// build picks the real header via __has_include in PoseDetector_internal.hpp.
//
// std::isnan / std::isinf are constant-folded to 0 under -ffast-math
// (__FINITE_MATH_ONLY__), so these bit-pattern fallbacks keep working when the
// pipeline is compiled fast-math. Keep API-compatible with ossia.
#include <cmath>
#include <cstdint>

namespace ossia
{

inline bool safe_isnan(double val) noexcept
{
#if __FINITE_MATH_ONLY__
#if defined(_MSC_VER)
  return std::isnan(val);
#elif defined(__APPLE__)
  return __isnand(val);
#elif defined(__EMSCRIPTEN__)
  return __fpclassifyl(val) == FP_NAN;
#else
  // On gcc / clang, with -ffast-math, std::isnan always returns 0
  union
  {
    double fp;
    uint64_t bits;
  } num{.fp = val};

  return ((unsigned)(num.bits >> 32) & 0x7fffffff) + ((unsigned)num.bits != 0)
         > 0x7ff00000;
#endif
#else
  return std::isnan(val);
#endif
}

inline bool safe_isinf(double val) noexcept
{
#if __FINITE_MATH_ONLY__
#if defined(_MSC_VER)
  return std::isinf(val);
#elif defined(__APPLE__)
  return __isinfd(val);
#elif defined(__EMSCRIPTEN__)
  return __fpclassifyl(val) == FP_INFINITE;
#else
  // On gcc / clang, with -ffast-math, std::isinf always returns 0
  union
  {
    double fp;
    uint64_t bits;
  } num{.fp = val};

  return ((unsigned)(num.bits >> 32) & 0x7fffffff) == 0x7ff00000
         && (unsigned)num.bits == 0;
#endif
#else
  return std::isinf(val);
#endif
}

}
