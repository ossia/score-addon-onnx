#pragma once
// Dependency-free value types for the (eventually Qt/ossia-free) ONNX core:
// 2D point, rectangle, and a float RGBA color, plus the color helpers the pose
// palette needs. Replaces QPointF / QRectF / QColor in the drawing + decode
// paths so they no longer pull in Qt.
#include <algorithm>
#include <cmath>

namespace Onnx
{
struct Vec2
{
  float x{}, y{};
};

struct Rect
{
  float x{}, y{}, w{}, h{};
};

// Color components in [0,1].
struct Rgba
{
  float r{}, g{}, b{}, a{1.f};
};

// From 8-bit channels (the palette is authored in 0..255).
inline Rgba rgb8(int r, int g, int b, float a = 1.f)
{
  return {r / 255.f, g / 255.f, b / 255.f, a};
}

inline Rgba withAlpha(Rgba c, float a)
{
  c.a = a;
  return c;
}

// Approximates QColor::lighter(pct): scale brightness by pct/100, clamped.
inline Rgba lighter(Rgba c, float pct)
{
  const float f = pct / 100.f;
  return {
      std::min(1.f, c.r * f), std::min(1.f, c.g * f), std::min(1.f, c.b * f),
      c.a};
}

// HSV -> RGB, all components in [0,1] (hue wraps).
inline Rgba hsv(float h, float s, float v, float a = 1.f)
{
  const float hh = std::fmod(h < 0.f ? h + 1.f : h, 1.f) * 6.f;
  const int i = static_cast<int>(hh);
  const float f = hh - i;
  const float p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
  switch(i % 6)
  {
    case 0: return {v, t, p, a};
    case 1: return {q, v, p, a};
    case 2: return {p, v, t, a};
    case 3: return {p, q, v, a};
    case 4: return {t, p, v, a};
    default: return {v, p, q, a};
  }
}
} // namespace Onnx
