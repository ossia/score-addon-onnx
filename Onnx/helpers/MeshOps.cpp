#include "MeshOps.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Onnx::MeshOps
{
void translate(
    std::span<const float> local, const float origin[3], std::span<float> out) noexcept
{
  const size_t n = std::min(local.size(), out.size()) / 3;
  const float ox = origin[0], oy = origin[1], oz = origin[2];
  // No __restrict: in place (out == local) is allowed.
  const float* s = local.data();
  float* d = out.data();
  for(size_t i = 0; i < n; i++)
  {
    d[3 * i + 0] = s[3 * i + 0] + ox;
    d[3 * i + 1] = s[3 * i + 1] + oy;
    d[3 * i + 2] = s[3 * i + 2] + oz;
  }
}

float* writeVertices(std::span<const float> verts, bool opengl, float* out) noexcept
{
  const size_t n = verts.size() / 3;
  const float* __restrict s = verts.data();
  float* __restrict d = out;
  const float sg = opengl ? -1.f : 1.f;
  for(size_t i = 0; i < n; i++)
  {
    d[3 * i + 0] = s[3 * i + 0];
    d[3 * i + 1] = sg * s[3 * i + 1];
    d[3 * i + 2] = sg * s[3 * i + 2];
  }
  return out + 3 * n;
}

float* writeTriangles(
    std::span<const float> verts, std::span<const int32_t> faces, bool opengl,
    float* out) noexcept
{
  // The faces were validated against the vertex count at load (Model::load),
  // and verts is always numVertices()*3 here: no per-index bound check.
  const float sg = opengl ? -1.f : 1.f;
  const float* v = verts.data();
  float* __restrict d = out;
  for(const int32_t idx : faces)
  {
    const size_t i = size_t(idx) * 3;
    d[0] = v[i];
    d[1] = sg * v[i + 1];
    d[2] = sg * v[i + 2];
    d += 3;
  }
  return d;
}

// ---------------------------------------------------------------------------
// Raster
// ---------------------------------------------------------------------------
void Raster::begin(int nw, int nh)
{
  const size_t n = size_t(std::max(nw, 0)) * size_t(std::max(nh, 0));
  if(nw != w || nh != h || depth.size() != n)
  {
    w = std::max(nw, 0);
    h = std::max(nh, 0);
    depth.assign(n, 0.f); // allocates only when the frame grows
    color.resize(n);
    span_lo.assign(size_t(h), w);
    span_hi.assign(size_t(h), 0);
  }
  else
  {
    // Only the previous frame's row spans can be non-empty.
    for(int y = y0; y < y1; y++)
    {
      const int a = span_lo[y], b = span_hi[y];
      if(b > a)
        std::fill_n(depth.data() + size_t(y) * w + a, b - a, 0.f);
      span_lo[y] = w;
      span_hi[y] = 0;
    }
  }
  x0 = y0 = 0;
  x1 = y1 = 0;
}

void Raster::draw(
    std::span<const float> verts, std::span<const int32_t> faces, float f, float cx,
    float cy, const uint8_t rgb[3])
{
  if(w <= 0 || h <= 0)
    return;
  constexpr float kNear = 0.05f; // metres
  const size_t nv = verts.size() / 3;
  if(screen.size() < nv * 3)
    screen.resize(nv * 3); // grows once per mesh size, then reused

  // Project every vertex once (u, v in px, 1/z), and bound the mesh on screen
  // for this frame's dirty rectangle.
  float umin = 1e30f, vmin = 1e30f, umax = -1e30f, vmax = -1e30f;
  for(size_t i = 0; i < nv; i++)
  {
    const float X = verts[3 * i], Y = verts[3 * i + 1], Z = verts[3 * i + 2];
    float* s = screen.data() + 3 * i;
    if(!(Z > kNear)) // also rejects NaN
    {
      s[2] = -1.f;
      continue;
    }
    const float iz = 1.f / Z;
    const float u = f * X * iz + cx, v = f * Y * iz + cy;
    // Non-finite (a NaN / inf X or Y) or absurdly far off screen: the vertex
    // is unusable, and its float -> int casts below would be undefined. Every
    // kept coordinate is within +-1e7 px, which ints hold exactly.
    if(!(std::fabs(u) < 1e7f) || !(std::fabs(v) < 1e7f))
    {
      s[2] = -1.f;
      continue;
    }
    s[0] = u;
    s[1] = v;
    s[2] = iz;
    umin = std::min(umin, s[0]);
    umax = std::max(umax, s[0]);
    vmin = std::min(vmin, s[1]);
    vmax = std::max(vmax, s[1]);
  }
  if(umax < umin)
    return;
  const int bx0 = std::clamp(int(std::floor(umin)), 0, w);
  const int bx1 = std::clamp(int(std::ceil(umax)) + 1, 0, w);
  const int by0 = std::clamp(int(std::floor(vmin)), 0, h);
  const int by1 = std::clamp(int(std::ceil(vmax)) + 1, 0, h);
  if(bx1 <= bx0 || by1 <= by0)
    return;
  if(empty())
  {
    x0 = bx0, x1 = bx1, y0 = by0, y1 = by1;
  }
  else
  {
    x0 = std::min(x0, bx0), x1 = std::max(x1, bx1);
    y0 = std::min(y0, by0), y1 = std::max(y1, by1);
  }

  const float* V = verts.data();
  const float* S = screen.data();
  const size_t nf = faces.size() / 3;
  for(size_t fi = 0; fi < nf; fi++)
  {
    const int32_t ia = faces[3 * fi], ib = faces[3 * fi + 1], ic = faces[3 * fi + 2];
    const float* sa = S + 3 * ia;
    const float* sb = S + 3 * ib;
    const float* sc = S + 3 * ic;
    if(sa[2] <= 0.f || sb[2] <= 0.f || sc[2] <= 0.f)
      continue;

    // Edge functions over the triangle's pixel bounding box, sampled at pixel
    // centres; 1/z is affine in screen space, so it interpolates exactly.
    const float area = (sb[0] - sa[0]) * (sc[1] - sa[1]) - (sb[1] - sa[1]) * (sc[0] - sa[0]);
    if(std::fabs(area) < 1e-6f)
      continue;
    const float inv_area = 1.f / area;
    // Only pixels whose CENTRE lies in the triangle's bounding box can pass
    // the edge test: most faces of a dense LOD cover no pixel centre at all
    // and are dropped here, before the 3D culling and shading below.
    const int tx0 = std::max(bx0, int(std::ceil(std::min({sa[0], sb[0], sc[0]}) - 0.5f)));
    const int tx1 = std::min(bx1 - 1, int(std::floor(std::max({sa[0], sb[0], sc[0]}) - 0.5f)));
    const int ty0 = std::max(by0, int(std::ceil(std::min({sa[1], sb[1], sc[1]}) - 0.5f)));
    const int ty1 = std::min(by1 - 1, int(std::floor(std::max({sa[1], sb[1], sc[1]}) - 0.5f)));
    if(tx1 < tx0 || ty1 < ty0)
      continue;

    // Back-face culling in 3D, exact under perspective: the camera sits at
    // the origin, so a face is seen from outside iff its outward normal
    // points back toward the camera, i.e. dot(n, a) < 0.
    const float* a = V + 3 * ia;
    const float* b = V + 3 * ib;
    const float* c = V + 3 * ic;
    const float e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
    const float e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
    const float nx = e1y * e2z - e1z * e2y;
    const float ny = e1z * e2x - e1x * e2z;
    const float nz = e1x * e2y - e1y * e2x;
    const float facing = -(nx * a[0] + ny * a[1] + nz * a[2]);
    if(!(facing > 0.f))
      continue;
    // Headlight Lambert: cos(normal, direction to the camera).
    const float nl = std::sqrt(nx * nx + ny * ny + nz * nz);
    const float al = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    const float cosv = nl > 0.f && al > 0.f ? std::min(1.f, facing / (nl * al)) : 0.f;
    const float shade = 0.3f + 0.7f * cosv;
    const uint32_t packed = uint32_t(rgb[0] * shade) | (uint32_t(rgb[1] * shade) << 8)
                            | (uint32_t(rgb[2] * shade) << 16);

    // w0 weights a (edge b->c), w1 weights b (edge c->a), w2 weights c.
    const float px0 = tx0 + 0.5f, py0 = ty0 + 0.5f;
    const float a0x = (sb[1] - sc[1]) * inv_area, a0y = (sc[0] - sb[0]) * inv_area;
    const float a1x = (sc[1] - sa[1]) * inv_area, a1y = (sa[0] - sc[0]) * inv_area;
    float w0r = ((sc[0] - sb[0]) * (py0 - sb[1]) - (sc[1] - sb[1]) * (px0 - sb[0])) * inv_area;
    float w1r = ((sa[0] - sc[0]) * (py0 - sc[1]) - (sa[1] - sc[1]) * (px0 - sc[0])) * inv_area;
    const float a2x = -a0x - a1x;
    for(int y = ty0; y <= ty1; y++, w0r += a0y, w1r += a1y)
    {
      // This row's pixels inside the triangle, from the three edge functions
      // (each is affine in x): a conservative [xs, xe] (one pixel of slack
      // for rounding), so the exact per-pixel test below only runs on the
      // covered span instead of the whole bounding-box row.
      float lo = 0.f, hi = float(tx1 - tx0);
      const float wr[3] = {w0r, w1r, 1.f - w0r - w1r};
      const float ax[3] = {a0x, a1x, a2x};
      for(int e = 0; e < 3; e++)
      {
        if(ax[e] > 0.f)
          lo = std::max(lo, -wr[e] / ax[e]);
        else if(ax[e] < 0.f)
          hi = std::min(hi, wr[e] / -ax[e]);
        else if(wr[e] < 0.f)
          hi = -1.f; // parallel edge, row outside
      }
      if(!(hi >= lo - 1.f)) // also NaN
        continue;
      const int xs = std::max(tx0, tx0 + int(std::floor(lo)) - 1);
      const int xe = std::min(tx1, tx0 + int(std::ceil(hi)) + 1);
      if(xe < xs)
        continue;
      span_lo[y] = std::min(span_lo[y], xs);
      span_hi[y] = std::max(span_hi[y], xe + 1);
      const float w0s = w0r + a0x * float(xs - tx0), w1s = w1r + a1x * float(xs - tx0);
      float* __restrict drow = depth.data() + size_t(y) * w + xs;
      uint32_t* __restrict crow = color.data() + size_t(y) * w + xs;
      const float za = sa[2], zb = sb[2], zc = sc[2];
      // Branch-free, so the span vectorizes: every lane computes, and only a
      // lane inside the triangle and in front of the depth buffer writes.
      const int n = xe - xs + 1;
      for(int i = 0; i < n; i++)
      {
        const float w0 = w0s + a0x * float(i), w1 = w1s + a1x * float(i);
        const float w2 = 1.f - w0 - w1;
        const float iz = w0 * za + w1 * zb + w2 * zc;
        const bool hit = (w0 >= 0.f) & (w1 >= 0.f) & (w2 >= 0.f) & (iz > drow[i]);
        drow[i] = hit ? iz : drow[i];
        crow[i] = hit ? packed : crow[i];
      }
    }
  }
}

void Raster::composite(uint8_t* rgba, int stride_bytes, float alpha) const noexcept
{
  if(empty() || !rgba)
    return;
  const int a = std::clamp(int(alpha * 256.f + 0.5f), 0, 256);
  for(int y = y0; y < y1; y++)
  {
    const int xa = std::max(x0, span_lo[y]), xb = std::min(x1, span_hi[y]);
    const float* drow = depth.data() + size_t(y) * w;
    const uint32_t* crow = color.data() + size_t(y) * w;
    uint8_t* p = rgba + size_t(y) * stride_bytes;
    // Branch-free (a covered pixel blends, an empty one keeps its value), so
    // the row vectorizes.
    for(int x = xa; x < xb; x++)
    {
      const uint32_t c = crow[x];
      const int k = drow[x] > 0.f ? a : 0, ik = 256 - k;
      uint8_t* q = p + size_t(x) * 4;
      q[0] = uint8_t((q[0] * ik + int(c & 0xff) * k) >> 8);
      q[1] = uint8_t((q[1] * ik + int((c >> 8) & 0xff) * k) >> 8);
      q[2] = uint8_t((q[2] * ik + int((c >> 16) & 0xff) * k) >> 8);
    }
  }
}
}
