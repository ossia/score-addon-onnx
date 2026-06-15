#pragma once
// Point-set I/O adapter for the GeometryProcessor node: maps an interleaved-xyz
// value payload <-> an ONNX point-cloud tensor, handling BOTH channel layouts
// ([1,N,3] "NPC" and [1,3,N] "NCP") plus optional extra per-point features.
//
// Dependency-free by design (no ORT, no Qt, no ossia): the geometry node builds
// the actual Ort::Value from the flat float buffer this produces, and the
// standalone test harness exercises layout detection + pack/unpack without a
// build of onnxruntime. Same spirit as ModelArchetype.hpp / TensorType.hpp.
//
// Real-time safety: PointLayout detection is pure arithmetic over the shape;
// pack()/unpack() write into caller-owned, preallocated buffers (resize only
// grows, reused across frames) and never allocate in steady state.
#include <Onnx/helpers/TensorType.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

namespace Onnx
{
// Where the coordinate axis (the "3") sits in a rank-3 point tensor, and how
// many extra per-point feature channels ride alongside xyz.
//   NPC: [B, N, C]  (channels-last)  — most PyTorch-exported PointNet variants
//   NCP: [B, C, N]  (channels-first) — original PointNet / many TF exports
// C == 3 is plain xyz; C in {4,6,9,...} carries xyz + (C-3) features per point.
enum class PointAxisLayout : uint8_t
{
  Unknown = 0,
  NPC, // [B, N, C] — coordinate is the last axis
  NCP, // [B, C, N] — coordinate is the middle axis
};

struct PointLayout
{
  PointAxisLayout layout = PointAxisLayout::Unknown;
  int batch = 1;       // leading dim (forced to 1 here; dynamic -> 1)
  int channels = 3;    // per-point components: 3 (xyz) or 3 + features
  int64_t count = 0;   // N points; <= 0 means dynamic (filled from the payload)
  int channel_axis = 1; // axis index holding `channels` (1 for NCP, 2 for NPC)
  int point_axis = 2;   // axis index holding `count`   (2 for NCP, 1 for NPC)

  bool valid() const noexcept { return layout != PointAxisLayout::Unknown; }
  bool dynamicCount() const noexcept { return count <= 0; }
  int features() const noexcept { return channels > 3 ? channels - 3 : 0; }
};

namespace detail
{
// A point-component axis is the small one (3, or a small xyz+feature count),
// the point axis is the large/dynamic one. We decide by inspecting which of the
// two trailing axes looks like "3 (+features)".
//
// Rules (in priority order), matching how ModelArchetype tags PointSet on rank-3
// [.,N,3] / [.,3,N]:
//   * exactly one axis == 3            -> that axis is the coordinate axis
//   * both could be a channel count    -> prefer the SMALLER as channels
//   * one is dynamic (<=0)             -> the dynamic one is the point count
inline bool isPlausibleChannelCount(int64_t c) noexcept
{
  // xyz, xyz+1..xyz+a-few; cap so we never mistake a big point count for a
  // channel count. Covers xyz(3), xyzr/xyzi(4), xyz+rgb(6), xyz+normal(6),
  // xyz+normal+rgb(9), and a little headroom.
  return c >= 3 && c <= 16;
}
} // namespace detail

// Detect the point layout from a rank-3 ONNX tensor shape. Returns an invalid
// PointLayout for shapes that are not point-set-like. `prefer_features` keeps a
// >3 channel count as features rather than collapsing to 3.
inline PointLayout detectPointLayout(const std::vector<int64_t>& shape) noexcept
{
  PointLayout pl;
  if(shape.size() != 3)
    return pl;

  const int64_t a0 = shape[0]; // batch
  const int64_t a1 = shape[1];
  const int64_t a2 = shape[2];

  pl.batch = (a0 > 0) ? (int)a0 : 1;

  const bool c1 = detail::isPlausibleChannelCount(a1);
  const bool c2 = detail::isPlausibleChannelCount(a2);

  auto setNPC = [&](int64_t chan, int64_t n)
  {
    pl.layout = PointAxisLayout::NPC;
    pl.channels = (int)chan;
    pl.count = n;
    pl.channel_axis = 2;
    pl.point_axis = 1;
  };
  auto setNCP = [&](int64_t chan, int64_t n)
  {
    pl.layout = PointAxisLayout::NCP;
    pl.channels = (int)chan;
    pl.count = n;
    pl.channel_axis = 1;
    pl.point_axis = 2;
  };

  // Exact "3" is the strongest signal for the coordinate axis.
  const bool three1 = (a1 == 3);
  const bool three2 = (a2 == 3);
  if(three2 && !three1)
  {
    setNPC(a2, a1);
    return pl;
  }
  if(three1 && !three2)
  {
    setNCP(a1, a2);
    return pl;
  }
  if(three1 && three2)
  {
    // [B,3,3] is ambiguous; default to channels-last (NPC) like most exports.
    setNPC(a2, a1);
    return pl;
  }

  // No exact 3: a dynamic axis is the point count, the other (if plausible) is
  // the channel/feature count.
  if(a2 <= 0 && c1)
  {
    setNCP(a1, a2);
    return pl;
  }
  if(a1 <= 0 && c2)
  {
    setNPC(a2, a1);
    return pl;
  }

  // Both concrete and neither is 3: default to channels-last (NPC), as the vast
  // majority of point-cloud exports are [B,N,C]. A real [B,C,N] uses C==3 (the
  // exact-3 branch above) or a large N (the c1-only branch below), so this
  // tie-break only affects tiny ambiguous shapes like [1,5,6].
  if(c1 && c2)
  {
    setNPC(a2, a1);
    return pl;
  }
  if(c1)
  {
    setNCP(a1, a2);
    return pl;
  }
  if(c2)
  {
    setNPC(a2, a1);
    return pl;
  }
  return pl; // not point-set-like
}

// Resolve the concrete element count for inference, filling a dynamic N from the
// number of points actually supplied. Returns the shape ORT should see (batch
// forced to 1) and the per-axis sizes via `resolved`.
inline std::vector<int64_t>
resolveShape(const PointLayout& pl, int64_t supplied_points)
{
  const int64_t n = pl.dynamicCount() ? supplied_points : pl.count;
  std::vector<int64_t> s(3);
  s[0] = 1;
  if(pl.layout == PointAxisLayout::NPC)
  {
    s[1] = n;
    s[2] = pl.channels;
  }
  else
  {
    s[1] = pl.channels;
    s[2] = n;
  }
  return s;
}

// Pack an interleaved-xyz(+features) payload into the flat tensor buffer in the
// memory order the model expects. `src` is interleaved per point:
//   [x0 y0 z0 (f...0) | x1 y1 z1 (f...1) | ...], src_stride floats per point.
// `npoints` points are written; the destination is sized n*channels and laid
// out NPC (contiguous per point) or NCP (planar per channel). Missing feature
// channels are zero-filled; extra source components past `channels` are dropped.
//
// RT-safe: `dst` is resized (grow-only) then written in place; no allocation in
// steady state once the buffer has reached its working size. `Buf` may be any
// contiguous float container (std::vector or boost::container::vector).
template <typename Buf>
inline void packPoints(
    const float* src, int64_t npoints, int src_stride, const PointLayout& pl,
    Buf& dst)
{
  const int C = pl.channels;
  const int64_t total = npoints * C;
  // Exact size so the element count matches the ORT tensor shape; shrinking an
  // already-reserved buffer doesn't reallocate, so this stays RT-safe.
  dst.resize((typename Buf::size_type)total);

  if(pl.layout == PointAxisLayout::NPC)
  {
    // Contiguous per point: copy min(C,stride), zero-fill the rest.
    const int copyc = src_stride < C ? src_stride : C;
    for(int64_t p = 0; p < npoints; ++p)
    {
      float* d = dst.data() + p * C;
      const float* s = src + p * src_stride;
      int c = 0;
      for(; c < copyc; ++c)
        d[c] = s[c];
      for(; c < C; ++c)
        d[c] = 0.f;
    }
  }
  else // NCP: planar, channel c occupies dst[c*npoints .. c*npoints+npoints)
  {
    for(int c = 0; c < C; ++c)
    {
      float* d = dst.data() + (int64_t)c * npoints;
      if(c < src_stride)
      {
        const float* s = src + c;
        for(int64_t p = 0; p < npoints; ++p)
          d[p] = s[p * src_stride];
      }
      else
      {
        std::memset(d, 0, sizeof(float) * (size_t)npoints);
      }
    }
  }
}

// Inverse of packPoints: read a flat point tensor (in `pl`'s layout) back into
// an interleaved-xyz(+features) payload, `out_stride` floats per point. Use this
// for point-cloud-output models (completion / upsampling / depth->cloud).
// RT-safe: `dst` is grow-only resized then written in place.
inline void unpackPoints(
    const float* tensor, int64_t npoints, const PointLayout& pl, int out_stride,
    std::vector<float>& dst)
{
  const int C = pl.channels;
  const int64_t total = npoints * out_stride;
  if((int64_t)dst.size() < total)
    dst.resize(total);

  const int copyc = out_stride < C ? out_stride : C;
  if(pl.layout == PointAxisLayout::NPC)
  {
    for(int64_t p = 0; p < npoints; ++p)
    {
      float* d = dst.data() + p * out_stride;
      const float* s = tensor + p * C;
      int c = 0;
      for(; c < copyc; ++c)
        d[c] = s[c];
      for(; c < out_stride; ++c)
        d[c] = 0.f;
    }
  }
  else // NCP planar
  {
    for(int64_t p = 0; p < npoints; ++p)
    {
      float* d = dst.data() + p * out_stride;
      int c = 0;
      for(; c < copyc; ++c)
        d[c] = tensor[(int64_t)c * npoints + p];
      for(; c < out_stride; ++c)
        d[c] = 0.f;
    }
  }
}

// Convert `count` raw elements of element type `e` to float, into `scratch`.
// Mirrors TensorToTexture::toFloat but is header-only & dependency-free so the
// geometry decode path (labels/seg/values) and the test harness can both use it
// without linking the imageops lib. fp16/bf16 are unpacked in software.
inline const float* tensorToFloat(
    const void* data, int64_t count, TensorElemType e,
    std::vector<float>& scratch)
{
  if(e == TensorElemType::Float)
    return static_cast<const float*>(data);

  if((int64_t)scratch.size() < count)
    scratch.resize(count);
  float* o = scratch.data();

  switch(e)
  {
    case TensorElemType::Double:
    {
      auto* p = static_cast<const double*>(data);
      for(int64_t i = 0; i < count; ++i)
        o[i] = (float)p[i];
      break;
    }
    case TensorElemType::Float16:
    {
      auto* p = static_cast<const uint16_t*>(data);
      for(int64_t i = 0; i < count; ++i)
      {
        const uint16_t h = p[i];
        const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1Fu;
        uint32_t mant = h & 0x3FFu;
        uint32_t f;
        if(exp == 0)
        {
          if(mant == 0)
            f = sign;
          else
          {
            exp = 127 - 15 + 1;
            while((mant & 0x400u) == 0)
            {
              mant <<= 1;
              --exp;
            }
            mant &= 0x3FFu;
            f = sign | (exp << 23) | (mant << 13);
          }
        }
        else if(exp == 0x1F)
        {
          f = sign | 0x7F800000u | (mant << 13);
        }
        else
        {
          f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        std::memcpy(o + i, &f, sizeof(float));
      }
      break;
    }
    case TensorElemType::BFloat16:
    {
      auto* p = static_cast<const uint16_t*>(data);
      for(int64_t i = 0; i < count; ++i)
      {
        const uint32_t f = (uint32_t)p[i] << 16;
        std::memcpy(o + i, &f, sizeof(float));
      }
      break;
    }
    case TensorElemType::Int64:
    {
      auto* p = static_cast<const int64_t*>(data);
      for(int64_t i = 0; i < count; ++i)
        o[i] = (float)p[i];
      break;
    }
    case TensorElemType::Int32:
    {
      auto* p = static_cast<const int32_t*>(data);
      for(int64_t i = 0; i < count; ++i)
        o[i] = (float)p[i];
      break;
    }
    case TensorElemType::Uint8:
    {
      auto* p = static_cast<const uint8_t*>(data);
      for(int64_t i = 0; i < count; ++i)
        o[i] = (float)p[i];
      break;
    }
    case TensorElemType::Int8:
    {
      auto* p = static_cast<const int8_t*>(data);
      for(int64_t i = 0; i < count; ++i)
        o[i] = (float)p[i];
      break;
    }
    case TensorElemType::Int16:
    {
      auto* p = static_cast<const int16_t*>(data);
      for(int64_t i = 0; i < count; ++i)
        o[i] = (float)p[i];
      break;
    }
    case TensorElemType::Uint16:
    {
      auto* p = static_cast<const uint16_t*>(data);
      for(int64_t i = 0; i < count; ++i)
        o[i] = (float)p[i];
      break;
    }
    case TensorElemType::Uint32:
    {
      auto* p = static_cast<const uint32_t*>(data);
      for(int64_t i = 0; i < count; ++i)
        o[i] = (float)p[i];
      break;
    }
    case TensorElemType::Uint64:
    {
      auto* p = static_cast<const uint64_t*>(data);
      for(int64_t i = 0; i < count; ++i)
        o[i] = (float)p[i];
      break;
    }
    case TensorElemType::Bool:
    {
      auto* p = static_cast<const uint8_t*>(data);
      for(int64_t i = 0; i < count; ++i)
        o[i] = p[i] ? 1.f : 0.f;
      break;
    }
    default:
      for(int64_t i = 0; i < count; ++i)
        o[i] = 0.f;
      break;
  }
  return o;
}

// Classify what a point-set model's selected output is, for routing to the
// geometry output (a cloud) vs the data output (labels / seg / values).
enum class GeomOutputKind : uint8_t
{
  Data,        // [.,K] class logits / [.,N] per-point seg or scalar field
  PointCloud,  // rank-3 [.,N,3]/[.,3,N] — completion / upsampling / depth->cloud
};

// Decide from the output shape: rank-3 point-set-like -> a cloud; everything
// else (rank-1/2 logits, per-point labels, SDF values) -> Data.
inline GeomOutputKind classifyGeomOutput(const std::vector<int64_t>& oshape)
{
  if(oshape.size() == 3 && detectPointLayout(oshape).valid())
    return GeomOutputKind::PointCloud;
  return GeomOutputKind::Data;
}
} // namespace Onnx
