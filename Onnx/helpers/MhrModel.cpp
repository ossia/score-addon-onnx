#include "MhrModel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <type_traits>
#include <unordered_map>

// MHR_NO_SIMD forces the portable scalar kernels (used to test them).
#if defined(MHR_NO_SIMD)
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>
#define MHR_SIMD_SSE 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MHR_SIMD_NEON 1
#endif

namespace Onnx::Mhr
{
namespace
{
// ---------------------------------------------------------------------------
// .mhrbin parsing
// ---------------------------------------------------------------------------
constexpr char kMagic[8] = {'M', 'H', 'R', 'B', 'I', 'N', '1', '\0'};
constexpr uint32_t kFormatVersion = 1;
enum : uint32_t
{
  DT_F32 = 1,
  DT_I32 = 2,
  DT_U8 = 3
};

struct ArrayDesc
{
  uint32_t dtype = 0;
  uint32_t ndim = 0;
  uint32_t dims[4]{};
  uint64_t offset = 0;
  uint64_t bytes = 0;
  uint64_t count() const noexcept
  {
    // Saturating: a hostile descriptor must not wrap around to a size that
    // happens to match its byte count.
    uint64_t n = 1;
    for(uint32_t i = 0; i < ndim && i < 4; i++)
    {
      if(dims[i] != 0 && n > UINT64_MAX / 8 / dims[i])
        return UINT64_MAX / 8;
      n *= dims[i];
    }
    return n;
  }
};

template <typename T>
T readLE(const char* p) noexcept
{
  // The file is little-endian and so is every platform score ships on.
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

struct Parser
{
  std::span<const char> bytes;
  std::unordered_map<std::string, ArrayDesc> arrays;
  std::string error;

  bool fail(std::string msg)
  {
    if(error.empty())
      error = std::move(msg);
    return false;
  }

  bool parseHeader()
  {
    if(bytes.size() < 24 || std::memcmp(bytes.data(), kMagic, 8) != 0)
      return fail("not an .mhrbin file (bad magic)");
    const auto version = readLE<uint32_t>(bytes.data() + 8);
    if(version != kFormatVersion)
      return fail("unsupported .mhrbin version " + std::to_string(version));
    const auto n = readLE<uint32_t>(bytes.data() + 12);
    if(uint64_t(n) * 64 + 24 > bytes.size())
      return fail("truncated array table");
    for(uint32_t i = 0; i < n; i++)
    {
      const char* d = bytes.data() + 24 + 64 * i;
      std::string name(d, strnlen(d, 24));
      ArrayDesc a;
      a.dtype = readLE<uint32_t>(d + 24);
      a.ndim = readLE<uint32_t>(d + 28);
      for(int k = 0; k < 4; k++)
        a.dims[k] = readLE<uint32_t>(d + 32 + 4 * k);
      a.offset = readLE<uint64_t>(d + 48);
      a.bytes = readLE<uint64_t>(d + 56);
      const uint64_t elt = a.dtype == DT_U8 ? 1 : 4;
      if(a.ndim < 1 || a.ndim > 4 || (a.dtype != DT_F32 && a.dtype != DT_I32 && a.dtype != DT_U8)
         || a.count() * elt != a.bytes || a.offset > bytes.size()
         || a.bytes > bytes.size() - a.offset)
        return fail("corrupt descriptor for array '" + name + "'");
      arrays.emplace(std::move(name), a);
    }
    return true;
  }

  bool has(const std::string& name) const { return arrays.contains(name); }

  // Copy array `name` into `out`, checking its type and total element count
  // (`expect` < 0: any count; the actual count is returned through `out`).
  template <typename T>
  bool get(const std::string& name, std::vector<T>& out, int64_t expect = -1)
  {
    auto it = arrays.find(name);
    if(it == arrays.end())
      return fail("missing array '" + name + "'");
    const ArrayDesc& a = it->second;
    constexpr uint32_t want = std::is_same_v<T, float> ? DT_F32 : std::is_same_v<T, int32_t> ? DT_I32 : DT_U8;
    if(a.dtype != want)
      return fail("array '" + name + "' has the wrong type");
    if(expect >= 0 && int64_t(a.count()) != expect)
      return fail(
          "array '" + name + "' has " + std::to_string(a.count()) + " elements, expected "
          + std::to_string(expect));
    out.resize(a.count());
    if(a.bytes)
      std::memcpy(out.data(), bytes.data() + a.offset, a.bytes);
    return true;
  }
};

bool inRange(std::span<const int32_t> v, int32_t lo, int32_t hi) noexcept
{
  for(int32_t x : v)
    if(x < lo || x >= hi)
      return false;
  return true;
}

// CSR sanity: monotone row pointer that ends at nnz, columns in [0, cols).
bool validCsr(
    const std::vector<int32_t>& rowptr, const std::vector<int32_t>& col, size_t nnz_vals,
    int32_t cols) noexcept
{
  if(rowptr.empty() || rowptr.front() != 0 || size_t(rowptr.back()) != col.size()
     || col.size() != nnz_vals)
    return false;
  for(size_t i = 1; i < rowptr.size(); i++)
    if(rowptr[i] < rowptr[i - 1])
      return false;
  return inRange(col, 0, cols);
}

bool loadBasis(
    Parser& p, const std::string& prefix, int32_t n_vertices, int32_t n_components,
    SparseBasis& b)
{
  std::vector<int32_t> groups;
  if(!p.get(prefix + "_groups", groups) || groups.size() % 3 != 0)
    return p.fail("bad '" + prefix + "_groups'");
  if(!p.get(prefix + "_supp", b.support_idx) || !p.get(prefix + "_data", b.data))
    return false;
  if(!inRange(b.support_idx, 0, n_vertices))
    return p.fail("'" + prefix + "_supp' indexes past the vertex count");
  uint64_t supp = 0, data = 0;
  b.groups.clear();
  b.num_components = n_components;
  b.max_support = 0;
  for(size_t g = 0; g < groups.size(); g += 3)
  {
    SparseBasis::Group gr;
    gr.first = groups[g];
    gr.count = groups[g + 1];
    gr.support = groups[g + 2];
    // 64-bit sum: first + count must not wrap past INT32_MAX into range.
    if(gr.first < 0 || gr.count < 0 || gr.support < 0
       || int64_t(gr.first) + int64_t(gr.count) > int64_t(n_components))
      return p.fail("'" + prefix + "_groups' component range out of bounds");
    if(supp > UINT32_MAX)
      return p.fail("'" + prefix + "_groups' support too large");
    gr.supp_offset = uint32_t(supp);
    gr.data_offset = data;
    supp += uint64_t(gr.support);
    data += uint64_t(gr.support) * gr.count * 3;
    b.max_support = std::max(b.max_support, gr.support);
    b.groups.push_back(gr);
  }
  if(supp != b.support_idx.size() || data != b.data.size())
    return p.fail("'" + prefix + "' group sizes do not match its data");
  return true;
}

bool loadVertexSet(
    Parser& p, const std::string& prefix, int32_t n, int32_t k, int32_t n_hidden,
    bool expression, VertexSet& vs)
{
  vs.count = n;
  vs.influences = k;
  if(!p.get(prefix + ".base", vs.base, int64_t(n) * 3)
     || !p.get(prefix + ".skin_idx", vs.skin_idx, int64_t(n) * k)
     || !p.get(prefix + ".skin_w", vs.skin_w, int64_t(n) * k))
    return false;
  if(!inRange(vs.skin_idx, 0, kNumJoints))
    return p.fail("'" + prefix + ".skin_idx' references a missing joint");
  if(!loadBasis(p, prefix + ".id", n, kNumIdentity, vs.identity))
    return false;
  if(expression && !loadBasis(p, prefix + ".ex", n, kNumIdentity + kNumExpression, vs.expression))
    return false;
  // Expression groups are stored with absolute component indices 45..116.
  return loadBasis(p, prefix + ".co", n, n_hidden, vs.correctives);
}

std::vector<std::string> splitNames(const std::vector<uint8_t>& blob)
{
  std::vector<std::string> out;
  std::string cur;
  for(uint8_t c : blob)
  {
    if(c == 0)
    {
      out.push_back(std::move(cur));
      cur.clear();
    }
    else
      cur.push_back(char(c));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Small float64 similarity-transform algebra (pymomentum's skel_state:
// t[3], q[4] = (x, y, z, w), s).
// ---------------------------------------------------------------------------
struct Quat
{
  double x, y, z, w;
};

inline Quat qmul(const Quat& a, const Quat& b) noexcept
{
  return {
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

inline Quat qnormalize(const Quat& q) noexcept
{
  const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  const double inv = n > 0 ? 1.0 / n : 0.0;
  return {q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

// v + 2 (w (a x v) + a x (a x v)), a = (x, y, z): pymomentum's rotate_vector.
inline void qrotate(const Quat& q, const double v[3], double out[3]) noexcept
{
  const double av[3]
      = {q.y * v[2] - q.z * v[1], q.z * v[0] - q.x * v[2], q.x * v[1] - q.y * v[0]};
  const double aav[3]
      = {q.y * av[2] - q.z * av[1], q.z * av[0] - q.x * av[2], q.x * av[1] - q.y * av[0]};
  for(int i = 0; i < 3; i++)
    out[i] = v[i] + 2.0 * (av[i] * q.w + aav[i]);
}

// pymomentum euler_xyz_to_quaternion: R = Rz(z) Ry(y) Rx(x). The half-angle
// sines/cosines are float32 like the reference's (the local state is float32
// there; only the chain composition runs in float64).
inline Quat eulerXYZ(float rx, float ry, float rz) noexcept
{
  const double cy = std::cos(rz * 0.5f), sy = std::sin(rz * 0.5f);
  const double cp = std::cos(ry * 0.5f), sp = std::sin(ry * 0.5f);
  const double cr = std::cos(rx * 0.5f), sr = std::sin(rx * 0.5f);
  return {
      sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
      cr * cp * cy + sr * sp * sy};
}

// ---------------------------------------------------------------------------
// acc[0..24) = sum_a cs[a] * ds[a][off .. off+24): the inner kernel of
// applyBasis, the hottest loop of the pass. Written with explicit 4-wide
// registers because compilers at -O2 keep a float[24] accumulator in memory
// (a load + store per component).
// ---------------------------------------------------------------------------
inline void accumulateChunk(
    const float* cs, const float* const* ds, int na, size_t off, float* __restrict acc) noexcept
{
#if defined(MHR_SIMD_SSE)
  __m128 a0 = _mm_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0;
  for(int a = 0; a < na; a++)
  {
    const __m128 c = _mm_set1_ps(cs[a]);
    const float* d = ds[a] + off;
    a0 = _mm_add_ps(a0, _mm_mul_ps(c, _mm_loadu_ps(d + 0)));
    a1 = _mm_add_ps(a1, _mm_mul_ps(c, _mm_loadu_ps(d + 4)));
    a2 = _mm_add_ps(a2, _mm_mul_ps(c, _mm_loadu_ps(d + 8)));
    a3 = _mm_add_ps(a3, _mm_mul_ps(c, _mm_loadu_ps(d + 12)));
    a4 = _mm_add_ps(a4, _mm_mul_ps(c, _mm_loadu_ps(d + 16)));
    a5 = _mm_add_ps(a5, _mm_mul_ps(c, _mm_loadu_ps(d + 20)));
  }
  _mm_store_ps(acc + 0, a0);
  _mm_store_ps(acc + 4, a1);
  _mm_store_ps(acc + 8, a2);
  _mm_store_ps(acc + 12, a3);
  _mm_store_ps(acc + 16, a4);
  _mm_store_ps(acc + 20, a5);
#elif defined(MHR_SIMD_NEON)
  float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0;
  for(int a = 0; a < na; a++)
  {
    const float c = cs[a];
    const float* d = ds[a] + off;
    a0 = vmlaq_n_f32(a0, vld1q_f32(d + 0), c);
    a1 = vmlaq_n_f32(a1, vld1q_f32(d + 4), c);
    a2 = vmlaq_n_f32(a2, vld1q_f32(d + 8), c);
    a3 = vmlaq_n_f32(a3, vld1q_f32(d + 12), c);
    a4 = vmlaq_n_f32(a4, vld1q_f32(d + 16), c);
    a5 = vmlaq_n_f32(a5, vld1q_f32(d + 20), c);
  }
  vst1q_f32(acc + 0, a0);
  vst1q_f32(acc + 4, a1);
  vst1q_f32(acc + 8, a2);
  vst1q_f32(acc + 12, a3);
  vst1q_f32(acc + 16, a4);
  vst1q_f32(acc + 20, a5);
#else
  for(int i = 0; i < 24; i++)
    acc[i] = 0.f;
  for(int a = 0; a < na; a++)
  {
    const float c = cs[a];
    const float* d = ds[a] + off;
    for(int i = 0; i < 24; i++)
      acc[i] += c * d[i];
  }
#endif
}

// ---------------------------------------------------------------------------
// Blend bases: dst[support] += sum_k coeff[k] * slab[k]
// ---------------------------------------------------------------------------
// Register-blocked: the group's non-zero coefficients are gathered first, then
// the support is walked 8 vertices (24 floats, 6 SSE / 3 AVX registers) at a
// time, accumulating every active component into those registers before one
// scatter-add per vertex. Each slab float is read once, and nothing but the
// final sums touches memory.
void applyBasis(const SparseBasis& b, const float* coeffs, float* __restrict dst) noexcept
{
  constexpr int kBatch = 32; // active components per pass (MHR groups: <= 24, or 72)
  constexpr int kChunk = 8;  // vertices per register block
  static_assert(kChunk * 3 == 24, "accumulateChunk() works on 24 floats");
  float cs[kBatch];
  const float* ds[kBatch];
  for(const auto& g : b.groups)
  {
    if(g.support == 0)
      continue;
    const float* c = coeffs + g.first;
    const float* slab = b.data.data() + g.data_offset;
    const size_t stride = size_t(g.support) * 3;
    const int32_t* sup = b.support_idx.data() + g.supp_offset;
    for(int k0 = 0; k0 < g.count; k0 += kBatch)
    {
      int na = 0;
      for(int k = k0; k < std::min(g.count, k0 + kBatch); k++)
        if(c[k] != 0.f)
        {
          cs[na] = c[k];
          ds[na] = slab + size_t(k) * stride;
          na++;
        }
      if(na == 0)
        continue;

      int s = 0;
      for(; s + kChunk <= g.support; s += kChunk)
      {
        alignas(16) float acc[kChunk * 3];
        accumulateChunk(cs, ds, na, size_t(s) * 3, acc);
        for(int t = 0; t < kChunk; t++)
        {
          float* o = dst + size_t(sup[s + t]) * 3;
          o[0] += acc[t * 3 + 0];
          o[1] += acc[t * 3 + 1];
          o[2] += acc[t * 3 + 2];
        }
      }
      for(; s < g.support; s++)
      {
        float x = 0.f, y = 0.f, z = 0.f;
        for(int a = 0; a < na; a++)
        {
          const float* d = ds[a] + size_t(s) * 3;
          x += cs[a] * d[0];
          y += cs[a] * d[1];
          z += cs[a] * d[2];
        }
        float* o = dst + size_t(sup[s]) * 3;
        o[0] += x;
        o[1] += y;
        o[2] += z;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Linear blend skinning with the vision-frame flip folded into `skin`.
// ---------------------------------------------------------------------------
// Each joint's 3x4 transform is stored column-major with a padding lane,
// 16 floats: c0 | c1 | c2 | t, each (x, y, z, 0). Then
//   out = sum_k w_k (c0_k x + c1_k y + c2_k z + t_k)
// is a sequence of 4-wide multiply-adds that the SLP vectoriser maps to single
// SSE/NEON instructions even at -O2. Padding influences carry weight 0 and are
// processed branch-free.
template <int K>
void skinFixed(
    const VertexSet& vs, const float* __restrict skin, const float* __restrict in,
    float* __restrict out) noexcept
{
  const int32_t* idx = vs.skin_idx.data();
  const float* wts = vs.skin_w.data();
  for(int v = 0; v < vs.count; v++)
  {
    const float x = in[v * 3], y = in[v * 3 + 1], z = in[v * 3 + 2];
    float acc[4]{};
    for(int k = 0; k < K; k++)
    {
      const float w = wts[v * K + k];
      const float* M = skin + size_t(idx[v * K + k]) * 16;
      for(int i = 0; i < 4; i++)
        acc[i] += w * (M[i] * x + M[4 + i] * y + M[8 + i] * z + M[12 + i]);
    }
    out[v * 3 + 0] = acc[0];
    out[v * 3 + 1] = acc[1];
    out[v * 3 + 2] = acc[2];
  }
}

void skinGeneric(const VertexSet& vs, const float* skin, const float* in, float* out) noexcept
{
  const int K = vs.influences;
  for(int v = 0; v < vs.count; v++)
  {
    const float x = in[v * 3], y = in[v * 3 + 1], z = in[v * 3 + 2];
    float acc[4]{};
    for(int k = 0; k < K; k++)
    {
      const float w = vs.skin_w[size_t(v) * K + k];
      const float* M = skin + size_t(vs.skin_idx[size_t(v) * K + k]) * 16;
      for(int i = 0; i < 4; i++)
        acc[i] += w * (M[i] * x + M[4 + i] * y + M[8 + i] * z + M[12 + i]);
    }
    out[v * 3 + 0] = acc[0];
    out[v * 3 + 1] = acc[1];
    out[v * 3 + 2] = acc[2];
  }
}

void skinVertices(const VertexSet& vs, const float* skin, const float* in, float* out) noexcept
{
  switch(vs.influences)
  {
    case 4:
      return skinFixed<4>(vs, skin, in, out);
    case 8:
      return skinFixed<8>(vs, skin, in, out);
    default:
      return skinGeneric(vs, skin, in, out);
  }
}

// Corrective layer 1 over one 24-unit block: h = relu(sum_c f[col c] * W[c]),
// the same 24-float register kernel as the blend bases, over the block's
// non-zero features only.
void denseBlock24(
    float* __restrict h, const int32_t* cols, const float* __restrict w, int ncols,
    const float* __restrict feat) noexcept
{
  constexpr int kMaxCols = 64; // MHR blocks read 12..48 columns
  float cs[kMaxCols];
  const float* ds[kMaxCols];
  alignas(16) float part[24];
  float acc[24]{};
  for(int c0 = 0; c0 < ncols; c0 += kMaxCols)
  {
    int na = 0;
    for(int c = c0; c < std::min(ncols, c0 + kMaxCols); c++)
    {
      const float f = feat[cols[c]];
      if(f != 0.f)
      {
        cs[na] = f;
        ds[na] = w + size_t(c) * 24;
        na++;
      }
    }
    if(na == 0)
      continue;
    accumulateChunk(cs, ds, na, 0, part);
    for(int u = 0; u < 24; u++)
      acc[u] += part[u];
  }
  for(int u = 0; u < 24; u++)
    h[u] = acc[u] > 0.f ? acc[u] : 0.f;
}

void denseBlockN(
    float* __restrict h, const int32_t* cols, const float* __restrict w, int ncols, int n,
    const float* __restrict feat) noexcept
{
  for(int u = 0; u < n; u++)
    h[u] = 0.f;
  for(int c = 0; c < ncols; c++)
  {
    const float f = feat[cols[c]];
    if(f == 0.f)
      continue;
    const float* row = w + size_t(c) * n;
    for(int u = 0; u < n; u++)
      h[u] += f * row[u];
  }
  for(int u = 0; u < n; u++)
    h[u] = h[u] > 0.f ? h[u] : 0.f;
}

// rest = base + identity (+ expression): recomputed only when the
// coefficients change (typically once per tracked person, or when the
// regressor's shape estimate moves).
void blendRest(
    const VertexSet& vs, const float* identity, const float* expr_full,
    std::vector<float>& rest) noexcept
{
  std::copy(vs.base.begin(), vs.base.end(), rest.begin());
  applyBasis(vs.identity, identity, rest.data());
  if(expr_full && !vs.expression.empty())
    applyBasis(vs.expression, expr_full, rest.data());
}
} // namespace

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
std::optional<Model> Model::load(std::span<const char> bytes, std::string* err)
{
  Parser p{bytes, {}, {}};
  auto failed = [&]() -> std::optional<Model> {
    if(err)
      *err = p.error.empty() ? "invalid .mhrbin" : p.error;
    return std::nullopt;
  };
  if(!p.parseHeader())
    return failed();

  std::vector<int32_t> info;
  if(!p.get("info", info, 16))
    return failed();
  const int32_t lod = info[1], V = info[2], F = info[3], J = info[4], P = info[5];
  const int32_t n_id = info[6], n_ex = info[7], H = info[8], NF = info[9], K = info[10];
  const bool has_reg = info[11] != 0;
  const int32_t L = info[12];
  if(J != kNumJoints || P != kNumModelParams || n_id != kNumIdentity
     || (n_ex != 0 && n_ex != kNumExpression) || NF != (kNumJoints - 2) * 6 || H <= 0
     || V <= 0 || F <= 0 || K <= 0 || K > 16 || L < 0)
  {
    p.fail("unexpected MHR dimensions in 'info'");
    return failed();
  }

  Model m;
  m.m_lod = lod;
  m.m_hidden = H;
  m.m_features = NF;
  if(!p.get("faces", m.m_faces, int64_t(F) * 3) || !p.get("joint_parent", m.m_parent, J)
     || !p.get("joint_offset", m.m_offset, J * 3) || !p.get("joint_prerot", m.m_prerot, J * 4)
     || !p.get("joint_invbind", m.m_invbind, J * 8)
     || !p.get("pt_rowptr", m.m_pt_rowptr, J * 7 + 1) || !p.get("pt_col", m.m_pt_col)
     || !p.get("pt_val", m.m_pt_val))
    return failed();
  if(!inRange(m.m_faces, 0, V))
    return p.fail("face index out of range"), failed();
  for(int j = 0; j < J; j++)
    if(m.m_parent[j] >= j || m.m_parent[j] < -1 || (j > 0 && m.m_parent[j] < 0))
      return p.fail("joint hierarchy is not topologically sorted"), failed();
  if(!validCsr(m.m_pt_rowptr, m.m_pt_col, m.m_pt_val.size(), P))
    return p.fail("bad parameter transform"), failed();
  {
    std::vector<int32_t> blocks;
    if(!p.get("corr_w1_groups", blocks) || blocks.size() % 3 != 0
       || !p.get("corr_w1_cols", m.m_w1_cols) || !p.get("corr_w1_data", m.m_w1_data))
      return p.fail("bad corrective layer 1"), failed();
    if(!inRange(m.m_w1_cols, 0, NF))
      return p.fail("corrective layer 1 column out of range"), failed();
    uint64_t cols = 0, data = 0;
    for(size_t i = 0; i < blocks.size(); i += 3)
    {
      Model::DenseBlock b;
      b.first = blocks[i];
      b.count = blocks[i + 1];
      b.cols = blocks[i + 2];
      if(b.first < 0 || b.count <= 0 || b.cols < 0
         || int64_t(b.first) + int64_t(b.count) > int64_t(H) || cols > UINT32_MAX)
        return p.fail("corrective layer 1 block out of range"), failed();
      b.col_offset = uint32_t(cols);
      b.data_offset = data;
      cols += uint64_t(b.cols);
      data += uint64_t(b.cols) * b.count;
      m.m_w1_blocks.push_back(b);
    }
    if(cols != m.m_w1_cols.size() || data != m.m_w1_data.size())
      return p.fail("corrective layer 1 sizes do not match its data"), failed();
  }

  if(!loadVertexSet(p, "mesh", V, K, H, n_ex != 0, m.mesh))
    return failed();

  if(has_reg && !p.get("kp_joint_reg", m.m_kp_reg, kNumKeypoints * J))
    return failed();
  if(L > 0)
  {
    if(!loadVertexSet(p, "lm", L, K, H, false, m.landmarks) || !p.get("lm_kp", m.m_lm_kp)
       || !p.get("lm_src", m.m_lm_src, int64_t(m.m_lm_kp.size()))
       || !p.get("lm_w", m.m_lm_w, int64_t(m.m_lm_kp.size())))
      return failed();
    if(int64_t(L) + J > INT32_MAX || !inRange(m.m_lm_kp, 0, kNumKeypoints)
       || !inRange(m.m_lm_src, 0, L + J))
      return p.fail("landmark readout index out of range"), failed();
  }

  std::vector<uint8_t> names;
  if(p.has("joint_names") && p.get("joint_names", names))
    m.m_joint_names = splitNames(names);
  if(p.has("param_names") && p.get("param_names", names))
    m.m_param_names = splitNames(names);
  return m;
}

std::optional<Model> Model::loadFile(const std::string& path, std::string* err)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if(!in)
  {
    if(err)
      *err = "cannot open " + path;
    return std::nullopt;
  }
  std::vector<char> buf(size_t(in.tellg()));
  in.seekg(0);
  if(!in.read(buf.data(), std::streamsize(buf.size())))
  {
    if(err)
      *err = "cannot read " + path;
    return std::nullopt;
  }
  return load(buf, err);
}

std::string_view Model::jointName(int j) const noexcept
{
  return j >= 0 && size_t(j) < m_joint_names.size() ? std::string_view(m_joint_names[j])
                                                     : std::string_view{};
}

std::string_view Model::paramName(int p) const noexcept
{
  return p >= 0 && size_t(p) < m_param_names.size() ? std::string_view(m_param_names[p])
                                                    : std::string_view{};
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------
std::array<bool, kNumModelParams> periodicParams(const Model& m) noexcept
{
  std::array<bool, kNumModelParams> rot{}, other{};
  const int rows = static_cast<int>(m.m_pt_rowptr.size()) - 1;
  for(int r = 0; r < rows; r++)
  {
    const int channel = r % 7; // t xyz, r xyz, log2 scale
    for(int k = m.m_pt_rowptr[r]; k < m.m_pt_rowptr[r + 1]; k++)
    {
      const int p = m.m_pt_col[k];
      if(p < 0 || p >= kNumModelParams)
        continue;
      const bool unit = std::abs(std::abs(m.m_pt_val[k]) - 1.f) < 1e-6f;
      (channel >= 3 && channel <= 5 && unit ? rot : other)[p] = true;
    }
  }
  std::array<bool, kNumModelParams> out{};
  for(int p = 0; p < kNumModelParams; p++)
    out[p] = rot[p] && !other[p];
  return out;
}

void Workspace::prepare(const Model& m)
{
  const int J = kNumJoints;
  global.assign(size_t(J) * 8, 0.0);
  jp.assign(size_t(J) * 7, 0.f);
  skin.assign(size_t(J) * 16, 0.f);
  features.assign(size_t(m.m_features), 0.f);
  hidden.assign(size_t(m.m_hidden), 0.f);
  rest_mesh.assign(size_t(m.mesh.count) * 3, 0.f);
  posed_mesh.assign(size_t(m.mesh.count) * 3, 0.f);
  rest_lm.assign(size_t(m.landmarks.count) * 3, 0.f);
  posed_lm.assign(size_t(m.landmarks.count) * 3, 0.f);
  out_lm.assign(size_t(m.landmarks.count) * 3, 0.f);
  joints_vision.assign(size_t(J) * 3, 0.f);
  mesh_rest_valid = lm_rest_valid = false;
  sized_for = &m;
  sized_vertices = m.mesh.count;
  sized_landmarks = m.landmarks.count;
}

bool Workspace::fits(const Model& m) const noexcept
{
  // Pointer + sizes: evaluate() refuses a Workspace prepared for another
  // Model (or for this one before it was moved). A different file loaded into
  // the same Model object is only caught when its sizes differ, so callers
  // must prepare() again after every (re)load -- that also drops the cached
  // rest shapes.
  return sized_for == &m && sized_vertices == m.mesh.count
         && sized_landmarks == m.landmarks.count
         && hidden.size() == size_t(m.m_hidden);
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------
bool evaluate(
    const Model& m, std::span<const float> pose, std::span<const float> identity,
    std::span<const float> expression, Workspace& ws, const Outputs& out) noexcept
{
  const int J = kNumJoints;
  if(pose.size() != size_t(kNumModelParams) || identity.size() != size_t(kNumIdentity)
     || (!expression.empty() && expression.size() != size_t(kNumExpression)))
    return false;
  if(!ws.fits(m))
    return false;
  if(!out.vertices.empty() && out.vertices.size() != size_t(m.mesh.count) * 3)
    return false;
  if(!out.joints.empty() && out.joints.size() != size_t(J) * 3)
    return false;
  const bool want_kp = !out.keypoints.empty();
  const bool exact_kp = want_kp && out.keypoint_method == KeypointMethod::Exact;
  if(want_kp
     && (out.keypoints.size() != size_t(kNumKeypoints) * 3
         || (exact_kp && !m.hasExactKeypoints())
         || (!exact_kp && !m.hasJointRegressor())))
    return false;

  // 1. Joint parameters: jp = T @ pose (sparse CSR, float32 like the torch
  //    einsum).
  for(int r = 0; r < J * 7; r++)
  {
    float acc = 0.f;
    for(int i = m.m_pt_rowptr[r]; i < m.m_pt_rowptr[r + 1]; i++)
      acc += m.m_pt_val[i] * pose[m.m_pt_col[i]];
    ws.jp[r] = acc;
  }

  // 2. Forward kinematics in float64 (pymomentum's use_double_precision).
  for(int j = 0; j < J; j++)
  {
    const float* p = ws.jp.data() + j * 7;
    const float* pr = m.m_prerot.data() + j * 4;
    const Quat prerot{pr[0], pr[1], pr[2], pr[3]};
    const Quat e = eulerXYZ(p[3], p[4], p[5]);
    const Quat lq = qnormalize(qmul(prerot, e));
    if(j >= 2)
    {
      // Corrective-MLP input (mhr.py _pose_features_from_joint_params): the
      // first two columns of R(jp.r) = Rz Ry Rx (utils.batch6DFromXYZ, the
      // joint's own rotation without its pre-rotation) minus the identity's.
      // Built from the same quaternion, so no second set of sin/cos; the
      // diagonal terms come out as -2(..) with no cancellation.
      float* f = ws.features.data() + (j - 2) * 6;
      f[0] = float(-2.0 * (e.y * e.y + e.z * e.z));    // R00 - 1
      f[1] = float(2.0 * (e.x * e.y + e.w * e.z));     // R10
      f[2] = float(2.0 * (e.x * e.z - e.w * e.y));     // R20
      f[3] = float(2.0 * (e.x * e.y - e.w * e.z));     // R01
      f[4] = float(-2.0 * (e.x * e.x + e.z * e.z));    // R11 - 1
      f[5] = float(2.0 * (e.y * e.z + e.w * e.x));     // R21
    }
    const double lt[3]
        = {double(p[0]) + m.m_offset[j * 3], double(p[1]) + m.m_offset[j * 3 + 1],
           double(p[2]) + m.m_offset[j * 3 + 2]};
    const double ls = std::exp2(double(p[6]));

    double* g = ws.global.data() + j * 8;
    const int par = m.m_parent[j];
    if(par < 0)
    {
      g[0] = lt[0], g[1] = lt[1], g[2] = lt[2];
      g[3] = lq.x, g[4] = lq.y, g[5] = lq.z, g[6] = lq.w;
      g[7] = ls;
    }
    else
    {
      const double* pg = ws.global.data() + par * 8;
      const Quat pq{pg[3], pg[4], pg[5], pg[6]};
      double rt[3];
      qrotate(pq, lt, rt);
      g[0] = pg[0] + pg[7] * rt[0];
      g[1] = pg[1] + pg[7] * rt[1];
      g[2] = pg[2] + pg[7] * rt[2];
      const Quat q = qnormalize(qmul(pq, lq));
      g[3] = q.x, g[4] = q.y, g[5] = q.z, g[6] = q.w;
      g[7] = pg[7] * ls;
    }
  }

  // 3. Skinning transforms: (global * invbind), as a 3x4 [sR | t] with the
  //    vision flip diag(0.01, -0.01, -0.01) folded in; and vision joints.
  for(int j = 0; j < J; j++)
  {
    const double* g = ws.global.data() + j * 8;
    const float* ib = m.m_invbind.data() + j * 8;
    const Quat gq{g[3], g[4], g[5], g[6]};
    const double ibt[3] = {ib[0], ib[1], ib[2]};
    double rt[3];
    qrotate(gq, ibt, rt);
    const double t[3] = {g[0] + g[7] * rt[0], g[1] + g[7] * rt[1], g[2] + g[7] * rt[2]};
    const Quat q = qnormalize(qmul(gq, Quat{ib[3], ib[4], ib[5], ib[6]}));
    const double s = g[7] * ib[7];

    const double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    const double R[9]
        = {1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
           2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
           2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)};
    constexpr double flip[3] = {0.01, -0.01, -0.01};
    float* M = ws.skin.data() + j * 16; // column-major c0 | c1 | c2 | t
    for(int r = 0; r < 3; r++)
    {
      M[0 + r] = float(flip[r] * s * R[r * 3 + 0]);
      M[4 + r] = float(flip[r] * s * R[r * 3 + 1]);
      M[8 + r] = float(flip[r] * s * R[r * 3 + 2]);
      M[12 + r] = float(flip[r] * t[r]);
    }
    M[3] = M[7] = M[11] = M[15] = 0.f;
    ws.joints_vision[j * 3 + 0] = float(g[0] * 0.01);
    ws.joints_vision[j * 3 + 1] = float(g[1] * -0.01);
    ws.joints_vision[j * 3 + 2] = float(g[2] * -0.01);
  }

  const bool want_mesh = !out.vertices.empty();
  if(want_mesh || exact_kp)
  {
    // 4. Rest shapes (identity cached per workspace).
    float expr_full[kNumIdentity + kNumExpression]{};
    const bool has_expr = !expression.empty();
    if(has_expr)
      std::copy(expression.begin(), expression.end(), expr_full + kNumIdentity);
    const bool same_id
        = std::equal(identity.begin(), identity.end(), ws.cached_identity.begin())
          && (has_expr ? std::equal(
                  expression.begin(), expression.end(), ws.cached_expression.begin())
                       : std::all_of(
                           ws.cached_expression.begin(), ws.cached_expression.end(),
                           [](float x) { return x == 0.f; }));
    if(!same_id)
    {
      std::copy(identity.begin(), identity.end(), ws.cached_identity.begin());
      if(has_expr)
        std::copy(expression.begin(), expression.end(), ws.cached_expression.begin());
      else
        ws.cached_expression.fill(0.f);
      ws.mesh_rest_valid = ws.lm_rest_valid = false;
    }
    // Each rest shape is rebuilt lazily, only when its output is requested.
    if(want_mesh && !ws.mesh_rest_valid)
    {
      blendRest(m.mesh, identity.data(), has_expr ? expr_full : nullptr, ws.rest_mesh);
      ws.mesh_rest_valid = true;
    }
    if(exact_kp && !ws.lm_rest_valid)
    {
      blendRest(m.landmarks, identity.data(), nullptr, ws.rest_lm);
      ws.lm_rest_valid = true;
    }

    // 5. Corrective MLP (features were filled during forward kinematics).
    // Layer 1 + ReLU, one dense block at a time: h[block] = sum over the
    // block's columns of feature * row -- independent FMAs across the block's
    // units, and a zero feature (a joint at its rest angle) is skipped
    // outright. MHR's blocks are all 24 wide and take the SIMD kernel.
    std::fill(ws.hidden.begin(), ws.hidden.end(), 0.f);
    for(const auto& b : m.m_w1_blocks)
    {
      float* h = ws.hidden.data() + b.first;
      const int32_t* cols = m.m_w1_cols.data() + b.col_offset;
      const float* w = m.m_w1_data.data() + b.data_offset;
      if(b.count == 24)
        denseBlock24(h, cols, w, b.cols, ws.features.data());
      else
        denseBlockN(h, cols, w, b.cols, b.count, ws.features.data());
    }

    // 6. Layer 2 into the posed rest shape, then skin.
    if(want_mesh)
    {
      std::copy(ws.rest_mesh.begin(), ws.rest_mesh.end(), ws.posed_mesh.begin());
      applyBasis(m.mesh.correctives, ws.hidden.data(), ws.posed_mesh.data());
      skinVertices(m.mesh, ws.skin.data(), ws.posed_mesh.data(), out.vertices.data());
    }
    if(exact_kp)
    {
      std::copy(ws.rest_lm.begin(), ws.rest_lm.end(), ws.posed_lm.begin());
      applyBasis(m.landmarks.correctives, ws.hidden.data(), ws.posed_lm.data());
      skinVertices(m.landmarks, ws.skin.data(), ws.posed_lm.data(), ws.out_lm.data());
    }
  }

  if(!out.joints.empty())
    std::copy(ws.joints_vision.begin(), ws.joints_vision.end(), out.joints.begin());

  if(want_kp)
  {
    float* kp = out.keypoints.data();
    std::fill_n(kp, kNumKeypoints * 3, 0.f);
    if(exact_kp)
    {
      const int L = m.landmarks.count;
      for(size_t i = 0; i < m.m_lm_kp.size(); i++)
      {
        const int src = m.m_lm_src[i];
        const float* s = src < L ? ws.out_lm.data() + src * 3
                                 : ws.joints_vision.data() + (src - L) * 3;
        float* o = kp + m.m_lm_kp[i] * 3;
        const float w = m.m_lm_w[i];
        o[0] += w * s[0], o[1] += w * s[1], o[2] += w * s[2];
      }
    }
    else
    {
      for(int k = 0; k < kNumKeypoints; k++)
      {
        const float* W = m.m_kp_reg.data() + k * J;
        float x = 0, y = 0, z = 0;
        for(int j = 0; j < J; j++)
        {
          x += W[j] * ws.joints_vision[j * 3];
          y += W[j] * ws.joints_vision[j * 3 + 1];
          z += W[j] * ws.joints_vision[j * 3 + 2];
        }
        kp[k * 3] = x, kp[k * 3 + 1] = y, kp[k * 3 + 2] = z;
      }
    }
  }
  return true;
}
}
