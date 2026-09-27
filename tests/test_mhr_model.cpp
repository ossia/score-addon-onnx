// Onnx::Mhr (Onnx/helpers/MhrModel.hpp): the native MHR body-model forward
// pass against Meta's official implementation (mhr + pymomentum).
//
// Data (SKIPs when absent), rooted at $ONNX_TEST_MHR:
//   bin/mhr_lod<L>.mhrbin   tools/mhr_export.py --assets assets --lod L
//                           --kp-regressor .../mhr_j127_to_kp70.npy
//                           --landmarks .../mhr_landmarks70.npz
//   ref/mhr_ref_lod<L>.bin  official forward on 8 cases: T-pose, shape only,
//                           InstantHMR on a real photo, random and extreme poses
// Tolerance: 1e-4 m on every vertex (float32).
#include <tests/TestPaths.hpp>

#include <Onnx/helpers/MeshOps.hpp>
#include <Onnx/helpers/MhrModel.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace Mhr = Onnx::Mhr;

namespace
{
std::vector<char> slurp(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

// mhr_ref.py layout: "MHRREF1\0", u32 n, V, J, K; per case pose[204],
// identity[45], verts[V*3], joints[J*3], kp_reg[K*3], kp_exact[K*3].
struct RefCase
{
  const float* pose;
  const float* identity;
  const float* verts;
  const float* joints;
  const float* kp_reg; // null when K == 0
  const float* kp_exact;
};

struct RefFile
{
  std::vector<char> raw;
  uint32_t n = 0, V = 0, J = 0, K = 0;
  bool load(const std::string& path)
  {
    raw = slurp(path);
    if(raw.size() < 24 || std::memcmp(raw.data(), "MHRREF1", 8) != 0)
      return false;
    uint32_t h[4];
    std::memcpy(h, raw.data() + 8, sizeof(h));
    n = h[0], V = h[1], J = h[2], K = h[3];
    return raw.size() == 24 + size_t(n) * record() * 4;
  }
  size_t record() const { return 204 + 45 + size_t(V) * 3 + size_t(J) * 3 + size_t(K) * 6; }
  RefCase at(uint32_t i) const
  {
    // The float payload starts at byte 24: aligned for float in practice
    // (std::vector storage), and the file is little-endian like the host.
    const float* r = reinterpret_cast<const float*>(raw.data() + 24) + i * record();
    RefCase c;
    c.pose = r;
    c.identity = r + 204;
    c.verts = c.identity + 45;
    c.joints = c.verts + size_t(V) * 3;
    c.kp_reg = K ? c.joints + size_t(J) * 3 : nullptr;
    c.kp_exact = K ? c.kp_reg + size_t(K) * 3 : nullptr;
    return c;
  }
};

double maxAbsDiff(std::span<const float> a, const float* b)
{
  double m = 0;
  for(size_t i = 0; i < a.size(); i++)
    m = std::max(m, double(std::abs(a[i] - b[i])));
  return m;
}
}

TEST_CASE("MHR: native forward matches the official one", "[mhr][model]")
{
  int tested = 0;
  for(int lod = 0; lod <= 6; lod++)
  {
    const auto bin = TestPaths::mhrBin(lod);
    const auto ref = TestPaths::mhr() + "/ref/mhr_ref_lod" + std::to_string(lod) + ".bin";
    if(!std::filesystem::exists(bin) || !std::filesystem::exists(ref))
      continue;
    DYNAMIC_SECTION("LOD " << lod)
    {
      std::string err;
      auto model = Mhr::Model::loadFile(bin, &err);
      INFO(err);
      REQUIRE(model);
      RefFile r;
      REQUIRE(r.load(ref));
      REQUIRE(int(r.V) == model->numVertices());
      REQUIRE(int(r.J) == Mhr::kNumJoints);
      CHECK(model->lod() == lod);

      Mhr::Workspace ws(*model);
      std::vector<float> verts(r.V * 3), joints(r.J * 3), kp(Mhr::kNumKeypoints * 3);
      for(uint32_t c = 0; c < r.n; c++)
      {
        INFO("case " << c);
        const auto rc = r.at(c);
        const std::span<const float> pose(rc.pose, 204), id(rc.identity, 45);

        // Keypoints first: exercises the lazily built landmark rest shape
        // before the mesh's own for a fresh identity.
        if(rc.kp_exact && model->hasExactKeypoints())
        {
          REQUIRE(Mhr::evaluate(*model, pose, id, {}, ws, {{}, {}, kp, Mhr::KeypointMethod::Exact}));
          CHECK(maxAbsDiff(kp, rc.kp_exact) < 1e-4);
        }
        REQUIRE(Mhr::evaluate(*model, pose, id, {}, ws, {verts, joints, {}, {}}));
        const double ev = maxAbsDiff(verts, rc.verts);
        INFO("max vertex error " << ev << " m");
        CHECK(ev < 1e-4);
        CHECK(maxAbsDiff(joints, rc.joints) < 1e-5);
        if(rc.kp_reg && model->hasJointRegressor())
        {
          REQUIRE(Mhr::evaluate(
              *model, pose, id, {}, ws, {{}, {}, kp, Mhr::KeypointMethod::JointRegressor}));
          CHECK(maxAbsDiff(kp, rc.kp_reg) < 1e-4);
        }
      }

      // Closed, outward-wound mesh: positive signed volume in the vision
      // frame for the T-pose (~0.08 m^3 for the mean identity).
      const std::vector<float> zeros(204, 0.f), zid(45, 0.f);
      REQUIRE(Mhr::evaluate(*model, zeros, zid, {}, ws, {verts, {}, {}, {}}));
      const auto F = model->faces();
      double vol = 0;
      for(size_t f = 0; f < F.size(); f += 3)
      {
        const float* a = &verts[F[f] * 3];
        const float* b = &verts[F[f + 1] * 3];
        const float* d = &verts[F[f + 2] * 3];
        vol += a[0] * (double(b[1]) * d[2] - double(b[2]) * d[1])
               - a[1] * (double(b[0]) * d[2] - double(b[2]) * d[0])
               + a[2] * (double(b[0]) * d[1] - double(b[1]) * d[0]);
      }
      vol /= 6;
      CHECK(vol > 0.06);
      CHECK(vol < 0.1);
      tested++;
    }
  }
  if(tested == 0)
    SKIP("needs $ONNX_TEST_MHR with bin/ and ref/ (see tools/mhr_export.py)");
}

TEST_CASE("MHR: a file that is not an .mhrbin is refused", "[mhr]")
{
  std::string err;
  const std::vector<char> garbage(4096, 'x');
  CHECK_FALSE(Mhr::Model::load(garbage, &err));
  CHECK_FALSE(err.empty());
  CHECK_FALSE(Mhr::Model::load({}, &err));
}

TEST_CASE("MHR: truncated files and misuse are refused", "[mhr][model]")
{
  std::string err;
  const auto bin = TestPaths::mhrBin(6);
  if(!std::filesystem::exists(bin))
    SKIP("needs $ONNX_TEST_MHR/bin/mhr_lod6.mhrbin");
  const auto raw = slurp(bin);
  REQUIRE(Mhr::Model::load(raw, &err));
  for(size_t cut : {size_t(24), size_t(200), raw.size() / 3, raw.size() - 1})
    CHECK_FALSE(Mhr::Model::load(std::span(raw).first(cut), &err));

  auto model = Mhr::Model::load(raw, &err);
  REQUIRE(model);
  std::vector<float> pose(204, 0.f), id(45, 0.f), verts(model->numVertices() * 3);
  Mhr::Workspace unprepared;
  CHECK_FALSE(Mhr::evaluate(*model, pose, id, {}, unprepared, {verts, {}, {}, {}}));
  Mhr::Workspace ws(*model);
  CHECK_FALSE(Mhr::evaluate(*model, std::span(pose).first(200), id, {}, ws, {verts, {}, {}, {}}));
  CHECK_FALSE(Mhr::evaluate(*model, pose, id, {}, ws, {std::span(verts).first(6), {}, {}, {}}));
  CHECK(Mhr::evaluate(*model, pose, id, {}, ws, {verts, {}, {}, {}}));
  if(!model->hasExpression())
  {
    // No expression basis in the file: zero coefficients are still accepted.
    std::vector<float> expr(72, 0.f);
    CHECK(Mhr::evaluate(*model, pose, id, expr, ws, {verts, {}, {}, {}}));
  }
  CHECK(model->jointName(1) == "root");
  CHECK(model->paramName(3) == "root_rx");
}

namespace
{
// Mutable view of one int32 array of an .mhrbin (see MhrModel.cpp's header
// layout: 24-byte header, 64-byte descriptors: name[24], dtype, ndim,
// dims[4], u64 offset, u64 bytes).
int32_t* findI32(std::vector<char>& raw, const char* name, size_t* count)
{
  uint32_t n;
  std::memcpy(&n, raw.data() + 12, 4);
  for(uint32_t i = 0; i < n; i++)
  {
    const char* d = raw.data() + 24 + 64 * size_t(i);
    if(std::strncmp(d, name, 24) != 0)
      continue;
    uint64_t off, bytes;
    std::memcpy(&off, d + 48, 8);
    std::memcpy(&bytes, d + 56, 8);
    *count = bytes / 4;
    return reinterpret_cast<int32_t*>(raw.data() + off);
  }
  return nullptr;
}
}

// A hostile file whose group range wraps int32 (first + count > INT32_MAX)
// with otherwise consistent sizes must be refused, else first + count wraps
// negative and the forward pass writes / reads far out of bounds.
TEST_CASE("MHR: int32-wrapping group ranges are refused", "[mhr][model]")
{
  const auto bin = TestPaths::mhrBin(6);
  if(!std::filesystem::exists(bin))
    SKIP("needs $ONNX_TEST_MHR/bin/mhr_lod6.mhrbin");
  const auto raw = slurp(bin);
  REQUIRE(Mhr::Model::load(raw));
  for(const char* arr : {"corr_w1_groups", "mesh.id_groups", "mesh.co_groups"})
  {
    DYNAMIC_SECTION(arr)
    {
      auto bad = raw;
      size_t n = 0;
      int32_t* g = findI32(bad, arr, &n);
      REQUIRE(g);
      REQUIRE(n % 3 == 0);
      // A group with count >= 2, sizes (count, support/cols) untouched.
      size_t i = 0;
      while(i < n && g[i + 1] < 2)
        i += 3;
      REQUIRE(i < n);
      g[i] = INT32_MAX - g[i + 1] + 2; // first + count == INT32_MAX + 2
      std::string err;
      CHECK_FALSE(Mhr::Model::load(bad, &err));
      CHECK(err.find("out of") != std::string::npos);
      // Values where the sizes then mismatch or overflow.
      g[i] = 0x7FFFFFF0;
      g[i + 1] = 0x20;
      CHECK_FALSE(Mhr::Model::load(bad, &err));
    }
  }
}

// Draw Mesh with degenerate vertices (a NaN / inf / astronomically far corner,
// e.g. from a diverged pose): those faces are skipped, never cast to int
// (undefined behaviour), and the sane face still draws.
TEST_CASE("MeshOps: the rasterizer skips non-finite and huge vertices", "[mhr]")
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  // Face 0: a sane triangle 2 m in front of the camera, facing it (CCW from
  // the camera = outward normal toward it). Faces 1-3 share a bad corner.
  const std::vector<float> v{
      -0.5f, -0.5f, 2.f,  0.5f, -0.5f, 2.f,  0.f, 0.5f, 2.f, // 0-2
      nan, 0.f, 2.f,      inf, 0.f, 2.f,     1e30f, 1e30f, 2.f};
  const std::vector<int32_t> f{0, 2, 1, 0, 2, 3, 0, 2, 4, 0, 2, 5};
  Onnx::MeshOps::Raster r;
  r.begin(64, 48);
  const uint8_t rgb[3] = {255, 0, 0};
  r.draw(v, f, 50.f, 32.f, 24.f, rgb);
  size_t drawn = 0;
  for(float d : r.depth)
    drawn += d > 0.f;
  CHECK(drawn > 50);
  CHECK(drawn < 64 * 48 / 2);
  std::vector<uint8_t> img(64 * 48 * 4, 0);
  r.composite(img.data(), 64 * 4, 1.f);
}
