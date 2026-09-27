#pragma once
// Native forward pass of Meta's MHR ("Momentum Human Rig") parametric body
// model: InstantHMR's per-person `mhr_params [204]` + `shape_params [45]` ->
// a dense skinned body mesh, the 127 skeleton joints and the 70 MHR70
// keypoints. Qt / ossia / onnxruntime / torch free.
//
// The official model is a torch module on top of pymomentum plus a small
// sparse MLP for pose correctives, but every stage is either a sparse linear
// map, a 127-joint kinematic chain, or linear blend skinning. The assets are converted once, offline, by tools/mhr_export.py
// into a single `.mhrbin` file (format documented there and in Model::load);
// the node takes that file from a file port -- Meta's assets are never bundled.
//
// The pass, exactly as mhr/mhr.py + pymomentum/torch/character.py compute it
// (all rig quantities in centimetres, Y up, the T-pose template facing +Z):
//
//   1. rest   = base + sum_i identity[i] * S_i (+ sum_e expression[e] * E_e)
//               S: 45 identity shapes = 20 body + 20 head + 5 hand components,
//               each group touching only its own vertex support.
//   2. jp     = T @ pose                    (sparse 889 x 204 matrix, 368 nnz)
//               per joint 7 values: translation (3), Euler XYZ angles (3,
//               R = Rz*Ry*Rx), log2 scale (1).
//   3. local  = { t = jp.t + offset, q = prerot * q_xyz(jp.r), s = 2^jp.s }
//      global = global[parent] * local      (similarity transforms, float64,
//               t = tp + sp * Rp t,  q = qp q,  s = sp s)
//   4. correctives: feat = [R(jp.r) column 0 - e0, column 1 - e1] for joints
//      2..126 (750 values); h = relu(W1 feat) (3000 hidden units, 53k nnz);
//      rest += W2 h. W1's hidden units come in 125 blocks of 24, one per joint,
//      and all 24 units of a block displace exactly the same vertices, so W2 is
//      stored as 125 dense [24][support][3] slabs. ReLU zeros skip whole units;
//      the zero pose activates none (correctives vanish in the T-pose).
//   5. verts  = sum_k w_k (global_k * invbind_k)(rest)   (LBS, <= 4 influences)
//
// Output frame: InstantHMR's "vision" frame -- metres, X right, Y down, Z away
// from the camera, rig-local; i.e. (x, -y, -z) / 100 of the rig. The caller
// adds the person's `cam_trans` to get camera space. The rig origin is NOT the
// pelvis: it is joint 0 ("body_world", fixed at the origin); the pelvis (joint
// 1, "root", where mhr_params 0:6 apply) always sits at
// (0, -0.924, 0) + (0.1, -0.1, -0.1) * mhr_params[0:3] metres, and in the
// T-pose the feet touch y = 0. The unit/axis flip is folded into the skinning
// matrices, so it costs nothing.
//
// Real-time contract: Model::load allocates (once, off the audio/render
// thread); Workspace allocates in its constructor; evaluate() never allocates,
// never throws, and only touches the Model read-only -- one Model can be shared
// by any number of threads, each with its own Workspace (one per tracked
// person is ideal: the identity blend is cached per Workspace and only redone
// when the 45 coefficients change).
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Onnx::Mhr
{
inline constexpr int kNumModelParams = 204; // 3 root t, 3 root r, 130 angles, 68 scales
inline constexpr int kNumIdentity = 45;     // 20 body + 20 head + 5 hands
inline constexpr int kNumExpression = 72;   // face expression (InstantHMR: none)
inline constexpr int kNumJoints = 127;
inline constexpr int kNumKeypoints = 70; // MHR70 (skeleton.py of InstantHMR)

// A family of blend directions stored sparsely: components are grouped, each
// group shares one vertex support, and the group's slab is dense over it
// ([component][support][xyz]). Used for identity shapes, expression shapes
// and the corrective MLP's output layer alike.
struct SparseBasis
{
  struct Group
  {
    int32_t first = 0;        // first component (coefficient index)
    int32_t count = 0;        // components in the group
    int32_t support = 0;      // vertices touched
    uint32_t supp_offset = 0; // into `support_idx`
    uint64_t data_offset = 0; // into `data` (floats)
  };
  std::vector<Group> groups;
  std::vector<int32_t> support_idx;
  std::vector<float> data;
  int32_t max_support = 0;
  int32_t num_components = 0;

  bool empty() const noexcept { return groups.empty(); }
};

// Everything per vertex: the template, its skinning and its blend bases.
// The model has one for the mesh and optionally one for the 468 LOD-1
// vertices the exact MHR70 readout references.
struct VertexSet
{
  int32_t count = 0;
  int32_t influences = 0;      // K skin slots per vertex (padding: weight 0)
  std::vector<float> base;     // count*3, cm
  std::vector<int32_t> skin_idx; // count*K
  std::vector<float> skin_w;     // count*K
  SparseBasis identity, expression, correctives;
};

struct Model
{
  // Parse a .mhrbin (see tools/mhr_export.py). Validates every index against
  // its array, so a truncated or hostile file yields nullopt + *err, never an
  // out-of-bounds access later. The bytes are copied; they need not outlive
  // the call nor be aligned.
  static std::optional<Model> load(std::span<const char> bytes, std::string* err = nullptr);
  static std::optional<Model> loadFile(const std::string& path, std::string* err = nullptr);

  int lod() const noexcept { return m_lod; }
  int numVertices() const noexcept { return mesh.count; }
  int numFaces() const noexcept { return int(m_faces.size() / 3); }
  int numJoints() const noexcept { return kNumJoints; }
  // Triangles as vertex-index triples, counter-clockwise seen from outside in
  // the rig frame (the FBX winding). The vision frame flips two axes, a
  // rotation by 180 deg about X, so the winding stays counter-clockwise.
  std::span<const int32_t> faces() const noexcept { return m_faces; }
  std::span<const int32_t> jointParents() const noexcept { return m_parent; }
  std::string_view jointName(int j) const noexcept;
  std::string_view paramName(int p) const noexcept;

  bool hasExpression() const noexcept { return !mesh.expression.empty(); }
  bool hasJointRegressor() const noexcept { return !m_kp_reg.empty(); }
  bool hasExactKeypoints() const noexcept { return landmarks.count > 0; }

  // --- data (public for the evaluator and tests; treat as read-only) ---
  int m_lod = 0;
  std::vector<int32_t> m_faces;
  std::vector<int32_t> m_parent;    // J
  std::vector<float> m_offset;      // J*3
  std::vector<float> m_prerot;      // J*4 (x, y, z, w)
  std::vector<float> m_invbind;     // J*8 (t, q xyzw, s)
  std::vector<int32_t> m_pt_rowptr; // J*7+1
  std::vector<int32_t> m_pt_col;
  std::vector<float> m_pt_val;
  // Corrective layer 1: blocks of hidden units sharing their input columns,
  // each stored dense as [column][unit].
  struct DenseBlock
  {
    int32_t first = 0, count = 0, cols = 0;
    uint32_t col_offset = 0;
    uint64_t data_offset = 0;
  };
  std::vector<DenseBlock> m_w1_blocks;
  std::vector<int32_t> m_w1_cols;
  std::vector<float> m_w1_data;
  int32_t m_hidden = 0;   // H (3000)
  int32_t m_features = 0; // 750
  VertexSet mesh;
  VertexSet landmarks;
  std::vector<float> m_kp_reg; // 70*J, applied to vision-frame joints
  std::vector<int32_t> m_lm_kp, m_lm_src; // exact readout (COO)
  std::vector<float> m_lm_w;
  std::vector<std::string> m_joint_names, m_param_names;
};

// Per-instance scratch. Sized for one Model at construction; evaluate()
// refuses (returns false) a Model it was not sized for instead of allocating.
// Call prepare() again whenever the Model is reloaded or moved.
struct Workspace
{
  Workspace() = default;
  explicit Workspace(const Model& m) { prepare(m); }
  void prepare(const Model& m); // allocates
  bool fits(const Model& m) const noexcept;

  // Joint state after evaluate(): float64 global similarity transforms in
  // the rig frame (cm): t[3], q[4] (x, y, z, w), s.
  std::vector<double> global;   // J*8
  std::vector<float> jp;        // J*7 joint parameters
  std::vector<float> skin;      // J*16 column-major 3x4 + pad (vision frame folded in)
  std::vector<float> features;  // 750
  std::vector<float> hidden;    // H
  std::vector<float> rest_mesh, posed_mesh; // V*3
  std::vector<float> rest_lm, posed_lm, out_lm; // L*3
  std::vector<float> joints_vision; // J*3

  // Identity cache: rest_* hold base + identity (+ expression) for these
  // coefficients when the matching *_valid flag is set.
  std::array<float, kNumIdentity> cached_identity{};
  std::array<float, kNumExpression> cached_expression{};
  bool mesh_rest_valid = false, lm_rest_valid = false;
  const Model* sized_for = nullptr;
  int32_t sized_vertices = -1, sized_landmarks = -1;
};

enum class KeypointMethod : uint8_t
{
  // SAM 3D Body's own readout, K = W_j J + W_v V over 468 LOD-1 vertices:
  // the operator InstantHMR's training labels came from. Needs a file
  // exported with --landmarks.
  Exact,
  // InstantHMR's fitted (70 x 127) skeleton-only regressor (shape-blind).
  // Needs --kp-regressor.
  JointRegressor,
};

struct Outputs
{
  std::span<float> vertices;  // numVertices()*3, or empty to skip the mesh
  std::span<float> joints;    // 127*3, or empty
  std::span<float> keypoints; // 70*3, or empty
  KeypointMethod keypoint_method = KeypointMethod::Exact;
};

// Which model parameters are genuine angles, i.e. 2*pi-periodic in their
// effect: every coefficient the parameter has in the parameter transform is
// +-1 on a joint's Euler rotation channel. The others (translations, scales,
// and the "distributed" twists / spine bends that feed several joints with
// fractional weights, e.g. l_upleg_twist at 0.2..0.8) must not be wrapped:
// adding 2*pi to them visibly changes the mesh.
std::array<bool, kNumModelParams> periodicParams(const Model& model) noexcept;

// One person. `pose` = InstantHMR mhr_params (204), `identity` =
// shape_params (45), `expression` = 72 coefficients or empty (zero). All
// outputs in the vision frame (metres, Y down, Z forward, rig-local).
// Returns false (outputs untouched) on a size mismatch, a Workspace not
// prepared for this Model, or a requested keypoint method the file lacks.
bool evaluate(
    const Model& model, std::span<const float> pose, std::span<const float> identity,
    std::span<const float> expression, Workspace& ws, const Outputs& out) noexcept;
}
