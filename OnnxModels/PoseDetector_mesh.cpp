// Body mesh of the human-mesh-recovery models (InstantHMR): the native MHR
// forward pass (Onnx/helpers/MhrModel.hpp) run on each emitted person's
// smoothed body parameters, per track id, for the mesh Data Formats, Draw Mesh
// and Mesh Keypoints.
//
// Where the mesh sits in camera space. MHR evaluates rig-local: origin = MHR
// joint 0, the frame InstantHMR's cam_trans places in the camera (upstream
// renders verts + cam_trans). The node, though, publishes translation =
// cam_trans + pelvis (pelvis = mid-hips of the joints_3d head, rig-local), so
// `world` can be pelvis-relative. The decoder therefore also keeps the pelvis
// itself as DetectedPose::rig_offset, and both are smoothed (single path:
// smooth3D; tracks: the tracker's payload) as separate signals. The mesh
// origin is then translation - rig_offset: exactly cam_trans without
// smoothing, and with it the mesh and the keypoints (world + translation) ride
// the SAME smoothed translation, so they cannot drift apart under motion;
// only the slowly-varying pelvis offset is filtered on its own.
//
// Cost model: nothing here runs unless an output needs the mesh (meshNeeds).
// Each evaluated person owns a MeshSlot (an MHR workspace + its buffers),
// kept across frames by track id so the workspace's cached identity blend
// (45 shape coefficients -> rest mesh) is reused; slots grow lazily up to the
// number of simultaneous people, then nothing allocates. The per-vertex loops
// are in the -O3 helper lib (MhrModel.cpp, MeshOps.cpp).
#include "PoseDetector_internal.hpp"

namespace OnnxModels
{

void PoseDetector::loadBodyModel()
{
  if(inputs.body_model.file.filename.empty())
  {
    if(m_body_models.model() || m_body_models.loading())
    {
      m_body_models.release(worker);
      bodyModelInstalled();
    }
    return;
  }
  if(!m_body_models.requested().is(inputs.body_model))
    m_body_models.request(worker, ModelFile::of(inputs.body_model));
}

void PoseDetector::bodyModelInstalled()
{
  // Every workspace is sized for (and points at) the old model: drop them
  // with it. They are rebuilt lazily for the new one.
  m_mesh_slots.clear();
  m_inst_mesh.clear();
  m_single_mesh = -1;
  m_mhr = m_body_models.model();
  if(m_mhr)
    m_mesh_slots.reserve(16); // Max Instances' ceiling: slots never move later
  setHmrParamKinds(); // the smoothing kinds of this rig's parameters
}

PoseDetector::MeshNeeds PoseDetector::meshNeeds() const noexcept
{
  MeshNeeds n;
  if(!m_mhr)
    return n;
  const auto f = inputs.data_format.value;
  n.verts = isMeshFormat(f) || inputs.draw_mesh.value;
  n.keypoints = inputs.mesh_keypoints.value
                && (m_mhr->hasExactKeypoints() || m_mhr->hasJointRegressor());
  return n;
}

int PoseDetector::claimMeshSlot(int track_id)
{
  // 1. The slot this id used last (its identity blend is cached there).
  if(track_id >= 0)
    for(size_t s = 0; s < m_mesh_slots.size(); ++s)
      if(m_mesh_slots[s].track_id == track_id && m_mesh_slots[s].claimed != m_mesh_frame)
      {
        m_mesh_slots[s].claimed = m_mesh_frame;
        return static_cast<int>(s);
      }
  // 2. The least recently used slot not taken this frame. Its previous owner
  //    is not on screen (it would have claimed it in pass 1 of this frame);
  //    reassigning costs one identity blend, done lazily by evaluate().
  int best = -1;
  for(size_t s = 0; s < m_mesh_slots.size(); ++s)
  {
    const auto& ms = m_mesh_slots[s];
    if(ms.claimed == m_mesh_frame)
      continue;
    if(best < 0 || ms.claimed < m_mesh_slots[best].claimed)
      best = static_cast<int>(s);
  }
  // 3. A new slot, up to one per simultaneous person (warm-up only).
  if(best < 0)
  {
    if(m_mesh_slots.size() >= 16)
      return -1;
    m_mesh_slots.emplace_back();
    best = static_cast<int>(m_mesh_slots.size()) - 1;
  }
  auto& ms = m_mesh_slots[best];
  ms.track_id = track_id;
  ms.claimed = m_mesh_frame;
  return best;
}

// A pose the body model can evaluate: MHR pose + identity parameters and the
// camera placement (translation, rig_offset). Every other model: none.
static bool hasBody(const DetectedPose& p) noexcept
{
  return p.body_params.size()
             >= size_t(Onnx::Mhr::kNumModelParams + Onnx::Mhr::kNumIdentity)
         && p.translation.size() >= 3 && p.rig_offset.size() >= 3;
}

bool PoseDetector::evaluateMesh(DetectedPose& pose, int slot, const MeshNeeds& needs)
{
  auto& ms = m_mesh_slots[slot];
  ms.has_verts = false;
  const auto& model = *m_mhr;
  constexpr size_t n_pose = Onnx::Mhr::kNumModelParams;
  constexpr size_t n_id = Onnx::Mhr::kNumIdentity;
  if(!hasBody(pose))
    return false;
  if(!ms.ws.fits(model))
    ms.ws.prepare(model); // first use of this slot with this model only

  const bool want_kp = needs.keypoints;
  const size_t nv = static_cast<size_t>(model.numVertices()) * 3;
  if(needs.verts)
    ms.verts.resize(nv); // no-op once sized
  if(want_kp)
    ms.kp.resize(Onnx::Mhr::kNumKeypoints * 3);
  const auto method = model.hasExactKeypoints()
                          ? Onnx::Mhr::KeypointMethod::Exact
                          : Onnx::Mhr::KeypointMethod::JointRegressor;
  const std::span<const float> params(pose.body_params);
  if(!Onnx::Mhr::evaluate(
         model, params.first(n_pose), params.subspan(n_pose, n_id), {}, ms.ws,
         {needs.verts ? std::span<float>(ms.verts) : std::span<float>{}, {},
          want_kp ? std::span<float>(ms.kp) : std::span<float>{}, method}))
    return false;

  // Rig origin in camera space (see the top of this file).
  const float origin[3]
      = {pose.translation[0] - pose.rig_offset[0],
         pose.translation[1] - pose.rig_offset[1],
         pose.translation[2] - pose.rig_offset[2]};
  if(needs.verts)
  {
    Onnx::MeshOps::translate(ms.verts, origin, ms.verts);
    ms.has_verts = true;
  }

  // Mesh Keypoints: the mesh's own MHR70 joints replace the keypoint heads'.
  // world keeps its contract (world + translation = camera space), so it is
  // kp - rig_offset: relative to the same (joints_3d) pelvis anchor as before,
  // just no longer exactly mid-hips of the new joints. The 2D keypoints are
  // their projection through InstantHMR's camera (instantHmrFocal: the frame
  // diagonal, x1.05 for the angular CLIFF form; principal point at the
  // centre), screen z = the world depth as decoded.
  // Only the native 70-joint layout is rewritten (before any remap).
  if(want_kp && pose.world.size() == Onnx::Mhr::kNumKeypoints)
  {
    const float* kp = ms.kp.data();
    for(int k = 0; k < Onnx::Mhr::kNumKeypoints; ++k)
    {
      pose.world[k].x = kp[3 * k + 0] - pose.rig_offset[0];
      pose.world[k].y = kp[3 * k + 1] - pose.rig_offset[1];
      pose.world[k].z = kp[3 * k + 2] - pose.rig_offset[2];
    }
    const int W = inputs.image.texture.width, H = inputs.image.texture.height;
    if(pose.keypoints.size() == Onnx::Mhr::kNumKeypoints && W > 0 && H > 0)
    {
      const float f = instantHmrFocal(W, H, m_hmr_cliff_focal);
      for(int k = 0; k < Onnx::Mhr::kNumKeypoints; ++k)
      {
        const float X = kp[3 * k + 0] + origin[0];
        const float Y = kp[3 * k + 1] + origin[1];
        const float Z = kp[3 * k + 2] + origin[2];
        if(!(Z > 0.05f))
          continue; // behind / at the camera: keep the model's 2D joint
        auto& p = pose.keypoints[k];
        p.x = (f * X / Z + 0.5f * W) / W;
        p.y = (f * Y / Z + 0.5f * H) / H;
        p.z = pose.world[k].z;
      }
    }
  }
  return true;
}

void PoseDetector::evaluateSingleMesh(DetectedPose& pose)
{
  m_single_mesh = -1;
  const MeshNeeds needs = meshNeeds();
  if(!needs.any() || !hasBody(pose))
    return;
  ONNX_PROF_SCOPE(Mesh);
  ++m_mesh_frame;
  if(m_mesh_slots.empty())
    m_mesh_slots.emplace_back();
  // One person: always slot 0, whatever its (absent) id.
  m_mesh_slots[0].claimed = m_mesh_frame;
  m_mesh_slots[0].track_id = pose.track_id;
  if(evaluateMesh(pose, 0, needs))
    m_single_mesh = 0;
}

void PoseDetector::evaluateInstanceMeshes()
{
  // One slot index per instance, always sized (appendGeometry / drawMeshes
  // index it); assign() only allocates when the instance count grows.
  m_inst_mesh.assign(m_instances.size(), -1);
  m_single_mesh = -1; // the multi path owns the slots now
  const MeshNeeds needs = meshNeeds();
  if(!needs.any())
    return;
  ONNX_PROF_SCOPE(Mesh);
  ++m_mesh_frame;
  const int max_inst
      = std::clamp(static_cast<int>(inputs.max_instances.value), 1, 16);
  const int n = std::min(static_cast<int>(m_instances.size()), max_inst);
  // Known ids first, so a person on screen keeps its slot (and its cached
  // identity) even when a newcomer is listed before it.
  for(int pass = 0; pass < 2; ++pass)
    for(int i = 0; i < n; ++i)
    {
      const int id = m_instances[i].track_id;
      if(m_inst_mesh[i] != -1) // done in pass 0 (-2: tried, no mesh)
        continue;
      if(!hasBody(m_instances[i]))
      {
        m_inst_mesh[i] = -2; // not a body-model pose: no slot taken
        continue;
      }
      bool known = false;
      if(pass == 0)
      {
        if(id < 0)
          continue;
        for(const auto& ms : m_mesh_slots)
          if(ms.track_id == id)
          {
            known = true;
            break;
          }
        if(!known)
          continue;
      }
      const int slot = claimMeshSlot(id);
      if(slot < 0)
        continue;
      m_inst_mesh[i] = evaluateMesh(m_instances[i], slot, needs) ? slot : -1;
      if(m_inst_mesh[i] < 0)
        m_inst_mesh[i] = -2; // tried: do not retry in pass 2
    }
  for(auto& s : m_inst_mesh)
    if(s < -1)
      s = -1;
}

} // namespace OnnxModels
