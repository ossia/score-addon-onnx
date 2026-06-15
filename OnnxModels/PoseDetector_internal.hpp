#pragma once
// Internal (non-exported) shared helpers for the PoseDetector implementation,
// which is split across several .cpp files (PoseDetector.cpp + PoseDetector_*.cpp)
// to keep each translation unit readable. This header is included ONLY by those
// .cpp files — never by the avnd-registered PoseDetector.hpp.
//
// Everything here is `inline` (one definition merged across the split TUs); it
// is the small set of free helpers used by MORE THAN ONE of the split files.
// Helpers used by a single file stay `static` in that file.
#include "PoseDetector.hpp"

#include <ossia/math/safe_math.hpp>

#include <Onnx/helpers/CtxOverlay.hpp>

#include <Onnx/helpers/BlazeFace.hpp>
#include <Onnx/helpers/BlazePose.hpp>
#include <Onnx/helpers/Detection.hpp>
#include <Onnx/helpers/FaceMesh.hpp>
#include <Onnx/helpers/MediaPipeHands.hpp>
#include <Onnx/helpers/ModelRole.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Profile.hpp>
#include <Onnx/helpers/ROI.hpp>
#include <Onnx/helpers/RTMPose.hpp>
#include <Onnx/helpers/Yolo.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

namespace OnnxModels
{

// NaN/Inf-robust finiteness check. Uses ossia's bit-pattern variants because
// std::isnan/isinf are compiled away under -ffast-math / -ffinite-math-only.
inline bool finitef(float v) noexcept
{
  return !ossia::safe_isnan(v) && !ossia::safe_isinf(v);
}

// Fill pose.box (top-left normalized form) from the confident keypoints' bbox,
// unless it is already set (box-only detections set it from the detector).
inline void fillBoxFromKeypoints(DetectedPose& pose)
{
  if(pose.box.w > 0.f && pose.box.h > 0.f)
    return;
  // Box from the keypoints, but built from only CONFIDENT joints. wholebody-133
  // emits many low-confidence face/hand points whose SimCC peak hops every
  // frame; including them makes the box jump ~25% frame-to-frame. Prefer
  // conf>=0.5; fall back to a looser threshold only if that's too sparse.
  auto build = [&](float thr) -> int {
    float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
    int n = 0;
    for(const auto& k : pose.keypoints)
    {
      if(k.confidence < thr)
        continue;
      ++n;
      minx = std::min(minx, k.x); maxx = std::max(maxx, k.x);
      miny = std::min(miny, k.y); maxy = std::max(maxy, k.y);
    }
    if(n >= 1 && maxx > minx)
      pose.box = {minx, miny, maxx - minx, maxy - miny};
    return n;
  };
  if(build(0.5f) < 3)
    build(0.2f);
}

// Prepare the output texture before drawing: copy the input frame in, or fill
// opaque black for skeleton-only mode. The ctx overlay then draws onto it.
inline void fillCanvas(
    unsigned char* dst, const unsigned char* src, int w, int h,
    bool skeleton_only)
{
  const size_t n = static_cast<size_t>(w) * h * 4;
  if(skeleton_only)
  {
    for(size_t i = 0; i < n; i += 4)
    {
      dst[i] = dst[i + 1] = dst[i + 2] = 0;
      dst[i + 3] = 255;
    }
  }
  else
  {
    std::memcpy(dst, src, n);
  }
}

// Map a classified model role to the PoseWorkflow used for drawing/skeletons.
inline PoseWorkflow workflowForRole(const Onnx::ModelRole& r)
{
  using K = Onnx::ModelKind;
  switch(r.kind)
  {
    case K::BlazePoseLandmark:
    case K::BlazePoseDetector:
      return PoseWorkflow::BlazePose;
    case K::HandLandmark:
    case K::PalmDetector:
      return PoseWorkflow::MediaPipeHands;
    case K::FaceMeshLandmark:
      return PoseWorkflow::FaceMesh;
    case K::MobileFaceNet:
      return PoseWorkflow::MobileFaceNet;
    case K::SimccPose:
      if(r.domain == Onnx::ModelDomain::Animal)
        return PoseWorkflow::AnimalPose;
      if(r.domain == Onnx::ModelDomain::Face)
        return PoseWorkflow::RTMPoseFace;
      return (r.num_keypoints > 50) ? PoseWorkflow::RTMPose_Whole
                                    : PoseWorkflow::RTMPose_COCO;
    case K::HeatmapPose:
      if(r.domain == Onnx::ModelDomain::Animal)
        return PoseWorkflow::AnimalPose;
      // A 68-keypoint heatmap is a dlib/300W FACE alignment model (2DFAN-4
      // etc.), NOT a body pose — route it to the face path so it draws the
      // dlib-68 face mesh, not a COCO body skeleton (the "spider web").
      if(r.domain == Onnx::ModelDomain::Face || r.num_keypoints == 68)
        return PoseWorkflow::MobileFaceNet;
      return PoseWorkflow::ViTPose;
    case K::YoloPose:
    case K::RtmoPose:
      return PoseWorkflow::YOLOPose;
    case K::MoveNetPose:
      return PoseWorkflow::RTMPose_COCO; // COCO-17 skeleton
    case K::XyScoreLandmark:
      if(r.domain == Onnx::ModelDomain::Hand)
        return PoseWorkflow::MediaPipeHands;
      if(r.domain == Onnx::ModelDomain::Face)
        return PoseWorkflow::MobileFaceNet; // generic face dots
      return (r.num_keypoints > 50) ? PoseWorkflow::RTMPose_Whole
                                    : PoseWorkflow::RTMPose_COCO;
    case K::BlazeFaceDetector:
    case K::RetinaFaceDetector:
    case K::FaceBoxesDetector:
      return PoseWorkflow::BlazeFace;
    case K::PersonDetector:
    case K::MultiClassDetector:
    case K::YoloxDetector:
      return PoseWorkflow::BoxDetection;
    default:
      return PoseWorkflow::BlazePose;
  }
}

// Normalization + layout for the fused samplers: out = (channel - mean)*invstd.
struct NormSpec
{
  Onnx::TensorLayout layout{Onnx::TensorLayout::NchwRgb};
  std::array<float, 3> mean{0, 0, 0};
  std::array<float, 3> invstd{1, 1, 1};
};

inline NormSpec normMeanStd(
    Onnx::TensorLayout layout, std::array<float, 3> mean, std::array<float, 3> std)
{
  return {layout, mean, {1.f / std[0], 1.f / std[1], 1.f / std[2]}};
}
// out = (px/255)*a + b  <=>  (px - mean)/std  with std = 255/a, mean = -b*std.
inline NormSpec normAB(Onnx::TensorLayout layout, float a, float b)
{
  const float s = 255.f / a;
  return normMeanStd(layout, {-b * s, -b * s, -b * s}, {s, s, s});
}

// Finalize a packed float buffer into an Ort tensor (batch forced to 1). The
// concrete (mw,mh) we just sampled into `storage` are used to resolve any
// dynamic (-1) spatial/channel dims in the model's declared shape — otherwise
// ORT gets a tensor descriptor with a negative element count and throws every
// frame (the bug that made dynamic-input detectors render black).
inline Onnx::FloatTensor finalizeTensor(
    const Onnx::ModelSpec::Port& port, boost::container::vector<float>& storage,
    int mw, int mh, bool nhwc)
{
  std::vector<std::int64_t> shape = port.shape;
  if(shape.size() == 4)
  {
    shape[0] = 1; // batch
    if(nhwc) // [1,H,W,C]
    {
      if(shape[1] <= 0) shape[1] = mh;
      if(shape[2] <= 0) shape[2] = mw;
      if(shape[3] <= 0) shape[3] = 3;
    }
    else // [1,C,H,W]
    {
      if(shape[1] <= 0) shape[1] = 3;
      if(shape[2] <= 0) shape[2] = mh;
      if(shape[3] <= 0) shape[3] = mw;
    }
  }
  else if(!shape.empty())
  {
    shape[0] = 1;
  }
  Onnx::FloatTensor f{
      .storage = {}, .value = Onnx::vec_to_tensor<float>(storage, shape)};
  f.storage = std::move(storage);
  return f;
}

// Fused: sample src through M (output px -> src px), normalize per `ns`, into a
// reused float buffer, then finalize to an Ort tensor (batch forced to 1). No
// intermediate RGBA buffer, no second normalize pass.
inline Onnx::FloatTensor fusedAffineTensor(
    const Onnx::ModelSpec::Port& port, const Onnx::ImageView& src,
    const Onnx::Affine& M, int mw, int mh, const NormSpec& ns,
    boost::container::vector<float>& storage,
    Onnx::prof::Bucket prof_bucket = Onnx::prof::WarpCrop)
{
  storage.resize(static_cast<size_t>(3) * mw * mh, boost::container::default_init);
  Onnx::sampleAffineToTensor(
      ns.layout, src, M, mw, mh, ns.mean.data(), ns.invstd.data(),
      storage.data(), prof_bucket);
  return finalizeTensor(
      port, storage, mw, mh, ns.layout == Onnx::TensorLayout::NhwcRgb);
}

// Fused: aspect-preserving letterbox + normalize into a reused buffer, finalize.
inline Onnx::FloatTensor fusedLetterboxTensor(
    const Onnx::ModelSpec::Port& port, const Onnx::ImageView& src, int mw, int mh,
    bool center, const NormSpec& ns, boost::container::vector<float>& storage,
    Onnx::LetterboxInfo& lb_out)
{
  storage.resize(static_cast<size_t>(3) * mw * mh, boost::container::default_init);
  lb_out = Onnx::letterboxToTensor(
      ns.layout, src, mw, mh, center, /*pad=*/0, ns.mean.data(),
      ns.invstd.data(), storage.data());
  return finalizeTensor(
      port, storage, mw, mh, ns.layout == Onnx::TensorLayout::NhwcRgb);
}

// A tracking ROI must be finite, non-tiny, and centered inside the frame.
inline bool rectValid(const Onnx::ROI::Rect& r, int W, int H)
{
  if(!std::isfinite(r.cx) || !std::isfinite(r.cy) || !std::isfinite(r.w)
     || !std::isfinite(r.h) || !std::isfinite(r.angle))
    return false;
  if(r.w < 0.04f * W || r.h < 0.04f * H)
    return false;
  if(r.cx < 0 || r.cx > W || r.cy < 0 || r.cy > H)
    return false;
  return true;
}
// Reject a tracked ROI that teleported or changed size implausibly vs the
// previous one (prevents drift/collapse from compounding — re-detect instead).
inline bool rectPlausible(const Onnx::ROI::Rect& c, const Onnx::ROI::Rect& p)
{
  const float move = std::hypot(c.cx - p.cx, c.cy - p.cy);
  if(move > 0.6f * std::max(p.w, p.h))
    return false;
  const float ratio = c.w / std::max(1.0f, p.w);
  if(ratio < 0.66f || ratio > 1.5f) // implausible per-frame size jump -> re-detect
    return false;
  return true;
}
// Deadband: a near-identical ROI is treated as unchanged, so a STILL subject
// yields the exact same crop every frame and the keypoints stop oscillating
// (the tracking ROI is a feedback loop; without this it shakes on a static
// image).
inline bool rectClose(const Onnx::ROI::Rect& c, const Onnx::ROI::Rect& p)
{
  const float sz = std::max(p.w, p.h);
  if(std::hypot(c.cx - p.cx, c.cy - p.cy) > 0.015f * sz)
    return false;
  if(std::fabs(c.w - p.w) > 0.03f * p.w || std::fabs(c.h - p.h) > 0.03f * p.h)
    return false;
  if(std::fabs(c.angle - p.angle) > 0.02f)
    return false;
  return true;
}

} // namespace OnnxModels
