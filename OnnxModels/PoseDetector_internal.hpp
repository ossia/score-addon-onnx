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

// ossia::safe_isnan / safe_isinf come from a vendored, API-compatible copy
// under Onnx/helpers/compat so this header carries no ossia/ include in either
// build (the fast-math-safe bit-pattern checks are identical to ossia's).
#include <Onnx/helpers/compat/safe_math.hpp>

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
  // Confidence-weighted ("soft") bbox. The previous version used a hard
  // confidence cut with a count-based 0.5<->0.2 fallback, which is BISTABLE: a
  // far joint (foot/ear/hand) flicking across the cut, or the confident-joint
  // count crossing 3, swapped the point set and made the box halve/double
  // frame-to-frame. Here each joint's pull on the box edge ramps smoothly with
  // its confidence (smoothstep over [lo,hi]); a fading joint retracts toward the
  // confidence-weighted centroid, so the extent moves continuously instead of
  // popping. wholebody-133's many low-confidence face/hand points barely move
  // the box.
  constexpr float lo = 0.2f, hi = 0.5f;
  auto weight = [](float c) {
    const float t = std::clamp((c - lo) / (hi - lo), 0.f, 1.f);
    return t * t * (3.f - 2.f * t); // smoothstep
  };
  float sw = 0.f, cx = 0.f, cy = 0.f;
  for(const auto& k : pose.keypoints)
  {
    const float w = weight(k.confidence);
    sw += w;
    cx += w * k.x;
    cy += w * k.y;
  }
  if(sw <= 1e-6f)
  {
    // No confident joint: fall back to the unweighted bbox of every joint so a
    // weak pose still gets a (rough) box rather than none.
    float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
    int n = 0;
    for(const auto& k : pose.keypoints)
    {
      ++n;
      minx = std::min(minx, k.x); maxx = std::max(maxx, k.x);
      miny = std::min(miny, k.y); maxy = std::max(maxy, k.y);
    }
    if(n >= 1 && maxx > minx && maxy > miny)
      pose.box = {minx, miny, maxx - minx, maxy - miny};
    return;
  }
  cx /= sw;
  cy /= sw;
  float minx = cx, maxx = cx, miny = cy, maxy = cy;
  for(const auto& k : pose.keypoints)
  {
    const float w = weight(k.confidence);
    const float ex = cx + w * (k.x - cx); // retracts to centroid as conf fades
    const float ey = cy + w * (k.y - cy);
    minx = std::min(minx, ex); maxx = std::max(maxx, ex);
    miny = std::min(miny, ey); maxy = std::max(maxy, ey);
  }
  if(maxx > minx && maxy > miny)
    pose.box = {minx, miny, maxx - minx, maxy - miny};
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
    case K::InstantHmr:
      return PoseWorkflow::InstantHMR;
    default:
      return PoseWorkflow::BlazePose;
  }
}

// One-Euro parameters for the 3D payload (DetectedPose::world / translation /
// body_params), from the node's Smoothing Amount. Real units (Hz, beta per
// metre/s or rad/s, stepped at Onnx::kNominalFrameDt): these are metric
// signals, NOT the normalized [0,1] screen coordinates applySmoothing tunes
// for, so the published per-signal settings are used as the anchor. At the
// default Amount 0.5 they are exactly upstream InstantHMR's (smoothing.py):
// joints_3d_local (min_cutoff 1.0 Hz, beta 4.0), cam_trans (0.6, 2.0), dcutoff
// 1 Hz. The Amount then bends them the same way applySmoothing bends the 2D
// ones: min_cutoff along the same exponential curve, beta along the same
// linear (1 + 4*amt) ramp normalized to 1 at 0.5, so heavy smoothing still
// opens up on fast motion.
// body_params are generic (units unknown here), so their default is a pure
// low-pass (beta 0) at the joints' cutoff; a model family that knows its
// parameters passes its own beta anchor (params_beta, per unit/s at Amount
// 0.5) and flags its angles / identity entries via the ParamKind mask.
// InstantHMR: upstream's mhr_params (1.0 Hz, beta 2.0 per rad/s) and its
// shape_params (0.3 Hz, beta 0) through ParamStatic.
struct Smoothing3D
{
  float world_min_cutoff, world_beta;
  float trans_min_cutoff, trans_beta;
  float params_min_cutoff, params_beta;
};
inline Smoothing3D smoothing3D(float amount, float params_beta = 0.f) noexcept
{
  const float amt = std::clamp(amount, 0.f, 1.f);
  const float mc = std::pow(0.02f / 5.0f, amt - 0.5f); // 1 at amt = 0.5
  const float b = (1.0f + 4.0f * amt) / 3.0f;          // 1 at amt = 0.5
  return {1.0f * mc, 4.0f * b, 0.6f * mc, 2.0f * b, 1.0f * mc, params_beta * b};
}

// --- InstantHMR (human mesh recovery) helpers, shared by the load, detect and
//     landmark files (and the tests) ---

// Upstream CROP_EXPAND: the person crop is the square max(bw,bh) * 1.2 around
// the detector box centre (instanthmr/inference.py _preprocess).
inline constexpr float kHmrCropExpand = 1.2f;
// Sizes of the MHR parameter blocks InstantHMR emits (DetectedPose::body_params
// = mhr_params ++ shape_params).
inline constexpr int kMhrPoseParams = 204;
inline constexpr int kMhrShapeParams = 45;

// The CLIFF box condition of one crop, from the ROI rect the crop was sampled
// with (image px). The rect is topdownRect(box, S, S, kHmrCropExpand), so its
// centre is the detector box centre and max(w,h) / 1.2 is max(bw,bh): exactly
// upstream's vector, but of the SMOOTHED ROI the crop really used, so crop and
// condition can never disagree. W, H: full frame.
//  pixel form (the shipped graph): [2cx/W - 1, 2cy/H - 1, max(bw,bh)/max(W,H)]
//  angular form (metadata cliff_focal=true, newer tools/pth_to_onnx.py
//  exports), f = 1.05 * diag:     [atan((cx-W/2)/f), atan((cy-H/2)/f), max/f]
// The focal length (px) of InstantHMR's camera for a W x H frame: the one its
// cam_trans places the person for, so every projection of the 3D payload
// (Draw Mesh, Mesh Keypoints) must use it. Pixel CLIFF: the frame diagonal;
// angular CLIFF (cliff_focal): 1.05 x the diagonal.
inline float instantHmrFocal(int W, int H, bool angular) noexcept
{
  const float fw = static_cast<float>(W), fh = static_cast<float>(H);
  return (angular ? 1.05f : 1.f) * std::sqrt(fw * fw + fh * fh);
}

inline void instantHmrCliff(
    const Onnx::ROI::Rect& r, int W, int H, bool angular, float* out) noexcept
{
  const float side = std::max(r.w, r.h) / kHmrCropExpand;
  const float fw = static_cast<float>(W), fh = static_cast<float>(H);
  if(angular)
  {
    const float f = instantHmrFocal(W, H, true);
    out[0] = std::atan((r.cx - 0.5f * fw) / f);
    out[1] = std::atan((r.cy - 0.5f * fh) / f);
    out[2] = side / f;
  }
  else
  {
    out[0] = 2.f * r.cx / fw - 1.f;
    out[1] = 2.f * r.cy / fh - 1.f;
    out[2] = side / std::max(fw, fh);
  }
}

// The (unrotated) ROI rect an affine from rectToAffine(r, mw, mh) was built
// from: rectToAffine at angle 0 is m0 = w/mw, m2 = cx - w/2 (and likewise in
// y), so it inverts exactly. Lets the CLIFF builder work from the M every
// landmark call already carries, with no rect threaded next to it.
inline Onnx::ROI::Rect rectFromAffine(const Onnx::Affine& M, int mw, int mh) noexcept
{
  const float w = M.m0 * mw, h = M.m4 * mh;
  return {M.m2 + 0.5f * w, M.m5 + 0.5f * h, w, h, 0.f};
}

// The affine InstantHMR's crop is SAMPLED with, from the crop affine M that
// maps its keypoints back. Upstream resizes the square patch with
// cv2.resize(INTER_LINEAR), which reads crop pixel u at the patch's
// (u + 0.5) * s - 0.5 (pixel centres), while it maps the regressed joints back
// with u * s (pixel corners), and so does M. Our sampler reads u * s, i.e.
// (s - 1) / 2 source px (1 px for a 672 px square) off what the model was
// trained on: shifting only the sampling reproduces upstream exactly, and the
// back-map stays M. Axis-aligned M only (InstantHMR ROIs are never rotated).
inline Onnx::Affine instantHmrSampleAffine(Onnx::Affine M) noexcept
{
  M.m2 += 0.5f * (M.m0 - 1.f);
  M.m5 += 0.5f * (M.m4 - 1.f);
  return M;
}

// InstantHMR's whole-frame ROI (no Detection Model): the frame itself is the
// person box, cropped exactly as upstream crops a detector box (1.2x square,
// black outside). The same rect then drives the CLIFF vector.
inline Onnx::ROI::Rect instantHmrFrameRect(int W, int H, int mw, int mh) noexcept
{
  return Onnx::ROI::topdownRect(
      Onnx::Rect{0.f, 0.f, static_cast<float>(W), static_cast<float>(H)}, mw,
      mh, kHmrCropExpand);
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
// intermediate RGBA buffer, no second normalize pass. border_rgb: nullptr =
// edge-clamped sampling (every existing caller), else the constant RGB colour
// written where the crop leaves the frame (see Onnx::sampleAffineToTensor).
inline Onnx::FloatTensor fusedAffineTensor(
    const Onnx::ModelSpec::Port& port, const Onnx::ImageView& src,
    const Onnx::Affine& M, int mw, int mh, const NormSpec& ns,
    boost::container::vector<float>& storage,
    Onnx::prof::Bucket prof_bucket = Onnx::prof::WarpCrop,
    const uint8_t* border_rgb = nullptr)
{
  storage.resize(static_cast<size_t>(3) * mw * mh, boost::container::default_init);
  Onnx::sampleAffineToTensor(
      ns.layout, src, M, mw, mh, ns.mean.data(), ns.invstd.data(),
      storage.data(), prof_bucket, border_rgb);
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


// Classifier input from a model: when its declared output dims are all
// dynamic (3DDFA's FaceBoxesProd under a provider that turns ORT's shape
// inference off, e.g. OpenVINO), the classifier cannot tell what it is. Run it
// once on a zero image (320 px for dynamic sides) and use the real output
// shapes instead. Returns false when no probe was needed or it failed.
inline bool probeOutputShapes(Onnx::OnnxRunContext& ctx, Onnx::ModelIO& io)
{
  const auto& spec = ctx.readModelSpec();
  if(spec.inputs.size() != 1 || spec.inputs[0].shape.size() != 4
     || spec.inputs[0].elem_type != Onnx::TensorElemType::Float)
    return false;
  const bool dynamic_out = std::any_of(
      spec.outputs.begin(), spec.outputs.end(), [](const auto& o) {
        return std::any_of(o.shape.begin(), o.shape.end(), [](int64_t d) { return d <= 0; });
      });
  if(!dynamic_out)
    return false;
  try
  {
    std::vector<int64_t> shape = spec.inputs[0].shape;
    shape[0] = 1;
    const bool nhwc = shape[3] == 3 || shape[3] == 1;
    for(std::size_t d = 1; d < 4; ++d)
      if(shape[d] <= 0)
        shape[d] = (!nhwc && d == 1) || (nhwc && d == 3) ? 3 : 320;
    std::size_t n = 1;
    for(auto d : shape)
      n *= (std::size_t)d;
    std::vector<float> zeros(n, 0.f);
    Ort::Value ins[1]{Onnx::vec_to_tensor<float>(zeros, shape)};
    std::vector<Ort::Value> outs;
    for(std::size_t i = 0; i < spec.outputs.size(); ++i)
      outs.emplace_back(nullptr);
    ctx.infer(spec, ins, outs);
    for(std::size_t i = 0; i < outs.size() && i < io.outputs.size(); ++i)
      if(outs[i] && outs[i].IsTensor())
        io.outputs[i].shape = outs[i].GetTensorTypeAndShapeInfo().GetShape();
    return true;
  }
  catch(...)
  {
    return false;
  }
}
} // namespace OnnxModels
