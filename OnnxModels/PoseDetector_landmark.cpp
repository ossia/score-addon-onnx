#include "PoseDetector_internal.hpp"

namespace OnnxModels
{

namespace
{
// One decoded landmark in MODEL-PIXEL space (x,y in [0,mw]x[0,mh]).
struct LandmarkKp
{
  float x, y, z, conf;
};

// Upper bound on the landmark outputs fetched per inference (the Ort::Value
// arrays below are fixed-size, no per-frame allocation). InstantHMR has 5
// (mhr_params, shape_params, cam_trans, joints_2d, joints_3d); one spare.
constexpr std::size_t kMaxLandmarkOutputs = 6;

// Where decodeLandmark leaves a mesh-recovery model's extra 3D payload
// (InstantHMR): the cached output indices, and the camera translation /
// body-parameter sinks. Null members = not wanted / not available.
struct HmrSink
{
  const Onnx::HmrOutputs* idx = nullptr;
  std::vector<float>* translation = nullptr;
  std::vector<float>* params = nullptr;
  // The pelvis in the rig frame (DetectedPose::rig_offset): set together
  // with translation, so translation - rig_offset is the raw cam_trans.
  std::vector<float>* rig_offset = nullptr;
};

// InstantHMR regresses joints_2d directly in crop coordinates [-1,1]; a joint
// placed beyond the padded crop (|u| > 1.25, i.e. well outside even the 1.2x
// margin) is an extrapolation the model never saw supervised: no confidence.
constexpr float kHmrCropLimit = 1.25f;

// Layout + normalization for a landmark model's input crop (used by the fused
// sampler to write the model input directly, no intermediate RGBA buffer).
NormSpec landmarkNorm(const Onnx::ModelRole& role)
{
  // MoveNet wants RAW [0,255] (NHWC, no scaling) — validated against the real
  // exports (tests/validate_decoders.py case A). Feeding it [0,1] gives a
  // near-black input and the keypoints collapse, so it must come before the
  // generic nhwc [0,1] branch below.
  if(role.kind == Onnx::ModelKind::MoveNetPose)
    return normMeanStd(
        Onnx::TensorLayout::NhwcRgb, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f});
  if(role.nhwc)
    return normAB(Onnx::TensorLayout::NhwcRgb, 1.f, 0.f);
  // InstantHMR: ImageNet mean/std on RGB in [0,1] (upstream IMAGENET_MEAN /
  // IMAGENET_STD) — the same numbers as the final fall-through, spelled out
  // so a later reorder of the branches above can't change it.
  if(role.kind == Onnx::ModelKind::InstantHmr)
    return normMeanStd(
        Onnx::TensorLayout::NchwRgb,
        {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f},
        {0.229f * 255.f, 0.224f * 255.f, 0.225f * 255.f});
  if(role.kind == Onnx::ModelKind::MobileFaceNet)
    return normMeanStd(
        Onnx::TensorLayout::NchwRgb,
        {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f},
        {0.229f * 255.f, 0.224f * 255.f, 0.225f * 255.f});
  if(role.kind == Onnx::ModelKind::FaceMeshLandmark
     || role.kind == Onnx::ModelKind::HandLandmark
     || role.kind == Onnx::ModelKind::BlazePoseLandmark)
    // MediaPipe landmark models want [0,1] regardless of layout.
    return normMeanStd(
        Onnx::TensorLayout::NchwRgb, {0.f, 0.f, 0.f}, {255.f, 255.f, 255.f});
  // xy-score landmark: the PINTO mmpose with_post exports (extra [1,2] bbox
  // input) want the mmpose mean/std below; single-input ones (Peppa-Pig) take
  // [0,1] (validated: /mnt/models/validate_decoders.py case B).
  if(role.kind == Onnx::ModelKind::XyScoreLandmark && role.num_inputs < 2)
    return normMeanStd(
        Onnx::TensorLayout::NchwRgb, {0.f, 0.f, 0.f}, {255.f, 255.f, 255.f});
  // 2D-FAN / face-alignment heatmap nets (68/98/106 kpts) take RGB in [0,1]
  // (img/255), NOT the ImageNet mean/std the body heatmap nets (ViTPose/HRNet)
  // below use. Verified upstream: 1adrianb/face-alignment api.py feeds
  // crop(...).astype(float32) / 255.0 with no mean/std. Feeding ImageNet
  // normalization instead flattens the heatmaps and scatters the keypoints.
  if(role.kind == Onnx::ModelKind::HeatmapPose
     && role.domain == Onnx::ModelDomain::Face)
    return normMeanStd(
        Onnx::TensorLayout::NchwRgb, {0.f, 0.f, 0.f}, {255.f, 255.f, 255.f});
  return normMeanStd(
      Onnx::TensorLayout::NchwRgb, {123.675f, 116.28f, 103.53f},
      {58.395f, 57.12f, 57.375f});
}

// Out-of-frame fill for a landmark crop (see Onnx::sampleAffineToTensor):
// nullptr = edge clamp. A family trained on constant-padded crops returns its
// pad colour here (InstantHMR: copyMakeBorder(BORDER_CONSTANT, 0) -> kBlackBorder).
// Shared by the single-crop and batched paths so they can't disagree.
constexpr uint8_t kBlackBorder[3] = {0, 0, 0};
const uint8_t* landmarkBorder(const Onnx::ModelRole& role)
{
  if(role.kind == Onnx::ModelKind::InstantHmr)
    return kBlackBorder;
  return nullptr;
}

// The affine a landmark crop is sampled through; keypoints still map back
// through the crop affine itself. Only InstantHMR differs (pixel-centre
// resize, see instantHmrSampleAffine). Shared by both paths like the border.
Onnx::Affine landmarkSampleAffine(const Onnx::ModelRole& role, const Onnx::Affine& M)
{
  return role.kind == Onnx::ModelKind::InstantHmr ? instantHmrSampleAffine(M) : M;
}

// A landmark model's second input, per element type: SimCC's int64 [N,2]
// bbox (w,h of the crop), or InstantHMR's float [N,3] CLIFF box condition,
// derived from the crop's own ROI (rectFromAffine: InstantHMR ROIs are never
// rotated).
bool isCliffInput(const Onnx::ModelSpec& spec)
{
  return spec.inputs.size() >= 2
         && spec.inputs[1].elem_type == Onnx::TensorElemType::Float;
}

// Decode ONE instance's landmark outputs (a [1,...] outspan) into MODEL-PIXEL
// keypoints. Shared by the single-crop and batched-slice paths. `world` (when
// non-null) receives the model's world-space 3D keypoints in METERS
// (hip-origin) for models that emit them — BlazePose full-body's [1,117]
// (39*3) world output. World coordinates are crop-independent: they bypass the
// affine un-mapping the screen keypoints go through.
void decodeLandmark(
    const Onnx::ModelRole& role, const Onnx::ModelSpec& spec,
    std::span<Ort::Value> outspan, int mw, int mh, float min_conf,
    std::vector<LandmarkKp>& kps, std::vector<LandmarkKp>* world = nullptr,
    bool crop_padded = false, const HmrSink& hmr = {})
{
  kps.clear();
  if(world)
    world->clear();
  if(hmr.translation)
    hmr.translation->clear();
  if(hmr.rig_offset)
    hmr.rig_offset->clear();
  if(hmr.params)
    hmr.params->clear();
  switch(role.kind)
  {
    case Onnx::ModelKind::BlazePoseLandmark:
    {
      // Handle every variant: full-body 195 (=39*5), upper-body 155 (=31*5)
      // or 124 (=31*4). Find the landmark vector output and derive the layout.
      const float* data = nullptr;
      int total = 0;
      for(size_t i = 0; i < outspan.size(); ++i)
      {
        auto info = outspan[i].GetTensorTypeAndShapeInfo();
        if(info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
          continue; // fp16/u8 buffers read as float would over-read
        const int64_t n = info.GetElementCount();
        if(n == 195 || n == 155 || n == 124)
        {
          data = outspan[i].GetTensorData<float>();
          total = static_cast<int>(n);
          break;
        }
      }
      if(data)
      {
        const int stride = (total % 5 == 0) ? 5 : 4; // x,y,z,vis[,pres]
        const int K = total / stride;
        const int body = std::min(K, 33); // first K are body, rest are aux
        kps.reserve(body);
        for(int i = 0; i < body; ++i)
        {
          const float* kp = data + i * stride;
          const float vis = 1.0f / (1.0f + std::exp(-kp[3]));
          const float pres
              = (stride == 5) ? 1.0f / (1.0f + std::exp(-kp[4])) : vis;
          kps.push_back({kp[0], kp[1], kp[2] / mw, pres});
        }

        // World-3D sidecar: a [1,117] (39*3) output of x,y,z in meters,
        // hip-origin (full-body models only). Same joint order as the screen
        // keypoints; confidences copied from them.
        if(world)
        {
          for(size_t i = 0; i < outspan.size(); ++i)
          {
            auto winfo = outspan[i].GetTensorTypeAndShapeInfo();
            if(winfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
              continue;
            const int64_t n = winfo.GetElementCount();
            if(n == 117)
            {
              const float* w = outspan[i].GetTensorData<float>();
              const int wk = std::min<int>(static_cast<int>(n / 3), body);
              world->reserve(wk);
              for(int k = 0; k < wk; ++k)
                world->push_back(
                    {w[k * 3], w[k * 3 + 1], w[k * 3 + 2],
                     k < static_cast<int>(kps.size()) ? kps[k].conf : 1.f});
              break;
            }
          }
        }
      }
      break;
    }
    case Onnx::ModelKind::HandLandmark:
    {
      std::optional<Onnx::MediaPipeHands::HandResult> r;
      Onnx::MediaPipeHands::processOutput(spec, outspan, r, min_conf);
      if(r)
      {
        kps.reserve(r->landmarks.size());
        for(const auto& lm : r->landmarks)
          kps.push_back({lm.x * mw, lm.y * mh, lm.z, r->hand_flag});
      }
      break;
    }
    case Onnx::ModelKind::FaceMeshLandmark:
    {
      std::optional<Onnx::FaceMesh::FaceMeshResult> r;
      Onnx::FaceMesh::processOutput(
          spec, outspan, Onnx::FaceMesh::NUM_LANDMARKS, r, min_conf);
      if(r)
      {
        // The model emits a face-presence score; use it instead of a hardcoded
        // 1.0 so confidence-based gating/coloring actually reflects the model.
        const float conf = std::clamp(r->face_flag, 0.f, 1.f);
        kps.reserve(r->landmarks.size());
        for(const auto& lm : r->landmarks)
          kps.push_back({lm.x * mw, lm.y * mh, lm.z, conf});
      }
      break;
    }
    case Onnx::ModelKind::SimccPose:
    {
      Onnx::RTMPose::OutputFormat fmt;
      auto config = Onnx::RTMPose::detectConfig(outspan, &fmt);
      config.input_width = mw;
      config.input_height = mh;
      std::optional<Onnx::RTMPose::PoseResult> r;
      Onnx::RTMPose::processOutput(spec, outspan, config, r, fmt);
      if(r)
      {
        kps.reserve(r->keypoints.size());
        for(const auto& kp : r->keypoints)
          // z: RTMW3D root-relative meters (simcc_z head); 0 for 2D RTMPose.
          kps.push_back({kp.x * mw, kp.y * mh, kp.z, kp.confidence});
      }
      break;
    }
    case Onnx::ModelKind::HeatmapPose:
    {
      // Stacked-hourglass face-alignment nets (2D-FAN) emit one heatmap per
      // stage; the refined prediction is the LAST stage (1adrianb/face-alignment
      // decodes out[-1]). Single-stage nets (ViTPose/HRNet) emit one heatmap, so
      // this picks that same one. Select the last [1,K,h,w] float output.
      const Ort::Value* hv = nullptr;
      for(size_t i = 0; i < outspan.size(); ++i)
      {
        if(!outspan[i].IsTensor())
          continue;
        auto hi = outspan[i].GetTensorTypeAndShapeInfo();
        if(hi.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
          continue;
        const auto hs = hi.GetShape();
        if(hs.size() == 4 && hs[2] > 1 && hs[3] > 1)
          hv = &outspan[i];
      }
      if(hv)
      {
        auto info = hv->GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        {
          const int hh = static_cast<int>(shape[2]);
          const int hw = static_cast<int>(shape[3]);
          // hw==0 would divide by zero (max_idx % hw); a declared K larger than
          // the real buffer would over-read -> clamp to actual element count.
          if(hh <= 0 || hw <= 0)
            break;
          const int64_t total = static_cast<int64_t>(info.GetElementCount());
          const int K = static_cast<int>(std::min<int64_t>(
              shape[1] > 0 ? shape[1] : 0, total / (static_cast<int64_t>(hh) * hw)));
          const float* hmaps = hv->GetTensorData<float>();
          kps.reserve(K);
          for(int k = 0; k < K; ++k)
          {
            const float* hm = hmaps + static_cast<size_t>(k) * hh * hw;
            int max_idx = 0;
            float max_val = hm[0];
            for(int i = 1; i < hh * hw; ++i)
              if(hm[i] > max_val)
              {
                max_val = hm[i];
                max_idx = i;
              }
            const int hx = max_idx % hw;
            const int hy = max_idx / hw;

            // DARK sub-pixel refinement (mmpose's heatmap decode).
            float ox = 0.f, oy = 0.f;
            if(hx >= 1 && hx < hw - 1 && hy >= 1 && hy < hh - 1)
            {
              auto L = [&](int x, int y) {
                return std::log(std::max(hm[y * hw + x], 1e-10f));
              };
              const float dx = 0.5f * (L(hx + 1, hy) - L(hx - 1, hy));
              const float dy = 0.5f * (L(hx, hy + 1) - L(hx, hy - 1));
              const float dxx = L(hx + 1, hy) - 2.f * L(hx, hy) + L(hx - 1, hy);
              const float dyy = L(hx, hy + 1) - 2.f * L(hx, hy) + L(hx, hy - 1);
              const float dxy = 0.25f
                                * (L(hx + 1, hy + 1) - L(hx + 1, hy - 1)
                                   - L(hx - 1, hy + 1) + L(hx - 1, hy - 1));
              const float det = dxx * dyy - dxy * dxy;
              if(std::fabs(det) > 1e-9f)
              {
                ox = std::clamp(-(dyy * dx - dxy * dy) / det, -1.f, 1.f);
                oy = std::clamp(-(dxx * dy - dxy * dx) / det, -1.f, 1.f);
              }
            }
            const float mx = (hx + ox + 0.5f) * mw / hw;
            const float my = (hy + oy + 0.5f) * mh / hh;
            kps.push_back({mx, my, 0.0f, max_val});
          }
        }
      }
      break;
    }
    case Onnx::ModelKind::MobileFaceNet:
    {
      if(!outspan.empty())
      {
        auto info = outspan[0].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        if(shape.size() >= 2
           && info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
          // The output is a flat (x,y) vector: [1,136] for the 68-landmark
          // export. shape[1] is the FLOAT count, not the landmark count —
          // reading shape[1] pairs would run 2x off the end of the buffer.
          const int n = static_cast<int>(info.GetElementCount() / 2);
          const float* data = outspan[0].GetTensorData<float>();
          kps.reserve(n);
          for(int i = 0; i < n; ++i)
            kps.push_back({data[i * 2] * mw, data[i * 2 + 1] * mh, 0.0f, 1.0f});
        }
      }
      break;
    }
    case Onnx::ModelKind::MoveNetPose:
    {
      // Single [1,1,K,3] output, rows are (y, x, score) normalized [0,1]
      // (validated on real ailia + PINTO exports).
      if(!outspan.empty())
      {
        auto info = outspan[0].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        if(shape.size() == 4 && shape[3] == 3
           && info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
          // Clamp by the real payload: a zero/dynamic leading dim would make
          // shape[2] rows read from an empty buffer.
          const int K = static_cast<int>(
              std::min<int64_t>(shape[2], info.GetElementCount() / 3));
          const float* data = outspan[0].GetTensorData<float>();
          kps.reserve(K);
          for(int i = 0; i < K; ++i)
          {
            const float* kp = data + i * 3;
            kps.push_back({kp[1] * mw, kp[0] * mh, 0.0f, kp[2]});
          }
        }
      }
      break;
    }
    case Onnx::ModelKind::InstantHmr:
    {
      // Outputs located once at load (HmrOutputs), each a [1,...] slice here.
      // A slot that is missing, not float32 or too short is treated as absent.
      if(!hmr.idx || !hmr.idx->valid())
        break;
      const auto& h = *hmr.idx;
      auto floats = [&](int i, int64_t n) -> const float* {
        if(i < 0 || i >= static_cast<int>(outspan.size()) || !outspan[i]
           || !outspan[i].IsTensor())
          return nullptr;
        auto info = outspan[i].GetTensorTypeAndShapeInfo();
        if(info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
           || static_cast<int64_t>(info.GetElementCount()) < n)
          return nullptr;
        return outspan[i].GetTensorData<float>();
      };
      const int K = h.num_keypoints;
      const float* j2 = K > 0 ? floats(h.joints_2d, 2 * K) : nullptr;
      if(!j2)
        break;

      // 2D: crop coords [-1,1] -> model px (upstream: (j + 1) * 0.5 * S); the
      // caller's affine then maps them to the frame. No per-joint score: 1
      // inside the padded crop, 0 beyond it or when non-finite (fp16 inside).
      // The detector score multiplies it at the call site.
      kps.reserve(K);
      for(int k = 0; k < K; ++k)
      {
        const float u = j2[2 * k], v = j2[2 * k + 1];
        if(!finitef(u) || !finitef(v))
        {
          kps.push_back({0.f, 0.f, 0.f, 0.f}); // keep the MHR70 index layout
          continue;
        }
        const bool inside
            = std::fabs(u) <= kHmrCropLimit && std::fabs(v) <= kHmrCropLimit;
        kps.push_back(
            {(u + 1.f) * 0.5f * mw, (v + 1.f) * 0.5f * mh, 0.f,
             inside ? 1.f : 0.f});
      }

      // 3D: joints_3d is rig-local (X right, Y down, Z forward, metres) with
      // an origin near the floor. world = pelvis-relative, the "hip-origin
      // metres" BlazePose's world already means, pelvis = mid(L hip 9, R hip
      // 10); the camera-space placement moves to translation = cam_trans +
      // pelvis, so world + translation == joints_3d + cam_trans exactly.
      const float* j3 = K > 10 ? floats(h.joints_3d, 3 * K) : nullptr;
      float pel[3] = {0.f, 0.f, 0.f};
      bool pel_ok = j3 != nullptr;
      for(int c = 0; pel_ok && c < 3; ++c)
      {
        pel[c] = 0.5f * (j3[9 * 3 + c] + j3[10 * 3 + c]);
        pel_ok = finitef(pel[c]);
      }
      if(pel_ok && world)
      {
        world->reserve(K);
        for(int k = 0; k < K; ++k)
        {
          const float* p = j3 + 3 * k;
          if(!finitef(p[0]) || !finitef(p[1]) || !finitef(p[2]))
          {
            world->push_back({0.f, 0.f, 0.f, 0.f});
            kps[k].conf = 0.f;
            continue;
          }
          world->push_back(
              {p[0] - pel[0], p[1] - pel[1], p[2] - pel[2], kps[k].conf});
          // Screen z: pelvis-relative depth in metres (RTMW3D's convention
          // for a metric z), so XYZArray carries it too.
          kps[k].z = p[2] - pel[2];
        }
      }
      if(pel_ok && hmr.translation)
      {
        const float* ct = floats(h.cam_trans, 3);
        if(ct && finitef(ct[0]) && finitef(ct[1]) && finitef(ct[2]))
        {
          hmr.translation->assign(
              {ct[0] + pel[0], ct[1] + pel[1], ct[2] + pel[2]});
          // The mesh (rig-local, origin = MHR joint 0) is placed at cam_trans,
          // not at the pelvis: keep the pelvis offset so translation -
          // rig_offset gives cam_trans back after both are smoothed.
          if(hmr.rig_offset)
            hmr.rig_offset->assign({pel[0], pel[1], pel[2]});
        }
      }

      // Body parameters: mhr_params (204) ++ shape_params (45), as emitted.
      // One non-finite entry drops the set for this frame (the tracker and
      // the single-path hold keep the last good one).
      if(hmr.params)
      {
        const float* mp = floats(h.mhr_params, kMhrPoseParams);
        const float* sp = floats(h.shape_params, kMhrShapeParams);
        if(mp)
        {
          auto& out = *hmr.params;
          out.insert(out.end(), mp, mp + kMhrPoseParams);
          if(sp)
            out.insert(out.end(), sp, sp + kMhrShapeParams);
          if(!std::all_of(out.begin(), out.end(), finitef))
            out.clear();
        }
      }
      break;
    }
    case Onnx::ModelKind::XyScoreLandmark:
    {
      // Single [1,K,3] output, rows are (x, y, score). Units vary by family:
      // Peppa-Pig emits normalized [0,1], the PINTO mmpose with_post graphs
      // emit crop pixels (we feed (mw,mh) as their bbox input) — sniff the
      // coordinate range. Scores can exceed 1 (raw SimCC amplitude): clamp.
      if(!outspan.empty())
      {
        auto info = outspan[0].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        if(shape.size() == 3 && shape[2] == 3
           && info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
          // Clamp by the real payload: a zero/dynamic leading dim would make
          // shape[1] rows read from an empty buffer.
          const int K = static_cast<int>(
              std::min<int64_t>(shape[1], info.GetElementCount() / 3));
          const float* data = outspan[0].GetTensorData<float>();
          float max_xy = 0.f;
          for(int i = 0; i < K; ++i)
          {
            max_xy = std::max(max_xy, std::fabs(data[i * 3]));
            max_xy = std::max(max_xy, std::fabs(data[i * 3 + 1]));
          }
          const bool normalized = max_xy <= 1.5f;
          kps.reserve(K);
          for(int i = 0; i < K; ++i)
          {
            const float* kp = data + i * 3;
            const float x = normalized ? kp[0] * mw : kp[0];
            const float y = normalized ? kp[1] * mh : kp[1];
            kps.push_back({x, y, 0.0f, std::clamp(kp[2], 0.f, 1.f)});
          }
        }
      }
      break;
    }
    default:
      break;
  }

  // Decode artifacts: top-down SimCC/heatmap pose models peg an ABSENT or
  // occluded joint to bin 0 / the last bin, i.e. a keypoint glued to the crop
  // edge (the "points stuck on the box border", e.g. rtmw3d / rtmpose / vitpose
  // on a partially-visible person). In a detector-cropped ROI (padded ~1.25x)
  // real joints sit well inside the edge, so drop (confidence = 0) any keypoint
  // within a few px of the model-crop border. Gated on crop_padded so the
  // whole-frame path — where a subject filling the frame has real head/feet
  // keypoints at the edge — is left untouched; and scoped to the top-down
  // decoders, since fill-the-crop models (FaceMesh/BlazePose/hand) legitimately
  // place points on the edge.
  if(crop_padded
     && (role.kind == Onnx::ModelKind::SimccPose
         || role.kind == Onnx::ModelKind::HeatmapPose))
  {
    const float mx = std::max(2.f, 0.015f * mw);
    const float my = std::max(2.f, 0.015f * mh);
    for(auto& k : kps)
      if(k.x <= mx || k.x >= mw - mx || k.y <= my || k.y >= mh - my)
        k.conf = 0.f;
  }
}
} // namespace

float PoseDetector::landmarkKeypoints(
    const Onnx::ModelRole& role, const Onnx::ImageView& src,
    const Onnx::Affine& M, std::vector<PoseKeypoint>& out,
    std::vector<PoseKeypoint>* out_world, bool crop_padded, float det_score)
{
  out.clear();
  if(out_world)
    out_world->clear();
  m_trans_scratch.clear();
  m_params_scratch.clear();
  m_offset_scratch.clear();
  auto& lctx = *this->ctx;
  const auto& spec = lctx.readModelSpec();
  if(spec.inputs.empty())
    return -1.f;

  int mw = role.input_w > 0 ? role.input_w : 256;
  int mh = role.input_h > 0 ? role.input_h : 256;
  if(spec.inputs[0].shape.size() == 4)
  {
    // Dynamic dims are -1: adopting them would turn the crop buffer resize
    // into a multi-exabyte request (or divide by zero in the decoders), so
    // only concrete dims may override the role defaults.
    const auto& s = spec.inputs[0].shape;
    const int64_t sh = role.nhwc ? s[1] : s[2];
    const int64_t sw = role.nhwc ? s[2] : s[3];
    if(sh > 0)
      mh = static_cast<int>(sh);
    if(sw > 0)
      mw = static_cast<int>(sw);
  }

  Onnx::FloatTensor t = fusedAffineTensor(
      spec.inputs[0], src, landmarkSampleAffine(role, M), mw, mh,
      landmarkNorm(role), storage, Onnx::prof::WarpCrop, landmarkBorder(role));

  Ort::Value outs[kMaxLandmarkOutputs]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}};
  const size_t n_out
      = std::min<size_t>(kMaxLandmarkOutputs, spec.output_names_char.size());

  // Some RTMPose exports take a second [w,h] bbox input.
  std::array<int64_t, 2> bbox_wh{mw, mh};
  std::array<int64_t, 2> bbox_shape{1, 2};
  if(isCliffInput(spec))
  {
    // InstantHMR: the [1,3] CLIFF condition of this crop, in the full frame.
    std::array<float, 3> cliff;
    instantHmrCliff(
        rectFromAffine(M, mw, mh), src.w, src.h, m_hmr_cliff_focal, cliff.data());
    std::array<int64_t, 2> cliff_shape{1, 3};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto cliff_tensor = Ort::Value::CreateTensor<float>(
        mem_info, cliff.data(), cliff.size(), cliff_shape.data(),
        cliff_shape.size());
    Ort::Value ins[2] = {std::move(t.value), std::move(cliff_tensor)};
    lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
  }
  else if(spec.inputs.size() >= 2)
  {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto bbox_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, bbox_wh.data(), bbox_wh.size(), bbox_shape.data(),
        bbox_shape.size());
    Ort::Value ins[2] = {std::move(t.value), std::move(bbox_tensor)};
    lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
  }
  else
  {
    Ort::Value ins[1] = {std::move(t.value)};
    lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
  }

  auto outspan = std::span<Ort::Value>(outs, n_out);

  // Decode scratch reused across frames (decodeLandmark clears it), so a crop
  // does not allocate. Function-local to this call site, so the batched path's
  // own scratch (which calls in here in its fallback loop) never aliases it.
  static thread_local std::vector<LandmarkKp> kps;
  static thread_local std::vector<LandmarkKp> wkps;
  decodeLandmark(
      role, spec, outspan, mw, mh, static_cast<float>(inputs.min_confidence),
      kps, out_world ? &wkps : nullptr, crop_padded,
      HmrSink{&m_hmr_out, &m_trans_scratch, &m_params_scratch, &m_offset_scratch});

  std::swap(storage, t.storage);

  if(kps.empty())
    return -1.f;

  // A model with no per-joint confidence (InstantHMR: 1 = usable, 0 = not)
  // takes its instance's detector score, and reports it as the mean.
  const bool score_conf = role.kind == Onnx::ModelKind::InstantHmr;
  const float cs = std::clamp(det_score, 0.f, 1.f);

  // Map model-pixel keypoints back through M -> image-normalized [0,1].
  const float iw = src.w, ih = src.h;
  out.reserve(kps.size());
  float sum_conf = 0.0f;
  for(const auto& k : kps)
  {
    const Onnx::Vec2 p = Onnx::ROI::applyAffine(M, k.x, k.y);
    const float c = score_conf ? k.conf * cs : k.conf;
    out.push_back({p.x / iw, p.y / ih, k.z, c});
    sum_conf += c;
  }
  // World coordinates are metric and crop-independent: no affine mapping.
  if(out_world)
  {
    out_world->reserve(wkps.size());
    for(const auto& k : wkps)
      out_world->push_back({k.x, k.y, k.z, score_conf ? k.conf * cs : k.conf});
  }
  if(score_conf)
    return sum_conf > 0.f ? cs : -1.f; // every joint unusable: no pose
  return sum_conf / out.size();
}

// Landmark every ROI, filling m_instances. Batches all crops into ONE
// [N,C,H,W] inference when the model's batch dim is dynamic/>=N (the common
// case); otherwise falls back to one inference per crop. Output decode is
// identical either way (each instance decoded from its own [1,...] slice).
void PoseDetector::runLandmarkBatch(
    const Onnx::ModelRole& role, PoseWorkflow draw, const Onnx::ImageView& src,
    const std::vector<Onnx::ROI::Rect>& rois)
{
  (void)draw;
  m_instances.clear();
  if(rois.empty())
    return;
  auto& lctx = *this->ctx;
  const auto& spec = lctx.readModelSpec();
  if(spec.inputs.empty())
    return;

  int mw = role.input_w > 0 ? role.input_w : 256;
  int mh = role.input_h > 0 ? role.input_h : 256;
  if(spec.inputs[0].shape.size() == 4)
  {
    // Dynamic dims are -1: only concrete spatial dims may override the role
    // defaults (otherwise the [N,C,H,W] resize below overflows).
    const auto& s = spec.inputs[0].shape;
    const int64_t sh = role.nhwc ? s[1] : s[2];
    const int64_t sw = role.nhwc ? s[2] : s[3];
    if(sh > 0)
      mh = static_cast<int>(sh);
    if(sw > 0)
      mw = static_cast<int>(sw);
  }

  const int N = static_cast<int>(rois.size());
  // readModelSpec() rewrites a dynamic batch dim to 1 in `shape` (so single-
  // image callers can feed it as-is): dynamic_batch is what says "any N". A
  // graph declaring a dynamic batch it can't actually run (a hard-coded
  // Reshape) throws once; it then stays per-crop for the model's lifetime.
  const int64_t batch_dim
      = (spec.inputs[0].shape.size() == 4) ? spec.inputs[0].shape[0] : 1;
  const bool dynamic = spec.inputs[0].dynamic_batch || batch_dim < 0;
  // On a GPU provider every change of the batch size re-plans the whole
  // graph (even back to a size seen before), so a changing person count would
  // stall the node. There the batch is padded to the largest count seen so far
  // (grow-only, at most Max Instances). On the CPU the cost is per crop: no
  // padding.
  const bool pad = m_landmark_pad_batch && dynamic && !m_landmark_no_batch;
  if(pad)
    m_landmark_batch_cap = std::clamp(std::max(m_landmark_batch_cap, N), 1, 16);
  const int NB = pad ? std::max(N, m_landmark_batch_cap) : N;
  const bool can_batch
      = NB >= 2 && !m_landmark_no_batch && (dynamic || batch_dim >= N);
  const float iw = src.w, ih = src.h;

  // Detector score per ROI (the caller fills m_roi_scores in parallel with
  // rois; 1 when it didn't): InstantHMR's joint confidence.
  auto roiScore = [&](size_t i) {
    return i < m_roi_scores.size() ? std::clamp(m_roi_scores[i], 0.f, 1.f) : 1.f;
  };
  const bool score_conf = role.kind == Onnx::ModelKind::InstantHmr;

  // Same mapping as landmarkKeypoints (keep the two in step).
  auto pushFromKps = [&](const std::vector<LandmarkKp>& kps,
                         const std::vector<LandmarkKp>& wkps,
                         const Onnx::Affine& M, float det_score) {
    if(kps.empty())
      return;
    DetectedPose pose;
    pose.keypoints.reserve(kps.size());
    float sum = 0.f;
    for(const auto& k : kps)
    {
      const Onnx::Vec2 p = Onnx::ROI::applyAffine(M, k.x, k.y);
      const float c = score_conf ? k.conf * det_score : k.conf;
      pose.keypoints.push_back({p.x / iw, p.y / ih, k.z, c});
      sum += c;
    }
    if(score_conf && sum <= 0.f)
      return; // every joint unusable: no pose
    // World coordinates are metric and crop-independent: no affine mapping.
    pose.world.reserve(wkps.size());
    for(const auto& k : wkps)
      pose.world.push_back(
          {k.x, k.y, k.z, score_conf ? k.conf * det_score : k.conf});
    pose.translation = m_trans_scratch;
    pose.body_params = m_params_scratch;
    pose.rig_offset = m_offset_scratch;
    pose.mean_confidence
        = score_conf ? det_score : sum / pose.keypoints.size();
    m_instances.push_back(std::move(pose));
  };

  // Fallback: one inference per ROI (fixed batch dim, or single instance).
  auto perCrop = [&] {
    m_instances.clear();
    for(size_t i = 0; i < rois.size(); ++i)
    {
      const Onnx::Affine M = Onnx::ROI::rectToAffine(rois[i], mw, mh);
      const float mc = landmarkKeypoints(
          role, src, M, m_kp_scratch, &m_world_scratch, /*crop_padded=*/true,
          roiScore(i));
      if(mc < 0.f || m_kp_scratch.empty())
        continue;
      DetectedPose pose;
      pose.keypoints = m_kp_scratch;
      pose.world = m_world_scratch;
      pose.translation = m_trans_scratch;
      pose.body_params = m_params_scratch;
      pose.rig_offset = m_offset_scratch;
      pose.mean_confidence = mc;
      m_instances.push_back(std::move(pose));
    }
  };
  if(!can_batch)
  {
    perCrop();
    return;
  }

  // --- Batched: pack N crops into one [N,C,H,W] input buffer. ---
  const int CHW = 3 * mw * mh;
  m_batch_storage.resize(
      static_cast<size_t>(NB) * CHW, boost::container::default_init);
  // Padding slots: zeros (decoded by nobody).
  std::fill(
      m_batch_storage.begin() + static_cast<size_t>(N) * CHW, m_batch_storage.end(),
      0.f);
  const NormSpec ns = landmarkNorm(role);
  const uint8_t* border = landmarkBorder(role);
  for(int b = 0; b < N; ++b)
  {
    const Onnx::Affine M = landmarkSampleAffine(
        role, Onnx::ROI::rectToAffine(rois[b], mw, mh));
    // Sample+normalize the crop straight into its [N,C,H,W] slice.
    Onnx::sampleAffineToTensor(
        ns.layout, src, M, mw, mh, ns.mean.data(), ns.invstd.data(),
        m_batch_storage.data() + static_cast<size_t>(b) * CHW,
        Onnx::prof::WarpCrop, border);
  }

  // [N,C,H,W] (or [N,H,W,C]) with every dynamic dim resolved to what was just
  // sampled, like finalizeTensor does for batch 1: a dynamic-H/W model would
  // otherwise get a negative element count and throw on every multi-person
  // frame.
  const bool nhwc = ns.layout == Onnx::TensorLayout::NhwcRgb;
  std::array<int64_t, 4> in_shape
      = nhwc ? std::array<int64_t, 4>{NB, mh, mw, 3}
             : std::array<int64_t, 4>{NB, 3, mh, mw};
  if(spec.inputs[0].shape.size() == 4)
    for(int d = 1; d < 4; ++d)
      if(spec.inputs[0].shape[d] > 0)
        in_shape[d] = spec.inputs[0].shape[d];

  Ort::Value outs[kMaxLandmarkOutputs]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}};
  const size_t n_out
      = std::min<size_t>(kMaxLandmarkOutputs, spec.output_names_char.size());
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  // Remember the model can't batch and redo this frame one crop at a time.
  auto giveUp = [&] {
    m_landmark_no_batch = true;
    perCrop();
  };

  // The batched inference, tensor creation included: whatever throws
  // (descriptor, shape, run) sends the model to the per-crop path for good.
  try
  {
    Ort::Value in0 = Ort::Value::CreateTensor<float>(
        mem_info, m_batch_storage.data(), m_batch_storage.size(),
        in_shape.data(), in_shape.size());
    if(isCliffInput(spec))
    {
      // InstantHMR's CLIFF condition, one row per crop, batched to [N,3] in a
      // reused member (no steady-state allocation).
      m_cliff.resize(static_cast<size_t>(NB) * 3);
      for(int b = 0; b < NB; ++b) // padding rows repeat the first crop's
        instantHmrCliff(
            rois[b < N ? b : 0], src.w, src.h, m_hmr_cliff_focal,
            m_cliff.data() + 3 * b);
      std::array<int64_t, 2> cshape{NB, 3};
      Ort::Value ins[2] = {
          std::move(in0), Ort::Value::CreateTensor<float>(
                              mem_info, m_cliff.data(), m_cliff.size(),
                              cshape.data(), cshape.size())};
      lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
    }
    else if(spec.inputs.size() >= 2)
    {
      // SimCC's second [w,h] input, batched to [N,2].
      m_bbox.resize(static_cast<size_t>(NB) * 2);
      for(int b = 0; b < NB; ++b)
      {
        m_bbox[2 * b] = mw;
        m_bbox[2 * b + 1] = mh;
      }
      std::array<int64_t, 2> bshape{NB, 2};
      Ort::Value ins[2] = {
          std::move(in0), Ort::Value::CreateTensor<int64_t>(
                              mem_info, m_bbox.data(), m_bbox.size(),
                              bshape.data(), bshape.size())};
      lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
    }
    else
    {
      Ort::Value ins[1] = {std::move(in0)};
      lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
    }
  }
  catch(const Ort::Exception&)
  {
    giveUp();
    return;
  }

  // How each output splits into per-instance [1,...] slices, resolved once
  // per frame (not per instance). The slices are non-owning views into the
  // batched outputs: no per-instance buffer, no copy.
  // Bytes per ONNX scalar element. 0 = a type we don't slice (the decoder's
  // own dtype guard then rejects the null slot).
  const auto elemSize = [](ONNXTensorElementDataType t) -> size_t {
    switch(t)
    {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return 4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return 2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return 8;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return 1;
      default:
        return 0;
    }
  };
  struct OutSlice
  {
    std::vector<int64_t> shape; // the slice's shape (leading dim 1 if batched)
    ONNXTensorElementDataType type{};
    std::uint8_t* data = nullptr;
    size_t per = 0;   // elements per slice
    size_t esz = 0;   // bytes per element (0: not sliced)
    bool batched = false;
  };
  static thread_local std::array<OutSlice, kMaxLandmarkOutputs> slices;
  for(size_t j = 0; j < n_out; ++j)
  {
    auto& o = slices[j];
    o.esz = 0;
    if(!outs[j] || !outs[j].IsTensor())
      continue;
    auto info = outs[j].GetTensorTypeAndShapeInfo();
    o.shape = info.GetShape();
    const size_t total = info.GetElementCount();
    // The model's DECLARED leading dim says whether this output follows the
    // batch: dynamic -> it must have resolved to N (and divide evenly); a
    // static N -> batched too; a static 1 with N >= 2 crops in means the graph
    // does not really batch (every instance would read slice 0); another
    // static value is a batch-independent constant, passed whole to every
    // instance (e.g. a [num_keypoints, ...] table).
    const int64_t decl0
        = (j < spec.outputs.size() && !spec.outputs[j].shape.empty())
              ? spec.outputs[j].shape[0]
              : -1;
    const bool follows = !o.shape.empty() && o.shape[0] == NB
                         && (total % static_cast<size_t>(NB)) == 0;
    if(!o.shape.empty() && (decl0 == 1 || (decl0 <= 0 && !follows)))
    {
      giveUp();
      return;
    }
    o.batched = follows && (decl0 <= 0 || decl0 == NB);
    o.per = o.batched ? total / static_cast<size_t>(NB) : total;
    if(o.batched)
      o.shape[0] = 1;
    // Keep the source element type: relabelling an int64/fp16 output as
    // float would read the wrong byte stride and defeat the decoders' dtype
    // guards.
    o.type = info.GetElementType();
    o.esz = elemSize(o.type);
    o.data = outs[j].GetTensorMutableData<std::uint8_t>();
  }

  // Decode each instance from its [1,...] slice of the batched outputs. The
  // decoders see exactly the single-instance shapes they already handle.
  static thread_local std::vector<LandmarkKp> kps; // reused decode scratch
  static thread_local std::vector<LandmarkKp> wkps;
  for(int b = 0; b < N; ++b)
  {
    Ort::Value sl[kMaxLandmarkOutputs]{
        Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
        Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}};
    for(size_t j = 0; j < n_out; ++j)
    {
      const auto& o = slices[j];
      if(o.esz == 0)
        continue; // leave slot null; decoder skips an unsupported dtype
      sl[j] = Ort::Value::CreateTensor(
          mem_info, o.data + (o.batched ? static_cast<size_t>(b) * o.per : 0) * o.esz,
          o.per * o.esz, o.shape.data(), o.shape.size(), o.type);
    }
    kps.clear();
    wkps.clear();
    decodeLandmark(
        role, spec, std::span<Ort::Value>(sl, n_out), mw, mh,
        static_cast<float>(inputs.min_confidence), kps, &wkps,
        /*crop_padded=*/true,
        HmrSink{&m_hmr_out, &m_trans_scratch, &m_params_scratch, &m_offset_scratch});
    pushFromKps(kps, wkps, Onnx::ROI::rectToAffine(rois[b], mw, mh), roiScore(b));
  }
}

void PoseDetector::runLandmark(
    const Onnx::ModelRole& role, PoseWorkflow draw, const Onnx::ImageView& src,
    const Onnx::Affine& M, int track_id, bool crop_padded)
{
  const float mean_conf = landmarkKeypoints(
      role, src, M, m_kp_scratch, &m_world_scratch, crop_padded,
      m_landmark_score);
  if(!finitef(mean_conf) || mean_conf < 0.f || m_kp_scratch.empty())
  {
    holdOrPassthrough(src);
    return;
  }

  DetectedPose detected;
  detected.keypoints = m_kp_scratch;
  detected.world = m_world_scratch; // metric coords: not affine-mapped
  detected.translation = m_trans_scratch;
  detected.body_params = m_params_scratch;
  detected.rig_offset = m_offset_scratch;
  detected.mean_confidence = mean_conf;
  detected.track_id = track_id; // set BEFORE draw so the id-color applies
  applySmoothing(detected, role.kind != Onnx::ModelKind::InstantHmr);
  fillBoxFromKeypoints(detected);
  outputs.detection.value = std::move(detected);
  // Snapshot the NATIVE keypoints before finalizeSingle() remaps them in place,
  // so the tracking-ROI loop (which indexes native joints) sees native data.
  m_native_keypoints = outputs.detection.value->keypoints;
  finalizeSingle(draw);
}

void PoseDetector::runYOLOPose(const Onnx::ImageView& src, const Onnx::Affine& M)
{
  auto& lctx = *this->ctx;
  const auto& spec = lctx.readModelSpec();

  int model_size = 640;
  if(!spec.inputs.empty() && spec.inputs[0].shape.size() == 4
     && spec.inputs[0].shape[2] > 0) // dynamic dims are -1: keep the default
    model_size = static_cast<int>(spec.inputs[0].shape[2]);

  // Cover-resize the whole frame through M (model px -> image px), the same
  // affine the keypoint mapback below uses, so input and output geometry agree.
  auto t = fusedAffineTensor(
      spec.inputs[0], src, M, model_size, model_size,
      normMeanStd(Onnx::TensorLayout::NchwRgb, {0.f, 0.f, 0.f}, {255.f, 255.f, 255.f}),
      storage, Onnx::prof::WarpDet);

  Ort::Value ins[1] = {std::move(t.value)};
  Ort::Value outs[1]{Ort::Value{nullptr}};
  lctx.infer(spec, ins, outs);

  static const Yolo::YOLO_pose yolo_pose;
  std::vector<Yolo::YOLO_pose::pose_type> poses;
  // Floor the threshold like runRTMO does: at ~0 every one of the 8400 grid
  // candidates survives and the helper's O(n^2) dedup blows up the frame time.
  // K from the output shape (17 for COCO; a hand head has 21).
  const int K = yolo_pose.processOutput(
      spec, outs, poses, 100,
      std::max(0.3f, static_cast<float>(inputs.min_confidence)), 0, 0,
      model_size, model_size, model_size, model_size);

  if(poses.empty())
  {
    holdOrPassthrough(src);
    std::swap(storage, t.storage);
    return;
  }

  const float iw = src.w, ih = src.h;

  // Multi-instance: YOLO-pose is single-stage but already finds every person.
  if(inputs.track_ids.value)
  {
    const int max_inst = std::clamp(
        static_cast<int>(
            inputs.max_instances.value),
        1, 16);
    if(static_cast<int>(poses.size()) > max_inst)
      std::partial_sort(
          poses.begin(), poses.begin() + max_inst, poses.end(),
          [](const auto& a, const auto& b) { return a.confidence > b.confidence; });
    const int np = std::min(static_cast<int>(poses.size()), max_inst);
    m_instances.clear();
    for(int pi = 0; pi < np; ++pi)
    {
      const auto& pp = poses[pi];
      DetectedPose dp;
      dp.keypoints.assign(K, PoseKeypoint{0.f, 0.f, 0.f, 0.f});
      for(const auto& kp : pp.keypoints)
        if(kp.kp >= 0 && kp.kp < K)
        {
          const Onnx::Vec2 p = Onnx::ROI::applyAffine(M, kp.x, kp.y);
          dp.keypoints[kp.kp] = {p.x / iw, p.y / ih, 0.0f, 1.0f};
        }
      dp.mean_confidence = pp.confidence;
      m_instances.push_back(std::move(dp));
    }
    std::swap(storage, t.storage);
    if(m_instances.empty())
    {
      coastOrPassthrough(PoseWorkflow::YOLOPose, src);
      return;
    }
    emitInstances(PoseWorkflow::YOLOPose, /*do_track=*/true);
    return;
  }

  const auto& pose = poses[0];
  DetectedPose detected;
  detected.keypoints.assign(K, PoseKeypoint{0.f, 0.f, 0.f, 0.f});
  for(const auto& kp : pose.keypoints)
  {
    if(kp.kp >= 0 && kp.kp < K)
    {
      const Onnx::Vec2 p = Onnx::ROI::applyAffine(M, kp.x, kp.y);
      detected.keypoints[kp.kp] = {p.x / iw, p.y / ih, 0.0f, 1.0f};
    }
  }
  detected.mean_confidence = pose.confidence;
  applySmoothing(detected);
  fillBoxFromKeypoints(detected);
  outputs.detection.value = std::move(detected);
  finalizeSingle(PoseWorkflow::YOLOPose);

  std::swap(storage, t.storage);
}

void PoseDetector::runRTMO(const Onnx::ImageView& src)
{
  auto& lctx = *this->ctx;
  const auto& spec = lctx.readModelSpec();
  if(spec.inputs.empty())
  {
    passthrough(src);
    return;
  }

  // NCHW input; RTMO is 640x640.
  int model = 640;
  if(spec.inputs[0].shape.size() == 4 && spec.inputs[0].shape[3] > 0)
    model = static_cast<int>(spec.inputs[0].shape[3]);

  Onnx::LetterboxInfo lb;
  Ort::Value input_value{nullptr};
  {
    // RTMO: raw BGR, no normalization (like YOLOX).
    auto t = fusedLetterboxTensor(
        spec.inputs[0], src, model, model, /*center=*/false,
        normMeanStd(Onnx::TensorLayout::NchwBgr, {0, 0, 0}, {1, 1, 1}), storage,
        lb);
    input_value = std::move(t.value);
    std::swap(storage, t.storage);
  }

  Ort::Value ins[1] = {std::move(input_value)};
  Ort::Value outs[4]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}};
  const size_t n_out = std::min<size_t>(4, spec.output_names_char.size());
  lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));

  // dets [1,N,5] (xyxy+score), keypoints [1,N,K,3] (x,y,score) in model px.
  const float* dets = nullptr;
  const float* kpt = nullptr;
  int n_det = 0, n_kpt = 0, K = 0;
  for(size_t i = 0; i < n_out; ++i)
  {
    if(!outs[i].IsTensor())
      continue;
    auto info = outs[i].GetTensorTypeAndShapeInfo();
    if(info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      continue; // fp16/u8 buffers read as float would over-read
    auto sh = info.GetShape();
    if(sh.size() == 3 && sh[2] == 5)
    {
      dets = outs[i].GetTensorData<float>();
      n_det = static_cast<int>(
          std::min<int64_t>(sh[1], info.GetElementCount() / 5));
    }
    else if(sh.size() == 4 && sh[3] == 3)
    {
      kpt = outs[i].GetTensorData<float>();
      K = static_cast<int>(sh[2]);
      if(K > 0)
        n_kpt = static_cast<int>(std::min<int64_t>(
            sh[1], info.GetElementCount() / (static_cast<int64_t>(K) * 3)));
    }
  }
  // The two tensors share the person axis but each carries its own row count:
  // index only rows that exist in BOTH, otherwise a mismatched export would
  // read past the smaller buffer.
  const int N = std::min(n_det, n_kpt);
  if(!dets || !kpt || K == 0 || N <= 0)
  {
    holdOrPassthrough(src);
    return;
  }

  // Multi-instance: RTMO is NMS-free and returns every person already.
  if(inputs.track_ids.value)
  {
    const float iw = src.w, ih = src.h;
    const float thr = std::max(0.3f, static_cast<float>(inputs.min_confidence));
    const int max_inst = std::clamp(
        static_cast<int>(
            inputs.max_instances.value),
        1, 16);
    auto& sel = m_box_sel; // reused member: no per-frame allocation
    sel.clear();
    sel.reserve(N);
    for(int i = 0; i < N; ++i)
      if(dets[i * 5 + 4] > thr)
        sel.push_back({dets[i * 5 + 4], i});
    if(static_cast<int>(sel.size()) > max_inst)
      std::partial_sort(
          sel.begin(), sel.begin() + max_inst, sel.end(),
          [](const auto& a, const auto& b) { return a.first > b.first; });
    const int ns = std::min(static_cast<int>(sel.size()), max_inst);
    m_instances.clear();
    for(int si = 0; si < ns; ++si)
    {
      const int idx = sel[si].second;
      DetectedPose dp;
      dp.keypoints.reserve(K);
      for(int k = 0; k < K; ++k)
      {
        const float* p = kpt + (static_cast<size_t>(idx) * K + k) * 3;
        dp.keypoints.push_back(
            {((p[0] - lb.pad_x) / lb.scale) / iw,
             ((p[1] - lb.pad_y) / lb.scale) / ih, 0.0f, p[2]});
      }
      dp.mean_confidence = sel[si].first;
      m_instances.push_back(std::move(dp));
    }
    if(m_instances.empty())
    {
      coastOrPassthrough(PoseWorkflow::YOLOPose, src);
      return;
    }
    emitInstances(PoseWorkflow::YOLOPose, /*do_track=*/true);
    return;
  }

  int best = -1;
  float bestc = std::max(0.3f, static_cast<float>(inputs.min_confidence));
  for(int i = 0; i < N; ++i)
    if(dets[i * 5 + 4] > bestc)
    {
      bestc = dets[i * 5 + 4];
      best = i;
    }
  if(best < 0)
  {
    holdOrPassthrough(src);
    return;
  }

  const float iw = src.w, ih = src.h;
  DetectedPose detected;
  detected.keypoints.reserve(K);
  for(int k = 0; k < K; ++k)
  {
    const float* p = kpt + (static_cast<size_t>(best) * K + k) * 3;
    detected.keypoints.push_back(
        {((p[0] - lb.pad_x) / lb.scale) / iw,
         ((p[1] - lb.pad_y) / lb.scale) / ih, 0.0f, p[2]});
  }
  detected.mean_confidence = bestc;
  applySmoothing(detected);
  fillBoxFromKeypoints(detected);
  outputs.detection.value = std::move(detected);
  finalizeSingle(PoseWorkflow::YOLOPose);
}

} // namespace OnnxModels
