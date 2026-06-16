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

// Decode ONE instance's landmark outputs (a [1,...] outspan) into MODEL-PIXEL
// keypoints. Shared by the single-crop and batched-slice paths. `world` (when
// non-null) receives the model's world-space 3D keypoints in METERS
// (hip-origin) for models that emit them — BlazePose full-body's [1,117]
// (39*3) world output. World coordinates are crop-independent: they bypass the
// affine un-mapping the screen keypoints go through.
void decodeLandmark(
    const Onnx::ModelRole& role, const Onnx::ModelSpec& spec,
    std::span<Ort::Value> outspan, int mw, int mh, float min_conf,
    std::vector<LandmarkKp>& kps, std::vector<LandmarkKp>* world = nullptr)
{
  kps.clear();
  if(world)
    world->clear();
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
}
} // namespace

float PoseDetector::landmarkKeypoints(
    const Onnx::ModelRole& role, const Onnx::ImageView& src,
    const Onnx::Affine& M, std::vector<PoseKeypoint>& out,
    std::vector<PoseKeypoint>* out_world)
{
  out.clear();
  if(out_world)
    out_world->clear();
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
      spec.inputs[0], src, M, mw, mh, landmarkNorm(role), storage);

  Ort::Value outs[5]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}};
  const size_t n_out = std::min<size_t>(5, spec.output_names_char.size());

  // Some RTMPose exports take a second [w,h] bbox input.
  std::array<int64_t, 2> bbox_wh{mw, mh};
  std::array<int64_t, 2> bbox_shape{1, 2};
  if(spec.inputs.size() >= 2)
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

  std::vector<LandmarkKp> kps;
  std::vector<LandmarkKp> wkps;
  decodeLandmark(
      role, spec, outspan, mw, mh, static_cast<float>(inputs.min_confidence),
      kps, out_world ? &wkps : nullptr);

  std::swap(storage, t.storage);

  if(kps.empty())
    return -1.f;

  // Map model-pixel keypoints back through M -> image-normalized [0,1].
  const float iw = src.w, ih = src.h;
  out.reserve(kps.size());
  float sum_conf = 0.0f;
  for(const auto& k : kps)
  {
    const Onnx::Vec2 p = Onnx::ROI::applyAffine(M, k.x, k.y);
    out.push_back({p.x / iw, p.y / ih, k.z, k.conf});
    sum_conf += k.conf;
  }
  // World coordinates are metric and crop-independent: no affine mapping.
  if(out_world)
  {
    out_world->reserve(wkps.size());
    for(const auto& k : wkps)
      out_world->push_back({k.x, k.y, k.z, k.conf});
  }
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
  const int64_t batch_dim
      = (spec.inputs[0].shape.size() == 4) ? spec.inputs[0].shape[0] : 1;
  const bool can_batch = N >= 2 && (batch_dim < 0 || batch_dim >= N);
  const float iw = src.w, ih = src.h;

  auto pushFromKps = [&](const std::vector<LandmarkKp>& kps,
                         const std::vector<LandmarkKp>& wkps,
                         const Onnx::Affine& M) {
    if(kps.empty())
      return;
    DetectedPose pose;
    pose.keypoints.reserve(kps.size());
    float sum = 0.f;
    for(const auto& k : kps)
    {
      const Onnx::Vec2 p = Onnx::ROI::applyAffine(M, k.x, k.y);
      pose.keypoints.push_back({p.x / iw, p.y / ih, k.z, k.conf});
      sum += k.conf;
    }
    // World coordinates are metric and crop-independent: no affine mapping.
    pose.world.reserve(wkps.size());
    for(const auto& k : wkps)
      pose.world.push_back({k.x, k.y, k.z, k.conf});
    pose.mean_confidence = sum / pose.keypoints.size();
    m_instances.push_back(std::move(pose));
  };

  // Fallback: one inference per ROI (fixed batch dim, or single instance).
  if(!can_batch)
  {
    for(const auto& r : rois)
    {
      const Onnx::Affine M = Onnx::ROI::rectToAffine(r, mw, mh);
      const float mc
          = landmarkKeypoints(role, src, M, m_kp_scratch, &m_world_scratch);
      if(mc < 0.f || m_kp_scratch.empty())
        continue;
      DetectedPose pose;
      pose.keypoints = m_kp_scratch;
      pose.world = m_world_scratch;
      pose.mean_confidence = mc;
      m_instances.push_back(std::move(pose));
    }
    return;
  }

  // --- Batched: pack N crops into one [N,C,H,W] input buffer. ---
  const int CHW = 3 * mw * mh;
  m_batch_storage.resize(
      static_cast<size_t>(N) * CHW, boost::container::default_init);
  const NormSpec ns = landmarkNorm(role);
  for(int b = 0; b < N; ++b)
  {
    const Onnx::Affine M = Onnx::ROI::rectToAffine(rois[b], mw, mh);
    // Sample+normalize the crop straight into its [N,C,H,W] slice.
    Onnx::sampleAffineToTensor(
        ns.layout, src, M, mw, mh, ns.mean.data(), ns.invstd.data(),
        m_batch_storage.data() + static_cast<size_t>(b) * CHW);
  }

  std::vector<int64_t> in_shape = spec.inputs[0].shape;
  if(in_shape.size() == 4)
    in_shape[0] = N;
  else
    in_shape = {N, 3, mh, mw};
  Ort::Value in0 = Onnx::vec_to_tensor<float>(m_batch_storage, in_shape);

  Ort::Value outs[5]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}};
  const size_t n_out = std::min<size_t>(5, spec.output_names_char.size());

  if(spec.inputs.size() >= 2)
  {
    // SimCC's second [w,h] input, batched to [N,2].
    m_bbox.resize(static_cast<size_t>(N) * 2);
    for(int b = 0; b < N; ++b)
    {
      m_bbox[2 * b] = mw;
      m_bbox[2 * b + 1] = mh;
    }
    std::array<int64_t, 2> bshape{N, 2};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto bbox_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, m_bbox.data(), m_bbox.size(), bshape.data(), bshape.size());
    Ort::Value ins[2] = {std::move(in0), std::move(bbox_tensor)};
    lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
  }
  else
  {
    Ort::Value ins[1] = {std::move(in0)};
    lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
  }

  // Decode each instance from its [1,...] slice of the batched outputs. The
  // decoders see exactly the single-instance shapes they already handle.
  Ort::AllocatorWithDefaultOptions alloc;
  std::vector<LandmarkKp> kps;
  std::vector<LandmarkKp> wkps;
  // Bytes per ONNX scalar element, for the type-preserving slice below. 0 = a
  // type we don't slice (the decoder's own dtype guard then rejects the slot).
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
  for(int b = 0; b < N; ++b)
  {
    Ort::Value sl[5]{
        Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
        Ort::Value{nullptr}, Ort::Value{nullptr}};
    for(size_t j = 0; j < n_out; ++j)
    {
      auto info = outs[j].GetTensorTypeAndShapeInfo();
      auto shp = info.GetShape();
      const size_t total = info.GetElementCount();
      // Treat as batched only if the model's DECLARED leading dim is the batch
      // axis (dynamic, or == N), it resolved to N, and the buffer divides evenly.
      // This avoids mis-slicing an output whose fixed leading dim coincidentally
      // equals the instance count (e.g. a constant [num_keypoints, ...]).
      const int64_t decl0
          = (j < spec.outputs.size() && !spec.outputs[j].shape.empty())
                ? spec.outputs[j].shape[0]
                : -1;
      const bool batched = !shp.empty() && shp[0] == N && (decl0 <= 0 || decl0 == N)
                           && (total % static_cast<size_t>(N)) == 0;
      const size_t per = batched ? total / static_cast<size_t>(N) : total;
      if(batched)
        shp[0] = 1;
      // Preserve the source element type: re-creating every slice as float (as a
      // prior version did) would relabel an int64/fp16 output as float and
      // memcpy the wrong byte stride, silently defeating the decoders' dtype
      // guards. Copy raw bytes at the true element size instead.
      const ONNXTensorElementDataType et = info.GetElementType();
      const size_t esz = elemSize(et);
      if(esz == 0)
        continue; // leave slot null; decoder skips an unsupported dtype
      sl[j] = Ort::Value::CreateTensor(alloc, shp.data(), shp.size(), et);
      std::memcpy(
          sl[j].GetTensorMutableData<std::uint8_t>(),
          outs[j].GetTensorData<std::uint8_t>()
              + (batched ? static_cast<size_t>(b) * per : 0) * esz,
          per * esz);
    }
    kps.clear();
    wkps.clear();
    decodeLandmark(
        role, spec, std::span<Ort::Value>(sl, n_out), mw, mh,
        static_cast<float>(inputs.min_confidence), kps, &wkps);
    pushFromKps(kps, wkps, Onnx::ROI::rectToAffine(rois[b], mw, mh));
  }
}

void PoseDetector::runLandmark(
    const Onnx::ModelRole& role, PoseWorkflow draw, const Onnx::ImageView& src,
    const Onnx::Affine& M, int track_id)
{
  const float mean_conf
      = landmarkKeypoints(role, src, M, m_kp_scratch, &m_world_scratch);
  if(!finitef(mean_conf) || mean_conf < 0.f || m_kp_scratch.empty())
  {
    passthrough(src);
    return;
  }

  DetectedPose detected;
  detected.keypoints = m_kp_scratch;
  detected.world = m_world_scratch; // metric coords: not smoothed/affine-mapped
  detected.mean_confidence = mean_conf;
  detected.track_id = track_id; // set BEFORE draw so the id-color applies
  applySmoothing(detected);
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
  yolo_pose.processOutput(
      spec, outs, poses, 100,
      std::max(0.3f, static_cast<float>(inputs.min_confidence)), 0, 0,
      model_size, model_size, model_size, model_size);

  if(poses.empty())
  {
    passthrough(src);
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
      dp.keypoints.assign(17, PoseKeypoint{0.f, 0.f, 0.f, 0.f});
      for(const auto& kp : pp.keypoints)
        if(kp.kp >= 0 && kp.kp < 17)
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
  detected.keypoints.assign(17, PoseKeypoint{0.f, 0.f, 0.f, 0.f});
  for(const auto& kp : pose.keypoints)
  {
    if(kp.kp >= 0 && kp.kp < 17)
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
    passthrough(src);
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
    passthrough(src);
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
