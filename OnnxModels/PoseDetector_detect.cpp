#include "PoseDetector_internal.hpp"

namespace OnnxModels
{

// ASCII case-insensitive substring test (model I/O names are ASCII).
static bool icontains(std::string_view hay, std::string_view needle) noexcept
{
  auto eq = [](char a, char b) {
    auto lower = [](char c) { return (c >= 'A' && c <= 'Z') ? char(c + 32) : c; };
    return lower(a) == lower(b);
  };
  return std::search(hay.begin(), hay.end(), needle.begin(), needle.end(), eq)
         != hay.end();
}

// ROI rect (image px) from a detection, mirroring the per-kind dialect choice.
Onnx::ROI::Rect PoseDetector::detectionRect(
    const Onnx::ModelRole& role, const Onnx::Detection::Detection& det, int W,
    int H)
{
  const int mw = role.input_w > 0 ? role.input_w : 256;
  const int mh = role.input_h > 0 ? role.input_h : 256;
  switch(role.kind)
  {
    case Onnx::ModelKind::BlazePoseLandmark:
      return Onnx::ROI::mediapipeRect(det, W, H, Onnx::ROI::poseParams());
    case Onnx::ModelKind::HandLandmark:
      return Onnx::ROI::mediapipeRect(det, W, H, Onnx::ROI::handParams());
    case Onnx::ModelKind::FaceMeshLandmark:
      return Onnx::ROI::mediapipeRect(det, W, H, Onnx::ROI::faceParams());
    case Onnx::ModelKind::MobileFaceNet:
      return Onnx::ROI::mediapipeRect(det, W, H, Onnx::ROI::mobileFaceParams());
    default:
    {
      const Onnx::Rect box{
          det.box().x * W, det.box().y * H, det.w * W, det.h * H};
      // A top-down HAND pose (e.g. RTMPose-hand) behind a PALM detector only
      // gets a palm-sized box centered on the palm; the fingers extend well
      // beyond it. A symmetric expansion must be large enough to reach the
      // fingertips in every direction (MediaPipe uses ~2.6x AND shifts toward
      // the fingers; we have no orientation here, so go a bit larger to be safe).
      const float expand = (role.domain == Onnx::ModelDomain::Hand) ? 3.2f : 1.25f;
      return Onnx::ROI::topdownRect(box, mw, mh, expand);
    }
  }
}

std::vector<Onnx::Detection::Detection>
PoseDetector::runDetector(
    const Onnx::ModelRole& role, const Onnx::ImageView& src, Onnx::ModelDomain target,
    int keep_class, Onnx::OnnxRunContext* ctx_override)
{
  Onnx::OnnxRunContext* dctx_ptr
      = ctx_override ? ctx_override : this->det_ctx.get();
  if(!dctx_ptr)
    return {};

  auto& dctx = *dctx_ptr;
  const auto& spec = dctx.readModelSpec();
  if(spec.inputs.empty() || spec.inputs[0].shape.size() != 4)
    return {};

  const int model = role.input_w > 0 ? role.input_w : 128;

  // The model's declared anchor count (from the box output's second dim).
  int num_boxes = 0;
  for(const auto& o : spec.outputs)
    if(o.shape.size() == 3
       && (o.shape.back() == 12 || o.shape.back() == 16 || o.shape.back() == 18))
      num_boxes = static_cast<int>(o.shape[1]);

  // Map detections from the letterboxed model square back to image-normalized
  // [0,1] coordinates (works for both centered and top-left letterboxes).
  const float iw = src.w, ih = src.h;
  auto removeLetterbox
      = [&](std::vector<Onnx::Detection::Detection>& dets,
            const Onnx::LetterboxInfo& lb) {
          auto fix = [&](float nx, float ny, float& ox, float& oy) {
            ox = ((nx * model - lb.pad_x) / lb.scale) / iw;
            oy = ((ny * model - lb.pad_y) / lb.scale) / ih;
          };
          for(auto& d : dets)
          {
            fix(d.xc, d.yc, d.xc, d.yc);
            d.w = (d.w * model / lb.scale) / iw;
            d.h = (d.h * model / lb.scale) / ih;
            for(auto& k : d.keypoints)
            {
              float ox, oy;
              fix(k.x, k.y, ox, oy);
              k = {ox, oy};
            }
          }
        };

  // --- End2end person/hand detector (YOLOX / RTMDet): BGR NCHW, top-left
  // letterbox, dets+labels output. ---
  if(role.kind == Onnx::ModelKind::PersonDetector)
  {
    // Heuristic: small input (<=320) -> RTMDet (mean/std); else YOLOX (raw).
    std::array<float, 3> mean_bgr{0, 0, 0}, std_bgr{1, 1, 1};
    if(model <= 320)
    {
      mean_bgr = {103.53f, 116.28f, 123.675f};
      std_bgr = {57.375f, 57.12f, 58.395f};
    }
    Onnx::LetterboxInfo lb;
    Ort::Value input_value{nullptr};
    {
      auto t = fusedLetterboxTensor(
          spec.inputs[0], src, model, model, /*center=*/false,
          normMeanStd(Onnx::TensorLayout::NchwBgr, mean_bgr, std_bgr),
          det_storage, lb);
      input_value = std::move(t.value);
      std::swap(det_storage, t.storage);
    }
    Ort::Value ins[1] = {std::move(input_value)};
    Ort::Value outs[4]{
        Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
        Ort::Value{nullptr}};
    const size_t n_out = std::min<size_t>(4, spec.output_names_char.size());
    dctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));

    // -2 (domain default) -> person(0); else the requested class (-1 = all).
    const int keep = (keep_class == -2) ? 0 : keep_class;
    auto dets = Onnx::Detection::decodeEnd2End(
        std::span<Ort::Value>(outs, n_out), model, keep, 0.3f);
    removeLetterbox(dets, lb);
    return dets;
  }

  // --- PINTO multi-class detector ([N,7] batchno,classid,score,xyxy): raw BGR,
  // top-left letterbox. Used as a body/person detector (class 0). ---
  if(role.kind == Onnx::ModelKind::MultiClassDetector)
  {
    const int mw = model;
    const int mh = role.input_h > 0 ? role.input_h : model;
    Onnx::LetterboxInfo lb;
    Ort::Value input_value{nullptr};
    {
      auto t = fusedLetterboxTensor(
          spec.inputs[0], src, mw, mh, /*center=*/false,
          normMeanStd(Onnx::TensorLayout::NchwBgr, {0, 0, 0}, {1, 1, 1}),
          det_storage, lb);
      input_value = std::move(t.value);
      std::swap(det_storage, t.storage);
    }
    Ort::Value ins[1] = {std::move(input_value)};
    Ort::Value outs[2]{Ort::Value{nullptr}, Ort::Value{nullptr}};
    const size_t n_out = std::min<size_t>(2, spec.output_names_char.size());
    dctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));

    const bool sbb = !spec.outputs.empty()
                     && icontains(spec.outputs[0].name, "score_x"); // score before box
    const int keep = (keep_class == -2) ? 0 : keep_class;
    auto dets = Onnx::Detection::decodeMultiClass(
        std::span<Ort::Value>(outs, n_out), mw, mh, keep, 0.4f, sbb);
    // remove top-left letterbox (non-square aware)
    for(auto& dd : dets)
    {
      auto fix = [&](float nx, float ny, float& ox, float& oy) {
        ox = ((nx * mw - lb.pad_x) / lb.scale) / iw;
        oy = ((ny * mh - lb.pad_y) / lb.scale) / ih;
      };
      fix(dd.xc, dd.yc, dd.xc, dd.yc);
      dd.w = (dd.w * mw / lb.scale) / iw;
      dd.h = (dd.h * mh / lb.scale) / ih;
    }
    return dets;
  }

  // --- Raw YOLOX COCO grid: BGR no-norm, top-left letterbox, class filter.
  // person=0 for body; COCO animals 14..23 for animal pose. ---
  if(role.kind == Onnx::ModelKind::YoloxDetector)
  {
    const int mw = model;
    const int mh = role.input_h > 0 ? role.input_h : model;
    int cls_lo = 0, cls_hi = 0;            // person
    if(target == Onnx::ModelDomain::Animal) { cls_lo = 14; cls_hi = 23; }
    if(keep_class == -1) { cls_lo = 0; cls_hi = 100000; }      // all classes
    else if(keep_class >= 0) { cls_lo = cls_hi = keep_class; } // one class
    Onnx::LetterboxInfo lb;
    Ort::Value input_value{nullptr};
    {
      // This PINTO YOLOX-COCO export wants BGR [0,1] (raw 0-255 gives garbage).
      auto t = fusedLetterboxTensor(
          spec.inputs[0], src, mw, mh, /*center=*/false,
          normMeanStd(Onnx::TensorLayout::NchwBgr, {0, 0, 0}, {255.f, 255.f, 255.f}),
          det_storage, lb);
      input_value = std::move(t.value);
      std::swap(det_storage, t.storage);
    }
    Ort::Value ins[1] = {std::move(input_value)};
    Ort::Value outs[2]{Ort::Value{nullptr}, Ort::Value{nullptr}};
    const size_t n_out = std::min<size_t>(2, spec.output_names_char.size());
    dctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
    auto dets = Onnx::Detection::decodeYoloxGrid(
        std::span<Ort::Value>(outs, n_out), mw, mh, cls_lo, cls_hi, 0.3f, 0.45f);
    for(auto& dd : dets)
    {
      auto fix = [&](float nx, float ny, float& ox, float& oy) {
        ox = ((nx * mw - lb.pad_x) / lb.scale) / iw;
        oy = ((ny * mh - lb.pad_y) / lb.scale) / ih;
      };
      fix(dd.xc, dd.yc, dd.xc, dd.yc);
      dd.w = (dd.w * mw / lb.scale) / iw;
      dd.h = (dd.h * mh / lb.scale) / ih;
    }
    return dets;
  }

  // --- PriorBox face detectors (RetinaFace / FaceBoxes): BGR - mean, NCHW,
  // variance-encoded SSD priors (decode validated against real exports +
  // PINTO faceboxes_post ground truth). ---
  if(role.kind == Onnx::ModelKind::RetinaFaceDetector
     || role.kind == Onnx::ModelKind::FaceBoxesDetector)
  {
    // Both families export dynamic input sizes; 320 is the validated default.
    const int mw = role.input_w > 0 ? role.input_w : 320;
    const int mh = role.input_h > 0 ? role.input_h : 320;
    Onnx::LetterboxInfo lb;
    Ort::Value input_value{nullptr};
    {
      auto t = fusedLetterboxTensor(
          spec.inputs[0], src, mw, mh, /*center=*/false,
          normMeanStd(
              Onnx::TensorLayout::NchwBgr, {104.f, 117.f, 123.f}, {1.f, 1.f, 1.f}),
          det_storage, lb);
      input_value = std::move(t.value);
      std::swap(det_storage, t.storage);
    }
    Ort::Value ins[1] = {std::move(input_value)};
    Ort::Value outs[3]{Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}};
    const size_t n_out = std::min<size_t>(3, spec.output_names_char.size());
    dctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));

    std::vector<Onnx::Detection::Detection> dets;
    // PINTO pre-decoded FaceBoxes: pick the boxes output by name.
    int predecoded_idx = -1;
    for(size_t i = 0; i < spec.outputs.size(); ++i)
      if(icontains(spec.outputs[i].name, "x1y1x2y2"))
      {
        predecoded_idx = static_cast<int>(i);
        break;
      }
    if(predecoded_idx >= 0)
      dets = Onnx::Detection::decodePreDecodedBoxes(
          std::span<Ort::Value>(outs, n_out), predecoded_idx, 0.3f);
    else
    {
      // Prior geometry only depends on (family, input size): rebuild on model
      // change, not per frame (4200 priors @320 RetinaFace).
      if(m_prior_kind != role.kind || m_prior_w != mw || m_prior_h != mh)
      {
        m_prior_cache = Onnx::Detection::generatePriorBoxes(
            role.kind == Onnx::ModelKind::RetinaFaceDetector
                ? Onnx::Detection::retinaFacePriorParams()
                : Onnx::Detection::faceBoxesPriorParams(),
            mw, mh);
        m_prior_kind = role.kind;
        m_prior_w = mw;
        m_prior_h = mh;
      }
      // Both families use the canonical SSD variances 0.1/0.2.
      dets = Onnx::Detection::decodePriorBox(
          std::span<Ort::Value>(outs, n_out), m_prior_cache, 0.1f, 0.2f, 0.4f);
    }
    dets = Onnx::Detection::nms(std::move(dets), 0.4f);
    // remove top-left letterbox (non-square aware)
    for(auto& dd : dets)
    {
      auto fix = [&](float nx, float ny, float& ox, float& oy) {
        ox = ((nx * mw - lb.pad_x) / lb.scale) / iw;
        oy = ((ny * mh - lb.pad_y) / lb.scale) / ih;
      };
      fix(dd.xc, dd.yc, dd.xc, dd.yc);
      dd.w = (dd.w * mw / lb.scale) / iw;
      dd.h = (dd.h * mh / lb.scale) / ih;
      for(auto& k : dd.keypoints)
      {
        float ox, oy;
        fix(k.x, k.y, ox, oy);
        k = {ox, oy};
      }
    }
    return dets;
  }

  // --- SSD-anchor detectors (BlazePose / palm / BlazeFace) ---
  if(role.kind != Onnx::ModelKind::BlazePoseDetector
     && role.kind != Onnx::ModelKind::PalmDetector
     && role.kind != Onnx::ModelKind::BlazeFaceDetector)
    return {};

  // Pixel mapping out = px*a + b: palm wants [0,1], the others [-1,1].
  const bool unit_range = role.kind == Onnx::ModelKind::PalmDetector;
  const float a = unit_range ? 1.f : 2.f;
  const float b = unit_range ? 0.f : -1.f;

  // Anchor geometry only depends on (family, input size, declared box count):
  // pick the candidate config whose anchor count matches the model (counted in
  // closed form) and (re)generate the anchors only when that key changes —
  // regenerating up to 2254 anchors per frame would allocate at frame rate.
  if(m_ssd_kind != role.kind || m_ssd_input != model
     || m_ssd_num_boxes != num_boxes)
  {
    std::vector<Onnx::Detection::SsdParams> candidates;
    switch(role.kind)
    {
      case Onnx::ModelKind::BlazePoseDetector:
        candidates = {
            Onnx::Detection::blazePoseParams(model),
            Onnx::Detection::blazePoseParams(128),
            Onnx::Detection::blazePoseParams(224)};
        break;
      case Onnx::ModelKind::PalmDetector:
        candidates = {Onnx::Detection::palmParams(model)};
        break;
      default:
        candidates = {Onnx::Detection::blazeFaceParams(model)};
        break;
    }
    Onnx::Detection::SsdParams params = candidates.front();
    for(auto& c : candidates)
      if(num_boxes > 0 && Onnx::Detection::anchorCount(c) == num_boxes)
      {
        params = c;
        break;
      }
    params.input_size = model;
    m_ssd_anchors = Onnx::Detection::generateAnchors(params);
    m_ssd_kind = role.kind;
    m_ssd_input = model;
    m_ssd_num_boxes = num_boxes;
  }

  Onnx::LetterboxInfo lb;
  Ort::Value input_value{nullptr};
  {
    // Detectors are usually NHWC, but some PINTO exports are NCHW. Both want
    // out = (px/255)*a + b; only the layout differs.
    const auto layout
        = role.nhwc ? Onnx::TensorLayout::NhwcRgb : Onnx::TensorLayout::NchwRgb;
    auto t = fusedLetterboxTensor(
        spec.inputs[0], src, model, model, /*center=*/true, normAB(layout, a, b),
        det_storage, lb);
    input_value = std::move(t.value);
    std::swap(det_storage, t.storage);
  }

  Ort::Value ins[1] = {std::move(input_value)};
  Ort::Value outs[6]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}};
  const size_t n_out = std::min<size_t>(6, spec.output_names_char.size());
  dctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));

  auto dets = Onnx::Detection::decode(
      std::span<Ort::Value>(outs, n_out), m_ssd_anchors, model, 0.5f);
  dets = Onnx::Detection::nms(std::move(dets), 0.3f);
  removeLetterbox(dets, lb);
  return dets;
}

void PoseDetector::runDetectorAsPose(
    const Onnx::ModelRole& role, const Onnx::ImageView& src)
{
  // The detector sits in the LANDMARK port here (standalone, e.g. RetinaFace
  // drawn as a 5-keypoint face pose), so run it on `ctx`, not `det_ctx`.
  auto dets = runDetector(role, src, Onnx::ModelDomain::Body, -2, this->ctx.get());
  if(dets.empty())
  {
    passthrough(src);
    return;
  }

  const auto& d = dets.front();
  if(ossia::safe_isnan(d.score))
  {
    passthrough(src);
    return;
  }

  DetectedPose detected;
  detected.keypoints.reserve(d.keypoints.size());
  for(const auto& k : d.keypoints)
    detected.keypoints.push_back({k.x, k.y, 0.0f, d.score});
  detected.mean_confidence = d.score;
  detected.class_id = d.class_id;
  detected.box = {d.box().x, d.box().y, d.w, d.h};
  applySmoothing(detected);
  fillBoxFromKeypoints(detected); // no-op if box already set above
  outputs.detection.value = std::move(detected);

  const PoseWorkflow draw = workflowForRole(role);
  finalizeSingle(draw);
}

void PoseDetector::runBoxDetection(
    const Onnx::ModelRole& detRole, const Onnx::ImageView& src)
{
  // detection_class: -1 = all classes, >=0 = that class id.
  const int class_sel = static_cast<int>(inputs.detection_class.value);
  m_dets = runDetector(detRole, src, Onnx::ModelDomain::Unknown, class_sel);
  // App-side NMS (idempotent for branches already NMS'd in runDetector): closes
  // the duplicate-box gap for end2end/PINTO detectors that feed the tracker.
  m_dets = Onnx::Detection::nms(std::move(m_dets), 0.45f, 0.8f);
  if(m_dets.empty())
  {
    // Box detection always tracks; age the tracker for ID continuity (coasted
    // re-emission needs keypoints, so box-only tracks just age, not redraw).
    coastOrPassthrough(PoseWorkflow::BoxDetection, src);
    return;
  }

  const int max_inst
      = std::clamp(static_cast<int>(inputs.max_instances.value), 1, 16);
  if(static_cast<int>(m_dets.size()) > max_inst)
  {
    std::partial_sort(
        m_dets.begin(), m_dets.begin() + max_inst, m_dets.end(),
        [](const auto& a, const auto& b) { return a.score > b.score; });
    m_dets.resize(max_inst);
  }

  m_instances.clear();
  m_instances.reserve(m_dets.size());
  for(const auto& d : m_dets)
  {
    DetectedPose p;
    p.mean_confidence = d.score;
    p.class_id = d.class_id;
    const auto b = d.box();
    p.box = {b.x, b.y, d.w, d.h};
    m_instances.push_back(std::move(p));
  }

  // Reuse the multi-instance back-end: tracking (box-only), Re-ID, per-id color,
  // and all the poses / detection / poses_geometry / count outputs.
  emitInstances(PoseWorkflow::BoxDetection, inputs.track_ids.value);
}

} // namespace OnnxModels
