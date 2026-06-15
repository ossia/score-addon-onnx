#include "PoseDetector_internal.hpp"

namespace OnnxModels
{

PoseDetector::PoseDetector() noexcept = default;
PoseDetector::~PoseDetector() = default;

// COCO-80 class names (the label set used by every YOLOX/YOLO/RTMDet COCO
// detector). Used to print a readable box label instead of the raw class id,
// unless the user loaded a custom class-names file.
static constexpr const char* COCO80_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

// Resolve a class id to a readable name: user-loaded custom list first, then the
// built-in COCO-80 table; returns nullptr when out of range (caller prints the
// numeric id).
const char* PoseDetector::className(int id) const
{
  if(id < 0)
    return nullptr;
  if(id < static_cast<int>(m_class_names.size()))
    return m_class_names[id].c_str();
  if(id < static_cast<int>(std::size(COCO80_NAMES)))
    return COCO80_NAMES[id];
  return nullptr;
}

// Lightweight ModelSpec -> ModelIO view for the classifier.
static Onnx::ModelIO toModelIO(const Onnx::ModelSpec& s)
{
  Onnx::ModelIO io;
  io.inputs.reserve(s.inputs.size());
  io.outputs.reserve(s.outputs.size());
  for(const auto& p : s.inputs)
    io.inputs.push_back({p.name, p.shape});
  for(const auto& p : s.outputs)
    io.outputs.push_back({p.name, p.shape});
  return io;
}

Onnx::ModelRole PoseDetector::roleForWorkflow(PoseWorkflow w) const
{
  // Keep the real model's input dims/layout; override the kind by selection.
  Onnx::ModelRole r = m_landmark_role;
  using K = Onnx::ModelKind;
  using S = Onnx::ModelStage;
  using D = Onnx::ModelDomain;
  switch(w)
  {
    case PoseWorkflow::BlazePose:
      r.kind = K::BlazePoseLandmark; r.stage = S::Landmark; r.domain = D::Body;
      break;
    case PoseWorkflow::RTMPose_COCO:
    case PoseWorkflow::RTMPose_Whole:
      r.kind = K::SimccPose; r.stage = S::Landmark; r.domain = D::Body;
      break;
    case PoseWorkflow::ViTPose:
      r.kind = K::HeatmapPose; r.stage = S::Landmark; r.domain = D::Body;
      break;
    case PoseWorkflow::AnimalPose:
      // keep the classified kind (SimccPose or HeatmapPose); just mark animal
      r.stage = S::Landmark; r.domain = D::Animal;
      break;
    case PoseWorkflow::YOLOPose:
      r.kind = K::YoloPose; r.stage = S::SingleStage; r.domain = D::Body;
      break;
    case PoseWorkflow::MediaPipeHands:
      r.kind = K::HandLandmark; r.stage = S::Landmark; r.domain = D::Hand;
      break;
    case PoseWorkflow::FaceMesh:
      r.kind = K::FaceMeshLandmark; r.stage = S::Landmark; r.domain = D::Face;
      break;
    case PoseWorkflow::BlazeFace:
      r.kind = K::BlazeFaceDetector; r.stage = S::Detector; r.domain = D::Face;
      break;
    case PoseWorkflow::MobileFaceNet:
      r.kind = K::MobileFaceNet; r.stage = S::Landmark; r.domain = D::Face;
      break;
    case PoseWorkflow::RTMPoseFace:
      r.kind = K::SimccPose; r.stage = S::Landmark; r.domain = D::Face;
      break;
    case PoseWorkflow::BoxDetection:
      // Detection-only: the relevant model is the Detection Model; the box path
      // dispatches on m_detector_role directly, so leave the landmark role as-is.
      break;
    case PoseWorkflow::Auto:
    default:
      break;
  }
  return r;
}

void PoseDetector::passthrough(const Onnx::ImageView& src)
{
  ONNX_PROF_SCOPE(Draw);
  outputs.detection.value.reset();
  outputs.geometry.value.clear();
  outputs.poses.value.clear();
  outputs.poses_geometry.value.clear();
  outputs.count.value = 0;
  outputs.image.create(src.w, src.h);
  // Honor Skeleton-only here too: fill black on a no-detection frame instead of
  // copying the input (otherwise the raw frame flashes through during dropouts).
  const bool skeleton_only
      = (inputs.output_mode.value == PoseRenderMode::SkeletonOnly);
  fillCanvas(
      reinterpret_cast<unsigned char*>(outputs.image.texture.bytes), src.data,
      src.w, src.h, skeleton_only);
  outputs.image.texture.changed = true;

  // Coast: keep temporal state for a few frames so a brief detection dropout
  // doesn't restart smoothing/tracking and lurch on re-acquisition. Only after
  // a sustained loss do we drop everything.
  ++m_lost_frames;
  if(m_lost_frames > 8)
  {
    m_smoother.reset();
    m_roi_smoother.reset();
    m_tracking = false;
    m_last_keypoints.clear();
  }
}

void PoseDetector::operator()()
try
{
  if(!available)
    return;

  auto& in_tex = inputs.image.texture;
  if(!in_tex.changed)
    return;
  if(!in_tex.bytes)
    return;

  // Parse the custom class-names file once, when it changes (the file_port
  // loads the bytes for us). One name per line; blank lines kept so line N maps
  // to class id N.
  if(std::string fn{inputs.class_file.file.filename}; fn != m_last_class_file)
  {
    m_last_class_file = std::move(fn);
    m_class_names.clear();
    const std::string_view bytes{inputs.class_file.file.bytes};
    for(std::size_t pos = 0; pos < bytes.size();)
    {
      const std::size_t nl = bytes.find('\n', pos);
      const std::size_t end = (nl == std::string_view::npos) ? bytes.size() : nl;
      std::string_view line = bytes.substr(pos, end - pos);
      if(!line.empty() && line.back() == '\r')
        line.remove_suffix(1);
      m_class_names.emplace_back(line);
      if(nl == std::string_view::npos)
        break;
      pos = nl + 1;
    }
  }

  const bool have_landmark = !inputs.model.current_model_invalid
                             && inputs.model.file.bytes.size() >= 32;
  const bool have_det = !inputs.det_model.current_model_invalid
                        && inputs.det_model.file.bytes.size() >= 32;
  const bool have_reid = inputs.reid.value
                         && !inputs.reid_model.current_model_invalid
                         && inputs.reid_model.file.bytes.size() >= 32;

  // Reset contexts on workflow / model change.
  const PoseWorkflow wf = inputs.workflow.value;

  // Box Detection runs the Detection Model with no landmark stage: explicit
  // workflow, or Auto with a Detection Model and no Landmark Model loaded.
  const bool box_only
      = have_det
        && (wf == PoseWorkflow::BoxDetection
            || (wf == PoseWorkflow::Auto && !have_landmark));
  if(!box_only && !have_landmark)
    return;

  ONNX_PROF_FRAME(); // counts only frames that actually process

  bool reinit = false;
  if(wf != m_last_workflow)
  {
    ctx.reset();
    det_ctx.reset();
    m_last_workflow = wf;
    reinit = true;
  }
  if(inputs.model.file.filename != m_last_model)
  {
    ctx.reset();
    m_last_model = std::string(inputs.model.file.filename);
    reinit = true;
  }
  if(inputs.det_model.file.filename != m_last_det_model)
  {
    det_ctx.reset();
    m_last_det_model = std::string(inputs.det_model.file.filename);
    reinit = true;
  }
  if(inputs.reid_model.file.filename != m_last_reid_model)
  {
    reid_ctx.reset();
    m_reid_spec = {};
    m_last_reid_model = std::string(inputs.reid_model.file.filename);
    reinit = true;
  }
  // A model/workflow change invalidates the temporal tracking/smoothing state.
  if(reinit)
  {
    m_tracking = false;
    m_last_keypoints.clear();
    m_roi_smoother.reset();
    m_smoother.reset();
    m_tracker.reset();
    m_lost_frames = 0;
    m_frames_since_detect = 0;
  }

  // Model construction is the only failure that should permanently invalidate
  // the node; a per-frame inference/decode exception must NOT (it would kill the
  // node forever after a single transient throw — see the function catch below).
  try
  {
    if(have_landmark && !this->ctx)
    {
      this->ctx = std::make_unique<Onnx::OnnxRunContext>(
          this->inputs.model.file.bytes);
      m_landmark_role = Onnx::classify(toModelIO(this->ctx->readModelSpec()));
    }
    if(have_det && !this->det_ctx)
    {
      this->det_ctx = std::make_unique<Onnx::OnnxRunContext>(
          this->inputs.det_model.file.bytes);
      m_detector_role
          = Onnx::classify(toModelIO(this->det_ctx->readModelSpec()));
    }
    if(have_reid && !this->reid_ctx)
    {
      this->reid_ctx = std::make_unique<Onnx::OnnxRunContext>(
          this->inputs.reid_model.file.bytes);
      m_reid_spec
          = Onnx::classifyReid(toModelIO(this->reid_ctx->readModelSpec()));
    }
  }
  catch(...)
  {
    // Invalidate whichever model we were actually trying to construct.
    if(box_only)
      inputs.det_model.current_model_invalid = true;
    else
      inputs.model.current_model_invalid = true;
    ctx.reset();
    det_ctx.reset();
    reid_ctx.reset();
    return;
  }

  Onnx::ImageView src{
      reinterpret_cast<const uint8_t*>(in_tex.bytes), in_tex.width,
      in_tex.height, 4, in_tex.width * 4};

  // --- Box Detection: run the Detection Model, emit boxes (no landmark) ---
  if(box_only)
  {
    runBoxDetection(m_detector_role, src);
    return;
  }

  const Onnx::ModelRole role
      = (wf == PoseWorkflow::Auto) ? m_landmark_role : roleForWorkflow(wf);
  const PoseWorkflow draw
      = (wf == PoseWorkflow::Auto) ? workflowForRole(role) : wf;

  // --- Two-stage: detector + landmark ---
  if(have_det && role.stage == Onnx::ModelStage::Landmark)
  {
    // Track IDs on -> multi-instance pipeline (all people, ids, per-id color).
    if(inputs.track_ids.value)
    {
      runMultiInstance(role, draw, src);
      return;
    }

    const int mw = role.input_w > 0 ? role.input_w : 256;
    const int mh = role.input_h > 0 ? role.input_h : 256;

    // --- ROI: tracking loop (skip detector) vs fresh detection -------------
    // The tracking ROI is derived from the model's own output, i.e. a feedback
    // loop. It is only stable for the MediaPipe-rotated landmark models, which
    // produce a well-localized rotated ROI (BlazePose/Hand/FaceMesh). For
    // top-down models (SimCC/heatmap) it is bbox->bbox feedback that jitters,
    // and MobileFaceNet (fill-the-crop) explodes — those re-detect every frame.
    const bool can_track
        = inputs.track_roi.value
          && (role.kind == Onnx::ModelKind::BlazePoseLandmark
              || role.kind == Onnx::ModelKind::HandLandmark
              || role.kind == Onnx::ModelKind::FaceMeshLandmark);
    Onnx::ROI::Rect rect;
    bool from_tracking = false;
    if(can_track && m_tracking && !m_last_keypoints.empty())
    {
      // Derive the ROI from last frame's landmarks — no detector this frame.
      Onnx::ROI::Rect cand = roiRectFromKeypoints(
          draw, m_last_keypoints, in_tex.width, in_tex.height, mw, mh);
      // Only trust it if it's well-formed and didn't teleport/shrink vs the
      // previous ROI — otherwise re-detect (prevents drift/center-collapse).
      if(rectValid(cand, in_tex.width, in_tex.height)
         && (!m_have_prev_roi || rectPlausible(cand, m_prev_roi)))
      {
        rect = cand;
        from_tracking = true;
      }
    }
    if(!from_tracking)
    {
      auto dets = runDetector(m_detector_role, src, role.domain);
      if(dets.empty())
      {
        passthrough(src);
        return;
      }
      rect = detectionRect(role, dets.front(), in_tex.width, in_tex.height);
      // Only reset the ROI smoother on a genuine RE-ACQUISITION (the previous
      // frame was a dropout, m_lost_frames>0) so we don't blend across the gap.
      // For a continuous top-down stream (which re-detects every frame, so
      // from_tracking is always false) the smoother must stay WARM — otherwise
      // smoothRoi restarts cold every frame, never actually smooths, and the
      // per-frame crop jitter shows up as jumpy keypoints (rtmpose-coco).
      if(m_lost_frames > 0 || !m_have_prev_roi)
      {
        m_roi_smoother.reset();
        m_have_prev_roi = false;
      }
    }

    // Stabilize the crop. Deadband first: if the tracked ROI barely changed,
    // reuse the previous one verbatim so a static subject gives a static crop
    // (kills the feedback shake); otherwise smooth toward the new ROI.
    if(from_tracking && m_have_prev_roi && rectClose(rect, m_prev_roi))
      rect = m_prev_roi;
    else
      rect = smoothRoi(rect);
    m_prev_roi = rect;
    m_have_prev_roi = true;
    const Onnx::Affine M = Onnx::ROI::rectToAffine(rect, mw, mh);
    runLandmark(role, draw, src, M);

    // --- Tracking gate: keep tracking only if the landmark model is confident
    if(inputs.track_roi.value && outputs.detection.value
       && outputs.detection.value->mean_confidence
              >= std::max(0.2f, static_cast<float>(inputs.min_confidence)))
    {
      m_tracking = true;
      m_last_keypoints = m_native_keypoints; // pre-remap native joints
    }
    else
    {
      m_tracking = false; // lost -> re-detect next frame
      m_last_keypoints.clear();
    }
    return;
  }

  // --- Single model ---
  switch(role.stage)
  {
    case Onnx::ModelStage::Detector:
      runDetectorAsPose(role, src);
      break;
    case Onnx::ModelStage::SingleStage:
    {
      if(role.kind == Onnx::ModelKind::RtmoPose)
      {
        runRTMO(src);
      }
      else
      {
        const int ms = role.input_w > 0 ? role.input_w : 640;
        const Onnx::Affine M = Onnx::ROI::wholeFrameAffine(
            in_tex.width, in_tex.height, ms, ms);
        runYOLOPose(src, M);
      }
      break;
    }
    case Onnx::ModelStage::Landmark:
    default:
    {
      const int mw = role.input_w > 0 ? role.input_w : 256;
      const int mh = role.input_h > 0 ? role.input_h : 256;
      const Onnx::Affine M = Onnx::ROI::wholeFrameAffine(
          in_tex.width, in_tex.height, mw, mh);
      runLandmark(role, draw, src, M);
      break;
    }
  }
}
catch(...)
{
  // Transient per-frame failure (a bad crop, an odd output shape on one frame,
  // an ORT hiccup). Skip this frame and retry next one — do NOT permanently
  // invalidate the model (construction failures are handled separately above).
}


} // namespace OnnxModels
