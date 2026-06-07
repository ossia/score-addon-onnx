#pragma once
#include <OnnxModels/Utils.hpp>

#include <Onnx/helpers/Detection.hpp>
#include <Onnx/helpers/ModelRole.hpp>
#include <Onnx/helpers/OnnxBase.hpp>
#include <Onnx/helpers/OneEuro.hpp>
#include <Onnx/helpers/PoseTracker.hpp>
#include <Onnx/helpers/ROI.hpp>
#include <Onnx/helpers/SkeletonFormats.hpp>

#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/geometry.hpp>
#include <halp/layout.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <optional>
#include <vector>


namespace Onnx
{
struct OnnxRunContext;
}

namespace OnnxModels
{
struct Overlay; // ctx-backed software overlay renderer (CtxOverlay.hpp)

// Unified keypoint structure for all pose models
struct PoseKeypoint
{
  float x, y, z;       // z is 0 for 2D-only models
  float confidence;

  halp_field_names(x, y, z, confidence);
};

// Axis-aligned bounding box, normalized [0,1], top-left form.
struct BoundingBox
{
  float x{}, y{}, w{}, h{};
  halp_field_names(x, y, w, h);
};

struct DetectedPose
{
  std::vector<PoseKeypoint> keypoints; // empty for box-only detections
  float mean_confidence;
  int track_id = -1; // persistent ID across frames (-1 = untracked)
  BoundingBox box;   // always filled: detector box, or the keypoint bbox
  int class_id = -1; // detector class (-1 = n/a, e.g. landmark-only output)

  halp_field_names(keypoints, mean_confidence, track_id, box, class_id);
};

// Available pose estimation workflows
enum class PoseWorkflow
{
  Auto, // Automatic detection from model structure

  // Body pose
  BlazePose,     // MediaPipe BlazePose (33 keypoints, NHWC, direct landmarks)
  RTMPose_COCO,  // RTMPose COCO format (17 keypoints, NCHW, SimCC)
  RTMPose_Whole, // RTMPose WholeBody (133 keypoints, NCHW, SimCC)
  ViTPose,       // ViTPose (17 keypoints, NCHW, heatmaps)
  YOLOPose,      // YOLO Pose (17 keypoints, NCHW, direct output)
  AnimalPose,    // AP10K/APT36K quadruped (17 keypoints, RTMPose/ViTPose)

  // Hand
  MediaPipeHands, // MediaPipe Hands (21 keypoints, NHWC, direct landmarks)

  // Face
  FaceMesh,      // MediaPipe FaceMesh (468 keypoints, NHWC, direct landmarks)
  BlazeFace,     // BlazeFace detection (6 keypoints, NHWC, anchor-based)
  MobileFaceNet, // MobileFaceNet (68 dlib landmarks, NCHW)
  RTMPoseFace,   // RTMPose-Face (106 LaPa landmarks, NCHW square, SimCC)

  // Detection-only
  BoxDetection,  // Bounding boxes from the Detection Model (no landmark stage)
};

// ReID input preprocessing (the part not inferable from the ONNX graph).
// Auto picks by input size: 112->ArcFace, 128->RawBGR, else ImageNet-RGB.
enum class ReidPreprocess : uint8_t
{
  Auto,
  ImageNetRGB,  // (x/255 - mean)/std, RGB  (OSNet, FastReID)
  RawBGR,       // x in [0,255], BGR        (OpenVINO person/face-reid)
  RawRGB,       // x in [0,255], RGB
  ZeroOneRGB,   // x/255, RGB
  ArcFaceRGB,   // (x-127.5)/128, RGB       (ArcFace)
};

// Output visualization mode
enum class OutputMode
{
  SkeletonOnImage, // Draw skeleton on top of input image
  SkeletonOnly,    // Draw skeleton on black background
};

// Keypoint data output format
enum class KeypointOutputFormat
{
  Raw,       // Raw structured output (x, y, z, confidence per keypoint)
  XYArray,   // Flat xyxy array
  XYZArray,  // Flat xyzxyz array
  LineArray, // Flat xyz pairs for e.g. GL_LINES (x1,y1,z1,x2,y2,z2,...)
};

struct PoseDetector : OnnxObject
{
public:
  halp_meta(name, "Pose Detector");
  halp_meta(c_name, "pose_detector");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "MediaPipe, MMPose, ONNX Runtime");
  halp_meta(
      description,
      "Unified keypoint detection for body pose, hands, and face landmarks, "
      "plus bounding-box object detection and multi-object tracking");
  halp_meta(uuid, "f8e7d6c5-b4a3-4291-8c0d-1e2f3a4b5c6d");
  halp_meta(manual_url, "https://ossia.io/score-docs/processes/ai-recognition.html")

  struct ins
  {
    halp::texture_input<"In"> image;
    ModelPort<"Landmark Model"> model;

    struct : halp::combobox_t<"Workflow", PoseWorkflow>
    {
      halp_meta(description, "Pose estimation model type");

    } workflow;

    struct : halp::enum_t<OutputMode, "Output Mode">
    {
      halp_meta(description, "Visualization output mode");
    } output_mode;

    halp::hslider_f32<"Min Confidence", halp::range{0., 1., 0.3}> min_confidence;

    struct : halp::toggle<"Draw Skeleton">
    {
      bool value = true;
    } draw_skeleton;

    struct : halp::enum_t<KeypointOutputFormat, "Data Format">
    {
      halp_meta(description, "Output data format for GPU rendering");
    } data_format;


    ModelPort<"Detection Model"> det_model;

    struct : halp::toggle<"Track ROI">
    {
      halp_meta(
          description,
          "Two-stage only: derive the ROI from the previous frame's landmarks "
          "and skip the detector (faster + steadier, like MediaPipe). "
          "Experimental — feedback can drift on some models; off by default.");
      bool value = false;
    } track_roi;

    struct : halp::toggle<"Smoothing">
    {
      halp_meta(description, "Temporal One-Euro smoothing of keypoints");
      bool value = true;
    } smoothing;

    struct : halp::hslider_f32<"Smoothing Amount", halp::range{0., 1., 0.5}>
    {
      halp_meta(description, "0 = responsive, 1 = very smooth");
    } smoothing_amount;

    struct : halp::toggle<"Track IDs">
    {
      halp_meta(
          description,
          "Track every person/hand/face across frames and emit them all with a "
          "persistent ID + stable per-ID color (ByteTrack: Kalman + two-stage + "
          "OKS). Enables multi-instance output. Requires a Detection Model.");
      bool value = false;
    } track_ids;

    struct : halp::spinbox_i32<"Max Instances", halp::range{1, 16, 5}>
    {
      halp_meta(
          description,
          "Max simultaneous tracked instances (top-K by score). Track IDs only.");
    } max_instances;

    struct : halp::spinbox_i32<"Detector Cadence", halp::range{1, 30, 4}>
    {
      halp_meta(
          description,
          "Track ROI: re-run the detector every N frames; reuse per-track ROIs "
          "in between (1 = detect every frame). Track IDs + Track ROI only.");
    } detector_cadence;

    ModelPort<"Re-ID Model"> reid_model;

    struct : halp::toggle<"Re-ID">
    {
      halp_meta(
          description,
          "Blend an appearance embedding (any ReID model in the Re-ID Model "
          "port) into tracking so IDs survive long occlusion / re-entry. "
          "Track IDs only.");
      bool value = false;
    } reid;

    struct : halp::hslider_f32<"Re-ID Weight", halp::range{0., 1., 0.25}>
    {
      halp_meta(description, "How strongly appearance influences association.");
    } reid_weight;

    struct : halp::enum_t<ReidPreprocess, "Re-ID Preprocess">
    {
      halp_meta(
          description,
          "Input normalization for the Re-ID model (not inferable from the "
          "graph). Auto guesses from input size.");
    } reid_preprocess;

    // Appended after reid_preprocess to keep existing presets' port indices
    // (1..17) stable; these two are 18 and 19.
    struct : halp::toggle<"Draw Boxes">
    {
      halp_meta(
          description,
          "Draw each instance's bounding box (always on in Box Detection).");
      bool value = false;
    } draw_boxes;

    struct : halp::spinbox_i32<"Detection Class", halp::range{-1, 90, -1}>
    {
      halp_meta(
          description,
          "Box Detection: keep only this class id (-1 = all). Multi-class "
          "detectors only (COCO ids); ignored by single-class detectors.");
    } detection_class;

    struct : halp::toggle<"Draw Landmarks">
    {
      halp_meta(
          description,
          "Draw the keypoint dots (independent of the skeleton lines).");
      bool value = true;
    } draw_landmarks;

    // --- Tracking plausibility gates (anti-jitter; Track IDs only) ---
    struct : halp::combobox_t<"Motion Gate", Onnx::Track::MotionGate>
    {
      halp_meta(
          description,
          "Reject an id->detection match that is an implausible jump: None, "
          "MaxSpeed (an id can't cross the frame in one step), or Mahalanobis "
          "(Kalman gating distance, gap-aware). An appearance/ReID match can "
          "still re-acquire across the gate.");
    } motion_gate;

    struct : halp::hslider_f32<"Max Speed", halp::range{0.25, 6., 2.}>
    {
      halp_meta(
          description,
          "MaxSpeed gate budget: max per-frame center move, in units of the "
          "track's box size (scaled by how long it was lost). Lower = stricter.");
    } max_speed;

    struct : halp::toggle<"Birth Gate">
    {
      halp_meta(
          description,
          "Suppress spurious new ids whose box sits mostly inside an existing "
          "tracked person (e.g. a raised arm read as a second person).");
      bool value = false;
    } birth_gate;

    struct : halp::toggle<"Strict Confirmation">
    {
      halp_meta(
          description,
          "Delete a tentative id the first frame it fails to re-appear "
          "(DeepSORT n_init), so a one-frame detection never persists.");
      bool value = false;
    } strict_confirm;

    struct : halp::combobox_t<"Skeleton", Onnx::Skel::TargetSkeleton>
    {
      halp_meta(
          description,
          "Remap output keypoints (overlay + ports) to a standard skeleton "
          "layout. Native = the model's own layout; unsupported (e.g. animal "
          "-> human) falls back to Native.");
    } skeleton_type;

  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

    struct
    {
      halp_meta(name, "Detection");
      std::optional<DetectedPose> value;
    } detection;

    struct
    {
      halp_meta(name, "Geometry");
      std::vector<float> value;
    } geometry;

    // --- Multi-instance outputs (populated when Track IDs is on) ---
    struct
    {
      halp_meta(name, "Poses");
      std::vector<DetectedPose> value; // every tracked instance, each w/ track_id
    } poses;

    struct
    {
      halp_meta(name, "Poses Geometry");
      // Fixed-stride, zero-padded: Max Instances slots, each = [track_id, then
      // the same layout as Geometry]. count gives the number of live slots.
      std::vector<float> value;
    } poses_geometry;

    struct
    {
      halp_meta(name, "Count");
      int value{}; // number of live tracked instances this frame
    } count;
  } outputs;

  // Inspector layout: group the controls into meaningful tabs.
  struct ui
  {
    halp_meta(name, "Pose Detector")
    halp_meta(layout, halp::layouts::tabs)
    halp_meta(background, halp::colors::background_mid)

    struct
    {
      halp_meta(name, "Models")
      halp_meta(layout, halp::layouts::vbox)
      halp::item<&ins::model> model;
      halp::item<&ins::det_model> det_model;
      halp::item<&ins::workflow> workflow;
      halp::item<&ins::detection_class> detection_class;
    } models_tab;

    struct
    {
      halp_meta(name, "Output")
      halp_meta(layout, halp::layouts::vbox)
      halp::item<&ins::min_confidence> min_confidence;
      halp::item<&ins::output_mode> output_mode;
      halp::item<&ins::draw_skeleton> draw_skeleton;
      halp::item<&ins::draw_landmarks> draw_landmarks;
      halp::item<&ins::draw_boxes> draw_boxes;
      halp::item<&ins::data_format> data_format;
      halp::item<&ins::skeleton_type> skeleton_type;
    } output_tab;

    struct
    {
      halp_meta(name, "Tracking")
      halp_meta(layout, halp::layouts::vbox)
      halp::item<&ins::track_ids> track_ids;
      halp::item<&ins::max_instances> max_instances;
      halp::item<&ins::track_roi> track_roi;
      halp::item<&ins::detector_cadence> detector_cadence;
      halp::item<&ins::motion_gate> motion_gate;
      halp::item<&ins::max_speed> max_speed;
      halp::item<&ins::birth_gate> birth_gate;
      halp::item<&ins::strict_confirm> strict_confirm;
    } tracking_tab;

    struct
    {
      halp_meta(name, "Re-ID")
      halp_meta(layout, halp::layouts::vbox)
      halp::item<&ins::reid_model> reid_model;
      halp::item<&ins::reid> reid;
      halp::item<&ins::reid_weight> reid_weight;
      halp::item<&ins::reid_preprocess> reid_preprocess;
    } reid_tab;

    struct
    {
      halp_meta(name, "Smoothing")
      halp_meta(layout, halp::layouts::vbox)
      halp::item<&ins::smoothing> smoothing;
      halp::item<&ins::smoothing_amount> smoothing_amount;
    } smoothing_tab;
  };

  PoseDetector() noexcept;
  ~PoseDetector();

  void operator()();

private:
  // --- Two-stage building blocks ---
  // Stage 1: run the detector on the full frame, return detections in
  // image-normalized [0,1] coordinates (letterbox removed).
  // keep_class: -2 = use the target-domain default class filter (person/animal),
  // -1 = keep all classes, >=0 = keep only that class id.
  std::vector<Onnx::Detection::Detection> runDetector(
      const Onnx::ModelRole& role, const Onnx::ImageView& src,
      Onnx::ModelDomain target = Onnx::ModelDomain::Body, int keep_class = -2);

  // Stage 2: run the landmark/pose model on the crop defined by M
  // (crop-pixels -> image-pixels), then map keypoints back through M.
  void runLandmark(
      const Onnx::ModelRole& role, PoseWorkflow draw, const Onnx::ImageView& src,
      const Onnx::Affine& M, int track_id = -1);

  // Stage 2 core: run the landmark model on one crop and decode keypoints into
  // `out` (image-normalized [0,1]); no smoothing/drawing/output. Returns the
  // mean confidence, or -1 on decode failure. Shared by single + multi paths.
  float landmarkKeypoints(
      const Onnx::ModelRole& role, const Onnx::ImageView& src, const Onnx::Affine& M,
      std::vector<PoseKeypoint>& out);

  // --- Multi-instance back-end (Track IDs path) ---
  // Run the detector (top-K) or per-track ROIs, landmark each, track, per-id
  // smooth, draw all, and fill the poses / primary / geometry / count outputs.
  void runMultiInstance(
      const Onnx::ModelRole& role, PoseWorkflow draw, const Onnx::ImageView& src);
  // Landmark every ROI into m_instances, batching all crops into one inference
  // when the model's batch dim allows it (else one inference per crop).
  void runLandmarkBatch(
      const Onnx::ModelRole& role, PoseWorkflow draw, const Onnx::ImageView& src,
      const std::vector<Onnx::ROI::Rect>& rois);
  // Track m_instances, assign ids + per-id smoothed keypoints + colors, then
  // draw all and publish every output port.
  void emitInstances(PoseWorkflow draw, bool do_track);
  // Crop each entry of m_track_in, run the Re-ID model (batched if it allows),
  // and write an L2-normalized embedding into each m_track_in[i].embedding.
  void embedInstances();

  // Draw a detector's own keypoints/box directly (detector used standalone).
  void runDetectorAsPose(const Onnx::ModelRole& role, const Onnx::ImageView& src);

  // Single-stage YOLO-pose on the full frame.
  void runYOLOPose(const Onnx::ImageView& src, const Onnx::Affine& M);

  // Single-stage RTMO on the full frame (dets + keypoints, NMS-free).
  void runRTMO(const Onnx::ImageView& src);

  // Detection-only: run the Detection Model, emit boxes as keypoint-less poses
  // (optionally tracked). Drawn as rectangles; box lives in each pose's metadata.
  void runBoxDetection(const Onnx::ModelRole& detRole, const Onnx::ImageView& src);

  // Map a manual workflow selection onto a concrete model role.
  Onnx::ModelRole roleForWorkflow(PoseWorkflow w) const;

  // Common visualization
  void drawSkeleton(const DetectedPose& pose, PoseWorkflow workflow);
  // Draw all of m_instances onto one output image (multi-instance path).
  void drawAllSkeletons(PoseWorkflow workflow);
  // Draw one pose's connections + points into an open ctx overlay.
  void drawOnePose(
      Overlay& ov, const DetectedPose& pose, PoseWorkflow workflow, int w,
      int h);
  void generateGeometryOutput(const DetectedPose& pose, PoseWorkflow workflow);

  // Skeleton remap: set m_remap_* from the control + workflow, remap a pose's
  // keypoints in place (native -> target), and finalize a single-instance pose
  // (remap + draw + geometry). Remap drives both overlay and output ports.
  void setRemapState(PoseWorkflow workflow, int num_kps);
  void remapPose(DetectedPose& pose);
  void finalizeSingle(PoseWorkflow workflow);
  // Append one pose's flattened geometry (current Data Format) to `out`.
  void appendGeometry(
      std::vector<float>& out, const DetectedPose& pose, PoseWorkflow workflow);
  void passthrough(const Onnx::ImageView& src);

  // Temporal One-Euro smoothing of the detected keypoints (in place).
  void applySmoothing(DetectedPose& pose);

  // ROI as a rect (image px) from a detection, by landmark kind.
  Onnx::ROI::Rect
  detectionRect(const Onnx::ModelRole& role,
                const Onnx::Detection::Detection& det, int W, int H);
  // ROI rect derived from the previous frame's landmarks (tracking loop).
  Onnx::ROI::Rect roiRectFromKeypoints(
      PoseWorkflow draw, const std::vector<PoseKeypoint>& kps, int W, int H,
      int model_w, int model_h);
  // Temporally smooth the ROI rect so the crop stays steady.
  Onnx::ROI::Rect smoothRoi(Onnx::ROI::Rect r);

  std::unique_ptr<Onnx::OnnxRunContext> ctx;     // main / landmark model
  std::unique_ptr<Onnx::OnnxRunContext> det_ctx; // optional stage-1 detector
  std::unique_ptr<Onnx::OnnxRunContext> reid_ctx; // optional appearance ReID
  boost::container::vector<float> storage;
  boost::container::vector<float> det_storage;

  // Cache for avoiding re-initialization
  PoseWorkflow m_last_workflow{PoseWorkflow::Auto};
  Onnx::ModelRole m_landmark_role;
  Onnx::ModelRole m_detector_role;
  Onnx::PoseSmoother m_smoother;

  // Multi-instance persistent-ID tracker (ByteTrack-style; opt-in via Track IDs).
  Onnx::Track::PoseTracker m_tracker;

  // Tracking-loop state (two-stage path)
  bool m_tracking{false};               // valid ROI carried from prev frame?
  Onnx::RectSmoother m_roi_smoother;    // temporal ROI stabilization
  std::vector<PoseKeypoint> m_last_keypoints; // image-normalized, prev frame
  int m_lost_frames{0};                 // consecutive frames without a pose
  Onnx::ROI::Rect m_prev_roi{};         // last smoothed ROI (plausibility check)
  bool m_have_prev_roi{false};

  Onnx::ReidSpec m_reid_spec;
  boost::container::vector<float> m_reid_batch; // packed [N,3,H,W] reid input
  boost::container::vector<float> m_reid_tmp;   // per-crop reid build scratch

  std::string m_last_model;
  std::string m_last_det_model;
  std::string m_last_reid_model;

  // --- Cached scratch reused every frame (zero steady-state allocation) ---
  std::vector<DetectedPose> m_instances;          // this frame's instances
  std::vector<PoseKeypoint> m_kp_scratch;         // landmark decode -> keypoints
  std::vector<Onnx::Track::Detection> m_track_in; // tracker input
  std::vector<Onnx::Detection::Detection> m_dets; // detector output (top-K)
  std::vector<Onnx::ROI::Rect> m_rois;            // ROIs to landmark this frame
  boost::container::vector<float> m_batch_storage; // packed [N,C,H,W] input
  std::vector<int64_t> m_bbox;                     // batched SimCC [N,2] bbox
  int m_frames_since_detect{0};                   // detector-cadence counter

  // Skeleton remap state (recomputed per emit from the Skeleton control).
  bool m_remap_active{false};
  Onnx::Skel::SourceSkeleton m_remap_src{Onnx::Skel::SourceSkeleton::Coco17};
  Onnx::Skel::TargetSkeleton m_active_target{Onnx::Skel::TargetSkeleton::Native};
  std::vector<PoseKeypoint> m_remap_scratch; // reused remap output buffer
};

} // namespace OnnxModels
