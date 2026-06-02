#pragma once
#include <OnnxModels/Utils.hpp>

#include <Onnx/helpers/Detection.hpp>
#include <Onnx/helpers/ModelRole.hpp>
#include <Onnx/helpers/OneEuro.hpp>
#include <Onnx/helpers/ROI.hpp>

#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/geometry.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <optional>
#include <vector>

class QImage;
class QTransform;

namespace Onnx
{
struct OnnxRunContext;
}

namespace OnnxModels
{
// Unified keypoint structure for all pose models
struct PoseKeypoint
{
  float x, y, z;       // z is 0 for 2D-only models
  float confidence;

  halp_field_names(x, y, z, confidence);
};

struct DetectedPose
{
  std::vector<PoseKeypoint> keypoints;
  float mean_confidence;

  halp_field_names(keypoints, mean_confidence);
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

  // Hand
  MediaPipeHands, // MediaPipe Hands (21 keypoints, NHWC, direct landmarks)

  // Face
  FaceMesh,      // MediaPipe FaceMesh (468 keypoints, NHWC, direct landmarks)
  BlazeFace,     // BlazeFace detection (6 keypoints, NHWC, anchor-based)
  MobileFaceNet, // MobileFaceNet (68 dlib landmarks, NCHW)

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
      "Unified keypoint detection for body pose, hands, and face landmarks");
  halp_meta(uuid, "f8e7d6c5-b4a3-4291-8c0d-1e2f3a4b5c6d");
  halp_meta(manual_url, "https://ossia.io/score-docs/processes/ai-recognition.html")

  struct
  {
    halp::texture_input<"In"> image;
    ModelPort<"Model"> model;

    // Optional stage-1 detector. When set (and the main model is a landmark
    // model), the detector locates the ROI on the full frame, the frame is
    // warp-cropped, and the main model runs inside the crop. Leave empty for
    // single-stage / whole-frame inference.
    ModelPort<"Detection Model"> det_model;

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

    struct : halp::toggle<"Track ROI">
    {
      halp_meta(
          description,
          "Two-stage only: derive the ROI from the previous frame's landmarks "
          "and skip the detector (faster + far steadier, like MediaPipe)");
      bool value = true;
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

    struct : halp::toggle<"Async">
    {
      halp_meta(description, "Run inference asynchronously");
    } async;
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
  } outputs;

  PoseDetector() noexcept;
  ~PoseDetector();

  void operator()();

private:
  // --- Two-stage building blocks ---
  // Stage 1: run the detector on the full frame, return detections in
  // image-normalized [0,1] coordinates (letterbox removed).
  std::vector<Onnx::Detection::Detection>
  runDetector(const Onnx::ModelRole& role, const QImage& src);

  // Stage 2: run the landmark/pose model on the crop defined by M
  // (crop-pixels -> image-pixels), then map keypoints back through M.
  void runLandmark(
      const Onnx::ModelRole& role, PoseWorkflow draw, const QImage& src,
      const QTransform& M);

  // Draw a detector's own keypoints/box directly (detector used standalone).
  void runDetectorAsPose(const Onnx::ModelRole& role, const QImage& src);

  // Single-stage YOLO-pose on the full frame.
  void runYOLOPose(const QImage& src, const QTransform& M);

  // Single-stage RTMO on the full frame (dets + keypoints, NMS-free).
  void runRTMO(const QImage& src);

  // Map a manual workflow selection onto a concrete model role.
  Onnx::ModelRole roleForWorkflow(PoseWorkflow w) const;

  // Common visualization
  void drawSkeleton(const DetectedPose& pose, PoseWorkflow workflow);
  void generateGeometryOutput(const DetectedPose& pose, PoseWorkflow workflow);
  void passthrough(const QImage& src);

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
  boost::container::vector<float> storage;
  boost::container::vector<float> det_storage;

  // Cache for avoiding re-initialization
  PoseWorkflow m_last_workflow{PoseWorkflow::Auto};
  Onnx::ModelRole m_landmark_role;
  Onnx::ModelRole m_detector_role;
  Onnx::PoseSmoother m_smoother;

  // Tracking-loop state (two-stage path)
  bool m_tracking{false};               // valid ROI carried from prev frame?
  Onnx::RectSmoother m_roi_smoother;    // temporal ROI stabilization
  std::vector<PoseKeypoint> m_last_keypoints; // image-normalized, prev frame
  int m_lost_frames{0};                 // consecutive frames without a pose

  std::string m_last_model;
  std::string m_last_det_model;
};

} // namespace OnnxModels
