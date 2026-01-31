#pragma once
#include <OnnxModels/Utils.hpp>

#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/geometry.hpp>
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
  // Workflow-specific inference functions
  void runBlazePose();
  void runRTMPose();
  void runViTPose();
  void runYOLOPose();
  void runMediaPipeHands();
  void runFaceMesh();
  void runBlazeFace();
  void runMobileFaceNet();

  // Auto-detection of workflow from model structure
  PoseWorkflow detectWorkflowFromModel();

  // Common visualization
  void drawSkeleton(const DetectedPose& pose, PoseWorkflow workflow);

  // Generate geometry output based on format setting
  void generateGeometryOutput(const DetectedPose& pose, PoseWorkflow workflow);

  std::unique_ptr<Onnx::OnnxRunContext> ctx;
  boost::container::vector<float> storage;

  // Cache for avoiding re-initialization
  PoseWorkflow m_last_workflow{PoseWorkflow::Auto};
  PoseWorkflow m_detected_workflow{PoseWorkflow::BlazePose};

  std::string m_last_model;
};

} // namespace OnnxModels
