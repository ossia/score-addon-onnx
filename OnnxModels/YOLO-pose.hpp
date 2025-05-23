#pragma once
#include <OnnxModels/Utils.hpp>

#include <cmath>
#include <halp/controls.hpp>
#include <halp/geometry.hpp>
#include <halp/meta.hpp>
#include <halp/sample_accurate_controls.hpp>
#include <halp/texture.hpp>

namespace Onnx
{
struct OnnxRunContext;
}

namespace OnnxModels::Yolo
{
struct Keypoint
{
  int kp;
  float x, y;
  halp_field_names(kp, x, y);
};

struct DetectedYoloPose
{
  std::string name;
  halp::rect2d<float> geometry;
  float probability{};
  std::vector<Keypoint> keypoints;

  halp_field_names(name, geometry, probability, keypoints);
};
struct PoseDetector : OnnxObject
{
public:
  halp_meta(name, "YOLO Pose");
  halp_meta(c_name, "yolov8_pose");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "YOLOv8 authors, Onnxruntime, TensorRT");
  halp_meta(description, "Estimates human pose using a YOLO-based DNN model.");
  halp_meta(uuid, "942d0486-c4d1-482c-bc80-81f6b2949037");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/ai-recognition.html#yolo-pose")

      struct
  {
    halp::fixed_texture_input<"In"> image;
    ModelPort model;
    halp::xy_spinboxes_i32<"Model input resolution", halp::range{1, 2048, 640}>
        resolution;

  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

    struct
    {
      halp_meta(name, "Detection");
      std::vector<DetectedYoloPose> value;
    } detection;
  } outputs;

  PoseDetector() noexcept;
  ~PoseDetector();

  void operator()();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;
  boost::container::vector<float> storage;
};
}
