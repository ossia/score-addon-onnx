#pragma once

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
struct PoseDetector
{
public:
  halp_meta(name, "YOLO Pose");
  halp_meta(c_name, "yolov8_pose");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "YOLOv8 authors, Onnxruntime, TensorRT");
  halp_meta(description, "YOLO pose recognizer using DNN.");
  halp_meta(uuid, "115e2f43-2ec5-477f-bbe8-bc1c592850c7");

  struct
  {
    halp::fixed_texture_input<"In"> image;
    halp::lineedit<"Model", ""> model;
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
};
}
