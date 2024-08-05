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

namespace OnnxModels
{
struct Keypoint
{
  int kp;
  float x, y;
  halp_field_names(kp, x, y);
};

struct DetectedTRTPose
{
  std::string name;
  halp::rect2d<float> geometry;
  float probability{};
  std::vector<Keypoint> keypoints;

  halp_field_names(name, geometry, probability, keypoints);
};

struct TRTPoseDetector
{
public:
  halp_meta(name, "TRT Pose");
  halp_meta(c_name, "trt_pose");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "TRT authors, Onnxruntime, TensorRT");
  halp_meta(description, "TRT pose recognizer using DNN.");
  halp_meta(uuid, "18889004-3d30-4d19-b7ec-cfc6a00bb9d8");

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
      std::vector<DetectedTRTPose> value;
    } detection;
  } outputs;

  TRTPoseDetector() noexcept;
  ~TRTPoseDetector();

  void operator()();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;
};
}