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
  int keypoint;
  float x, y;
  halp_field_names(keypoint, x, y);
};

struct DetectedBlazePose
{
  std::array<Keypoint, 39> keypoints;

  halp_field_names(keypoints);
};

struct BlazePoseDetector
{
public:
  halp_meta(name, "Blaze Pose");
  halp_meta(c_name, "blazepose");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "BlazePose authors, Onnxruntime");
  halp_meta(description, "BlazePose recognizer using DNN.");
  halp_meta(uuid, "236d610b-cc61-4e4a-80f1-dccd08f8b2b0");

  struct
  {
    halp::fixed_texture_input<"In"> image;
    halp::lineedit<"Model", ""> model;
    halp::xy_spinboxes_i32<"Model input resolution", halp::range{1, 2048, 256}>
        resolution;

  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

    struct
    {
      halp_meta(name, "Detection");
      std::vector<DetectedBlazePose> value;
    } detection;
  } outputs;

  BlazePoseDetector() noexcept;
  ~BlazePoseDetector();

  void operator()();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;
};
}