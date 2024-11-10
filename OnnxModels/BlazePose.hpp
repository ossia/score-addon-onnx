#pragma once

#include <cmath>
#include <halp/controls.hpp>
#include <halp/geometry.hpp>
#include <halp/meta.hpp>
#include <halp/sample_accurate_controls.hpp>
#include <halp/texture.hpp>

#include <optional>

namespace Onnx
{
struct OnnxRunContext;
}

namespace OnnxModels
{
struct Keypoint
{
  halp::xyz_type<float> position;
  float visibility;
  float presence;

  float confidence() const noexcept { return presence; }
  halp_field_names(position, visibility, presence);
};
static_assert(sizeof(Keypoint) == 5 * sizeof(float));

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
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "BlazePose authors, Onnxruntime");
  halp_meta(description, "BlazePose recognizer using DNN.");
  halp_meta(uuid, "236d610b-cc61-4e4a-80f1-dccd08f8b2b0");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/ai-recognition.html#blazepose")

      struct
  {
    halp::fixed_texture_input<"In"> image;
    halp::lineedit<"Model", ""> model;
    halp::xy_spinboxes_i32<"Model input resolution", halp::range{1, 2048, 256}>
        resolution;
    halp::hslider_f32<"Minimum confidence", halp::range{0., 1., 0.5}>
        min_confidence;

  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

    struct
    {
      halp_meta(name, "Detection");
      std::optional<DetectedBlazePose> value;
    } detection;
  } outputs;

  BlazePoseDetector() noexcept;
  ~BlazePoseDetector();

  void operator()();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;
};
}
