//Copied from BlazePose.hpp

#pragma once
#include <OnnxModels/Utils.hpp>
#include <cmath>
#include <halp/controls.hpp>
#include <halp/file_port.hpp>
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

struct DetectedRTMPose
{
  std::array<Keypoint, 26> keypoints;

  halp_field_names(keypoints);
};

// detector class
struct RTMPoseDetector
{
public:
  halp_meta(name, "RTMPose");
  halp_meta(c_name, "rtmpose");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "RTMPose authors, Onnxruntime");
  halp_meta(description, "RTMPose recognizer using DNN.");
  halp_meta(uuid, "676202f7-1a6c-4bde-a389-f36835a14d7c");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/ai-recognition.html#rtmpose");
  struct
  {
    halp::fixed_texture_input<"In"> image;
    ModelPort model;
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
      std::optional<DetectedRTMPose> value;
    } detection;
  } outputs;

  RTMPoseDetector() noexcept;
  ~RTMPoseDetector();

  void operator()();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;
};
}
