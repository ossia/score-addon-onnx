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
struct DetectedYoloBlob
{
  std::string name;
  halp::rect2d<float> geometry;
  float probability{};

  halp_field_names(name, geometry, probability);
};

struct YOLOBlobDetector
{
public:
  halp_meta(name, "YOLO Blob");
  halp_meta(c_name, "yolo_blob");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "YOLO authors, Onnxruntime, TensorRT");
  halp_meta(description, "YOLO blob recognizer using DNN.");
  halp_meta(uuid, "3303df15-5774-4abc-b636-b51c9bb6d1fb");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/ai-recognition.html#yolo-blob")

      struct
  {
    halp::fixed_texture_input<"In"> image;
    halp::lineedit<"Model", ""> model;
    halp::lineedit<"Classes", ""> classes;
    halp::xy_spinboxes_i32<"Model input resolution", halp::range{1, 2048, 640}>
        resolution;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

    struct
    {
      halp_meta(name, "Detection");
      std::vector<DetectedYoloBlob> value;
    } detection;
  } outputs;

  YOLOBlobDetector() noexcept;
  ~YOLOBlobDetector();

  void operator()();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;
};
}
