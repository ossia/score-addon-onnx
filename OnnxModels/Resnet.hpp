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
struct DetectedElement
{
  std::string name;
  float probability{};
  halp_field_names(name, probability);
};

struct ResnetDetector
{
public:
  halp_meta(name, "Resnet detector");
  halp_meta(c_name, "resnet");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "Resnet authors, Onnxruntime");
  halp_meta(description, "Resnet recognizer using DNN.");
  halp_meta(uuid, "9f4b7448-5bb5-4db6-8c96-57d53896208b");

  struct
  {
    halp::fixed_texture_input<"In"> image;
    halp::lineedit<"Model", ""> model;
    halp::lineedit<"Classes", ""> classes;
    halp::xy_spinboxes_i32<"Model input resolution", halp::range{1, 2048, 224}>
        resolution;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

    struct
    {
      halp_meta(name, "Detection");
      std::vector<DetectedElement> value;
    } detection;
  } outputs;

  ResnetDetector() noexcept;
  ~ResnetDetector();

  void operator()();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;
};
}
