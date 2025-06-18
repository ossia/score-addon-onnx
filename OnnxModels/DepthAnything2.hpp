#pragma once
#include <OnnxModels/Utils.hpp>
#include <Onnx/helpers/Resnet.hpp>
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
struct DepthAnythingV2 : OnnxObject
{
public:
  halp_meta(name, "Depth Anything v2");
  halp_meta(c_name, "emotionnet");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "Depth Anything authors, Onnxruntime");
  halp_meta(description, "Estimate depth from an RGB image.");
  halp_meta(uuid, "66366f10-3d67-4980-8d7b-a9cc2938c276");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/"
      "ai-recognition.html#depth-anything")

      struct
  {
    halp::fixed_texture_input<"In"> image;
    ModelPort model;
    halp::xy_spinboxes_i32<"Model input resolution", halp::range{1, 2048, 518}>
        resolution;
  } inputs;

  struct
  {
    halp::texture_output<"Out", halp::r8_texture> image;
  } outputs;

  DepthAnythingV2() noexcept;
  ~DepthAnythingV2();

  void operator()();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;
  boost::container::vector<float> storage;
};
}
