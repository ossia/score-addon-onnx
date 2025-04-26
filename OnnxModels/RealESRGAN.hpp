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

namespace OnnxModels
{
struct RealESRGANScaler : OnnxObject
{
public:
  halp_meta(name, "Realesrgan 2x");
  halp_meta(c_name, "realesrgan");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "RealESRGAN authors, Onnxruntime");
  halp_meta(description, "RealESRGAN image scaler");
  halp_meta(uuid, "75b17e98-894e-4730-90e1-84804986294d");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/realesrgan.html")

      struct
  {
    halp::texture_input<"In"> image;
    ModelPort model;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

  } outputs;

  RealESRGANScaler() noexcept;
  ~RealESRGANScaler();

  void operator()();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;

  boost::container::vector<float> storage;
};
}
