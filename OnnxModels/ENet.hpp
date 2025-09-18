#pragma once
#include <Onnx/helpers/Resnet.hpp>
#include <OnnxModels/Utils.hpp>
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
struct ENetDetectedElement
{
  std::string name;
  float probability{};
  halp_field_names(name, probability);
};

struct EmotionNetDetector : OnnxObject
{
public:
  halp_meta(name, "EmotionNet detector");
  halp_meta(c_name, "emotionnet");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "Resnet authors, Onnxruntime");
  halp_meta(description, "Resnet recognizer using a DNN model.");
  halp_meta(uuid, "db264156-4d19-4134-9381-4a43adb57fd0");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/ai-recognition.html#resnet")

      struct
  {
    halp::texture_input<"In"> image;
    ModelPort<"Model"> model;
    halp::xy_spinboxes_i32<"Model input resolution", halp::range{1, 2048, 224}>
        resolution;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

    struct
    {
      halp_meta(name, "Detection");
      std::vector<ENetDetectedElement> value;
    } detection;
  } outputs;

  EmotionNetDetector() noexcept;
  ~EmotionNetDetector();

  void operator()();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;

  EmotionNet resnet;
  boost::container::vector<float> storage;
};
}
