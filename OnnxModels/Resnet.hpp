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
struct DetectedElement
{
  std::string name;
  float probability{};
  halp_field_names(name, probability);
};

struct ResnetDetector : OnnxObject
{
public:
  halp_meta(name, "Resnet detector");
  halp_meta(c_name, "resnet");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "Resnet authors, Onnxruntime");
  halp_meta(description, "Resnet recognizer using a DNN model.");
  halp_meta(uuid, "9f4b7448-5bb5-4db6-8c96-57d53896208b");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/ai-recognition.html#resnet")

      struct
  {
    halp::texture_input<"In"> image;
    ModelPort model;

    struct : halp::file_port<"Classes">
    {
      void update(ResnetDetector& self)
      {
        self.resnet.loadClasses(this->file.filename);
      }
    } classes;
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

  Resnet resnet;
  boost::container::vector<float> storage;
};
}
