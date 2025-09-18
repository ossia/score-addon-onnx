#pragma once
#include <Onnx/helpers/Yolo.hpp>
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
struct YOLO8Segmentation : OnnxObject
{
public:
  halp_meta(name, "YOLO Segmentation");
  halp_meta(c_name, "YOLO Segmentation");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "YOLO authors, Onnxruntime");
  halp_meta(description, "Segments objects using a YOLO model.");
  halp_meta(uuid, "5ab19152-bb7c-4a83-a9aa-ca3021aea528");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/"
      "ai-recognition.html#yolo-segmentation")

      struct
  {
    halp::fixed_texture_input<"In"> image;
    ModelPort<"Model"> model;
    struct : halp::file_port<"Classes">
    {
      void update(YOLO8Segmentation& self) { self.loadClasses(); }
    } classes;
    halp::xy_spinboxes_i32<"Model input resolution", halp::range{1, 2048, 640}>
        resolution;
    halp::knob_f32<"Min. confidence"> confidence;
    halp::knob_f32<"IoU threshold"> iou_threshold;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

  } outputs;

  YOLO8Segmentation() noexcept;
  ~YOLO8Segmentation();

  void operator()();

  void loadClasses();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;
  std::vector<std::string> classes;
  boost::container::vector<float> storage;
  Yolo::YOLO_segmentation detector;
};
}
