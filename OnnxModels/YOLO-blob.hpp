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
struct DetectedYoloBlob
{
  std::string name;
  halp::rect2d<float> geometry;
  float probability{};

  halp_field_names(name, geometry, probability);
};

struct YOLO7BlobDetector : OnnxObject
{
public:
  halp_meta(name, "YOLOv7 Blob");
  halp_meta(c_name, "yolov7_blob");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "YOLO authors, Kin-Yiu Wong, Onnxruntime");
  halp_meta(description, "Identifies objects using a YOLOv7 blob model.");
  halp_meta(uuid, "3303df15-5774-4abc-b636-b51c9bb6d1fb");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/ai-recognition.html#yolo-blob")

      struct
  {
    halp::fixed_texture_input<"In"> image;
    ModelPort model;
    struct : halp::file_port<"Classes">
    {
      void update(YOLO7BlobDetector& self) { self.loadClasses(); }
    } classes;
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

  YOLO7BlobDetector() noexcept;
  ~YOLO7BlobDetector();

  void operator()();

  void loadClasses();

private:
  std::unique_ptr<Onnx::OnnxRunContext> ctx;
  std::vector<std::string> classes;
};
}
