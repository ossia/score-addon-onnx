#include "YOLO-pose.hpp"

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Yolo.hpp>

namespace OnnxModels::Yolo
{
PoseDetector::PoseDetector() noexcept
{
  inputs.image.request_height = 640;
  inputs.image.request_width = 640;
}
PoseDetector::~PoseDetector() = default;

void PoseDetector::operator()()
try
{
  if (!available)
    return;
  if (inputs.model.current_model_invalid)
    return;

  auto& in_tex = inputs.image.texture;

  if (!in_tex.changed)
    return;
   if (!in_tex.bytes)
     return;
   if (in_tex.width != 640 || in_tex.height != 640)
     return;

  if (!this->ctx)
  {
    this->ctx
        = std::make_unique<Onnx::OnnxRunContext>(this->inputs.model.file.bytes);
  }
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();
  auto t = nchw_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      640,
      640,
      storage,
      {0.f, 0.f, 0.f},
      {255.f, 255.f, 255.f});
  Ort::Value tt[1] = {std::move(t.value)};

  assert(1 == spec.output_names_char.size());
  Ort::Value out_tt[1]{Ort::Value{nullptr}};
  ctx.infer(spec, tt, out_tt);

  static const Yolo::YOLO_pose pose;
  pose.processOutput(
      spec,
      out_tt,
      reinterpret_cast<std::vector<Yolo::YOLO_pose::pose_type>&>(
          outputs.detection.value));

  auto img = Onnx::drawRects(
      inputs.image.texture.bytes,
      in_tex.width,
      in_tex.height,
      outputs.detection.value);

  outputs.image.create(in_tex.width, in_tex.height);
  memcpy(
      outputs.image.texture.bytes,
      img.constBits(),
      in_tex.width * in_tex.height * 4);
  outputs.image.texture.changed = true;
  std::swap(storage, t.storage);
}
catch (...)
{
  inputs.model.current_model_invalid = true;
}
}
