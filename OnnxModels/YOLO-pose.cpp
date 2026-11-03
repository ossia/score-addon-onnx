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
  const auto& spec = ctx.readModelSpec();
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
  thread_local std::vector<Yolo::YOLO_pose::pose_type> poses;
  pose.processOutput(spec, out_tt, poses);
  auto& out = outputs.detection.value;
  out.resize(poses.size());
  for(std::size_t i = 0; i < poses.size(); i++)
  {
    const auto& p = poses[i];
    out[i].name = p.name;
    out[i].geometry = {p.geometry.x, p.geometry.y, p.geometry.w, p.geometry.h};
    out[i].probability = p.confidence;
    out[i].keypoints.resize(p.keypoints.size());
    for(std::size_t k = 0; k < p.keypoints.size(); k++)
      out[i].keypoints[k] = {p.keypoints[k].kp, p.keypoints[k].x, p.keypoints[k].y};
  }

  auto img = Onnx::drawRects(
      inputs.image.texture.bytes,
      in_tex.width,
      in_tex.height,
      outputs.detection.value);

  outputs.image.create(in_tex.width, in_tex.height);
  memcpy(
      outputs.image.texture.bytes,
      img.pixels.data(),
      in_tex.width * in_tex.height * 4);
  outputs.image.texture.changed = true;
  std::swap(storage, t.storage);
}
catch (...)
{
  inputs.model.current_model_invalid = true;
}
}
