#include "TRTPose.hpp"

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Trt.hpp>

namespace OnnxModels
{
TRTPoseDetector::TRTPoseDetector() noexcept
{
  inputs.image.request_height = 640;
  inputs.image.request_width = 640;
}
TRTPoseDetector::~TRTPoseDetector() = default;

void TRTPoseDetector::operator()()
{
  if (!available)
    return;

  auto& in_tex = inputs.image.texture;

  if (!in_tex.changed)
    return;

  if (!this->ctx)
  {
    this->ctx
        = std::make_unique<Onnx::OnnxRunContext>(this->inputs.model.value);
  }
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();
  auto t = tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      640,
      640,
      storage,
      false);
  Ort::Value tt[1] = {std::move(t.value)};

  assert(1 == spec.output_names_char.size());
  Ort::Value out_tt[1]{Ort::Value{nullptr}};
  ctx.infer(spec, tt, out_tt);

  static const TRT_pose pose;
  pose.processOutput(
      spec,
      out_tt,
      reinterpret_cast<std::vector<TRT_pose::pose_type>&>(
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
}
