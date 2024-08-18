#include "BlazePose.hpp"

#include <Onnx/helpers/BlazePose.hpp>
#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>

namespace OnnxModels
{
BlazePoseDetector::BlazePoseDetector() noexcept
{
  inputs.image.request_height = 256;
  inputs.image.request_width = 256;
}
BlazePoseDetector::~BlazePoseDetector() = default;

void BlazePoseDetector::operator()()
{
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
  auto t = nhwc_rgb_tensorFromRGBA(
      spec.inputs[0], in_tex.bytes, in_tex.width, in_tex.height, 256, 256);
  Ort::Value tensor_inputs[1] = {std::move(t.value)};

  assert(
      1 <= spec.output_names_char.size()
      && spec.output_names_char.size() <= 5);

  Ort::Value tensor_outputs[5]{
      Ort::Value{nullptr},
      Ort::Value{nullptr},
      Ort::Value{nullptr},
      Ort::Value{nullptr},
      Ort::Value{nullptr}};
  ctx.infer(
      spec,
      tensor_inputs,
      std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()));

  std::optional<Blazepose::BlazePose_fullbody::pose_data> out;
  Blazepose::BlazePose_fullbody::processOutput(spec, tensor_outputs, out);

  if (out)
  {
    static_assert(sizeof(*out) == sizeof(*outputs.detection.value));
    outputs.detection.value.emplace();
    memcpy((void*)&*outputs.detection.value, (void*)&*out, sizeof(*out));
    auto img = Onnx::drawKeypoints(
        inputs.image.texture.bytes,
        in_tex.width,
        in_tex.height,
        std::pow(inputs.min_confidence, 6),
        out->keypoints);

    outputs.image.create(in_tex.width, in_tex.height);
    memcpy(
        outputs.image.texture.bytes,
        img.constBits(),
        in_tex.width * in_tex.height * 4);
    outputs.image.texture.changed = true;
  }
}

}
