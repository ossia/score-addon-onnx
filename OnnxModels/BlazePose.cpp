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
  Ort::Value tt[1] = {std::move(t.value)};

  assert(1 <= spec.output_names_char.size());
  Ort::Value out_tt[3]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}};
  ctx.infer(spec, tt, out_tt);

  std::optional<BlazePose_fullbody::pose_data> out;
  BlazePose_fullbody::processOutput(spec, out_tt, out);

  if (out)
  {
    auto img = Onnx::drawKeypoints(
        inputs.image.texture.bytes,
        in_tex.width,
        in_tex.height,
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
