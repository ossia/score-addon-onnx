#include "BlazePose.hpp"

#include <QApplication>
#include <QImage>

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
try
{
  if (!available)
    return;

  if (this->inputs.det_model.current_model_invalid)
    return;

  if (this->inputs.lm_model.current_model_invalid)
    return;

  auto& in_tex = inputs.image.texture;

  if (!in_tex.changed)
    return;

  if (!this->det_ctx)
  {
    this->det_ctx = std::make_unique<Onnx::OnnxRunContext>(
        this->inputs.det_model.file.bytes);
  }

  auto& det_ctx = *this->det_ctx;
  auto spec = det_ctx.readModelSpec();
  auto t = nhwc_rgb_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      224,
      224,
      storage);
  Ort::Value tensor_inputs[1] = {std::move(t.value)};

  qDebug() << "image height:" << in_tex.height << "width:" << in_tex.width;

  assert(
      1 <= spec.output_names_char.size()
      && spec.output_names_char.size() <= 5);

  Ort::Value tensor_outputs[2]{
      Ort::Value{nullptr},
      Ort::Value{nullptr}};
  det_ctx.infer(
      spec,
      tensor_inputs,
      std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()));

  qDebug() << "Output values: " << tensor_outputs[0].IsTensor() << " "
           << tensor_outputs[0].HasValue() << " "
           << tensor_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

  qDebug() << "Output values: " << tensor_outputs[1].IsTensor() << " "
           << tensor_outputs[1].HasValue() << " "
           << tensor_outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();
  //std::optional<Blazepose::BlazePose_fullbody::pose_data> out;
  //Blazepose::BlazePose_fullbody::processOutput(spec, tensor_outputs, out);

  // if (out)
  // {
  //   static_assert(sizeof(*out) == sizeof(*outputs.detection.value));
  //   outputs.detection.value.emplace();
  //   memcpy((void*)&*outputs.detection.value, (void*)&*out, sizeof(*out));
  //   auto img = Onnx::drawKeypoints(
  //       inputs.image.texture.bytes,
  //       in_tex.width,
  //       in_tex.height,
  //       std::pow(inputs.min_confidence, 6),
  //       out->keypoints);

  //   outputs.image.create(in_tex.width, in_tex.height);
  //   memcpy(
  //       outputs.image.texture.bytes,
  //       img.constBits(),
  //       in_tex.width * in_tex.height * 4);
  //   outputs.image.texture.changed = true;
  //   std::swap(storage, t.storage);
  // }
}
catch (...)
{
  inputs.det_model.current_model_invalid = true;
  inputs.lm_model.current_model_invalid = true;
}
}
