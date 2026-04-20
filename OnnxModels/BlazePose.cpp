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
  // inputs.image.request_height = 256;
  // inputs.image.request_width = 256;
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

  if (!this->lm_ctx)
  {
    this->lm_ctx = std::make_unique<Onnx::OnnxRunContext>(
        this->inputs.lm_model.file.bytes);
  }

  auto& det_ctx = *this->det_ctx;
  auto det_spec = det_ctx.readModelSpec();
  auto det_t = nhwc_rgb_tensorFromRGBA(
      det_spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      224,
      224,
      det_storage);
  Ort::Value det_tensor_inputs[1] = {std::move(det_t.value)};

  qDebug() << "image height:" << in_tex.height << "width:" << in_tex.width;

  assert(
      1 <= det_spec.output_names_char.size()
      && det_spec.output_names_char.size() <= 5);

  Ort::Value det_tensor_outputs[2]{
      Ort::Value{nullptr},
      Ort::Value{nullptr}};
  det_ctx.infer(
      det_spec,
      det_tensor_inputs,
      std::span<Ort::Value>(det_tensor_outputs, det_spec.output_names_char.size()));

  qDebug() << "Output values: " << det_tensor_outputs[0].IsTensor() << " "
           << det_tensor_outputs[0].HasValue() << " "
           << det_tensor_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

  qDebug() << "Output values: " << det_tensor_outputs[1].IsTensor() << " "
           << det_tensor_outputs[1].HasValue() << " "
           << det_tensor_outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();

  std::optional<Blazepose::BlazePose_alignment::pose_align> det_out;
  Blazepose::BlazePose_alignment::processOutput(det_spec, det_tensor_outputs, det_out, in_tex.width, in_tex.height);

  if (det_out)
  {
    auto M = Blazepose::BlazePose_alignment::getROITransform(det_out->detections.front());

    // here I'll put some image transformation code that should eventually make 
    // its way into the helpers, but for now I want to test the output of the model
    QImage cropped(256, 256, QImage::Format_RGBA8888);
    cropped.fill(Qt::black);
    QPainter p(&cropped);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    p.setTransform(M);
    p.drawImage(0, 0, QImage(in_tex.bytes, in_tex.width, in_tex.height, QImage::Format_RGBA8888));
    
    auto& lm_ctx = *this->lm_ctx;
    auto lm_spec = lm_ctx.readModelSpec();
    auto lm_t = nhwc_rgb_tensorFromRGBA(
        lm_spec.inputs[0],
        cropped.constBits(),
        cropped.width(),
        cropped.height(),
        256,
        256,
        lm_storage);
    Ort::Value lm_tensor_inputs[1] = {std::move(lm_t.value)};

    qDebug() << "image height:" << cropped.height() << "width:" << cropped.width();

    assert(
        1 <= lm_spec.output_names_char.size()
        && lm_spec.output_names_char.size() <= 5);

    Ort::Value lm_tensor_outputs[5]{
        Ort::Value{nullptr},
        Ort::Value{nullptr},
        Ort::Value{nullptr},
        Ort::Value{nullptr},
        Ort::Value{nullptr}};
    lm_ctx.infer(
        lm_spec,
        lm_tensor_inputs,
        std::span<Ort::Value>(lm_tensor_outputs, lm_spec.output_names_char.size()));

    std::optional<Blazepose::BlazePose_fullbody::pose_data> lm_out;
    Blazepose::BlazePose_fullbody::processOutput(lm_spec, lm_tensor_outputs, lm_out);
    auto orig_img_out = Blazepose::BlazePose_fullbody::transformToOriginalImage(lm_out, M);

    qDebug() << "Output values: " << lm_tensor_outputs[0].IsTensor() << " "
          << lm_tensor_outputs[0].HasValue() << " "
          << lm_tensor_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (lm_out)
    {
      auto orig_img_out = Blazepose::BlazePose_fullbody::transformToOriginalImage(lm_out, M);
      static_assert(sizeof(*lm_out) == sizeof(*outputs.detection.value));
      outputs.detection.value.emplace();
      memcpy((void*)&*outputs.detection.value, (void*)&*lm_out, sizeof(*lm_out));
      auto img = Onnx::drawKeypoints(
          inputs.image.texture.bytes,
          in_tex.width,
          in_tex.height,
          std::pow(inputs.min_confidence, 6),
          orig_img_out.keypoints);

      outputs.image.create(in_tex.width, in_tex.height);
      memcpy(
          outputs.image.texture.bytes,
          img.constBits(),
          in_tex.width * in_tex.height * 4);
      outputs.image.texture.changed = true;
      std::swap(det_storage, det_t.storage);
      std::swap(lm_storage, lm_t.storage);

    }
  }
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
