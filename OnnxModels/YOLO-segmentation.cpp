#include "YOLO-segmentation.hpp"

#include <boost/algorithm/string.hpp>

#include <QPainter>

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Yolo.hpp>
namespace OnnxModels
{

YOLO8Segmentation::YOLO8Segmentation() noexcept
{
  inputs.image.request_height = 640;
  inputs.image.request_width = 640;
}
YOLO8Segmentation::~YOLO8Segmentation() = default;

void YOLO8Segmentation::loadClasses()
{
  if (!available)
    return;

  classes.clear();
  boost::split(
      classes, this->inputs.classes.file.bytes, boost::is_any_of("\n"));
}

void YOLO8Segmentation::operator()()
try
{
  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
    return;
  if (inputs.model.current_model_invalid)
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
      this->inputs.resolution.value.x,
      this->inputs.resolution.value.y,
      storage,
      {0.f, 0.f, 0.f},
      {255.f, 255.f, 255.f});
  Ort::Value tt[1] = {std::move(t.value)};

  assert(2 == spec.output_names_char.size());
  Ort::Value out_tt[2]{Ort::Value{nullptr}, Ort::Value{nullptr}};
  ctx.infer(spec, tt, out_tt);

  std::vector<Yolo::YOLO_segmentation::segmentation_type> ttt;
  detector.processOutput(
      classes,
      out_tt,
      ttt,
      640,
      640,
      640,
      640,
      inputs.iou_threshold,
      inputs.confidence);

  auto img = Onnx::drawBlobAndSegmentation(
      inputs.image.texture.bytes, in_tex.width, in_tex.height, ttt);

  outputs.image.create(in_tex.width, in_tex.height);
  memcpy(
      outputs.image.texture.bytes,
      img.bits(),
      in_tex.width * in_tex.height * 4);
  outputs.image.texture.changed = true;
  std::swap(storage, t.storage);
}
catch (...)
{
  inputs.model.current_model_invalid = true;
}
}
