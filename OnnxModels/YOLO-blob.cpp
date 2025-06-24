#include "YOLO-blob.hpp"

#include <boost/algorithm/string.hpp>

#include <QPainter>

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Yolo.hpp>
namespace OnnxModels
{

YOLO7BlobDetector::YOLO7BlobDetector() noexcept
{
  inputs.image.request_height = 640;
  inputs.image.request_width = 640;
}
YOLO7BlobDetector::~YOLO7BlobDetector() = default;

void YOLO7BlobDetector::loadClasses()
{
  if (!available)
    return;

  classes.clear();
  boost::split(
      classes, this->inputs.classes.file.bytes, boost::is_any_of("\n"));
}

void YOLO7BlobDetector::operator()()
{
  if (!available)
    return;
  if (this->inputs.model.file.bytes.empty())
    return;

  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
    return;

  if (!this->ctx)
    try
    {
      this->ctx = std::make_unique<Onnx::OnnxRunContext>(
          this->inputs.model.file.bytes);
    }
    catch (std::exception& e)
    {
      std::cerr << "Error while loading the model: " << e.what() << "\n";
      available = false;
      return;
    }
    catch (...)
    {

      std::cerr << "Error while loading the model\n";
      available = false;
      return;
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

  assert(1 == spec.output_names_char.size());
  Ort::Value out_tt[1]{Ort::Value{nullptr}};
  ctx.infer(spec, tt, out_tt);

  detector.processOutput(
      classes,
      spec,
      out_tt,
      reinterpret_cast<std::vector<Yolo::YOLO_blob::blob_type>&>(
          outputs.detection.value),
      0,
      0,
      640,
      640,
      640,
      640,
      inputs.iou_threshold,
      inputs.confidence);

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
