#include "Resnet.hpp"

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Resnet.hpp>

namespace OnnxModels
{

ResnetDetector::ResnetDetector() noexcept
{
  inputs.image.request_height = 224;
  inputs.image.request_width = 224;
}

ResnetDetector::~ResnetDetector() = default;

void ResnetDetector::operator()()
{
  if (!available)
    return;

  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
    return;

  if (this->inputs.model.file.bytes.empty())
    return;

  try
  {
    if (!this->ctx)
    {
      this->ctx = std::make_unique<Onnx::OnnxRunContext>(
          this->inputs.model.file.bytes);
    }
    auto& ctx = *this->ctx;
    auto spec = ctx.readModelSpec();
    auto t = Onnx::tensorFromARGB(
        spec.inputs[0].shape,
        in_tex.bytes,
        in_tex.width,
        in_tex.height,
        this->inputs.resolution.value.x,
        this->inputs.resolution.value.y,
        storage,
        true);
    Ort::Value tt[1] = {std::move(t.value)};

    assert(1 == spec.output_names_char.size());
    Ort::Value out_tt[1]{Ort::Value{nullptr}};
    ctx.infer(spec, tt, out_tt);

    outputs.detection.value.clear();
    resnet.processOutput(
        spec,
        out_tt,
        reinterpret_cast<std::vector<OnnxModels::Resnet::recognition_type>&>(
            outputs.detection.value));

    outputs.image.texture
        = {.bytes = in_tex.bytes,
           .width = in_tex.width,
           .height = in_tex.height,
           .changed = true};
    std::swap(storage, t.storage);
  }
  catch (const std::exception& e)
  {
    qDebug() << e.what();
  }
}
}
