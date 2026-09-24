#include "Resnet.hpp"

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Resnet.hpp>

namespace OnnxModels
{

ResnetDetector::ResnetDetector() noexcept { }

ResnetDetector::~ResnetDetector() = default;

void ResnetDetector::operator()()
try
{
  if (!available)
    return;
  if (inputs.model.current_model_invalid)
    return;

  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
    return;

  // Recreate the session when the model file changes: keyed on "no ctx yet"
  // only, a model swap kept running the previous model. Only a failed load
  // marks the model invalid; a per-frame failure (e.g. a resolution the model
  // rejects) just skips that frame.
  if (!this->ctx || lastModelPath != this->inputs.model.file.filename)
  {
    this->ctx.reset();
    try
    {
      this->ctx
          = std::make_unique<Onnx::OnnxRunContext>(
          this->inputs.model.file.bytes, this->inputs.model.file.filename);
      lastModelPath = this->inputs.model.file.filename;
    }
    catch (...)
    {
      inputs.model.current_model_invalid = true;
      return;
    }
  }
  auto& ctx = *this->ctx;
  const auto& spec = ctx.readModelSpec();
  auto t = nchw_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      this->inputs.resolution.value.x,
      this->inputs.resolution.value.y,
      storage,
      {255.f * 0.485f, 255.f * 0.456f, 255.f * 0.406f},
      {255.f * 0.229f, 255.f * 0.224f, 255.f * 0.225f});
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
catch (...)
{
  outputs.detection.value.clear();
}
}
