#include "ENet.hpp"

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Resnet.hpp>

namespace OnnxModels
{
// Tested models:
// enet_b0_8_best_afew.onnx
// enet_b0_8_best_vgaf.onnx
// from: https://github.com/sb-ai-lab/EmotiEffLib
// or: https://github.com/av-savchenko/face-emotion-recognition.git

// model shape:
// inputs:
// - tensor: float32[batch_size,3,224,224]
// outputs:
// - tensor: float32[batch_size,8]

EmotionNetDetector::EmotionNetDetector() noexcept
{
  inputs.image.request_height = 224;
  inputs.image.request_width = 224;
}

EmotionNetDetector::~EmotionNetDetector() = default;

void EmotionNetDetector::operator()()
try
{
  if (!available)
    return;

  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
    return;

  if (!this->ctx)
  {
    this->ctx
        = std::make_unique<Onnx::OnnxRunContext>(this->inputs.model.file.bytes);
  }
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();
  auto t = nchw_tensorFromARGB(
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
      reinterpret_cast<std::vector<OnnxModels::EmotionNet::recognition_type>&>(
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
}
}
