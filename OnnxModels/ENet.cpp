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

EmotionNetDetector::EmotionNetDetector() noexcept { }

EmotionNetDetector::~EmotionNetDetector() = default;

void EmotionNetDetector::operator()()
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
  // The model's own input size; the resolution knob only sizes dynamic
  // exports. FER+ (1 channel) takes raw 0-255 luma, the EmotiEffLib models
  // ImageNet-normalised RGB.
  const auto [mw, mh] = nchwInputSize(
      spec.inputs[0], this->inputs.resolution.value.x,
      this->inputs.resolution.value.y);
  const bool gray
      = spec.inputs[0].shape.size() == 4 && spec.inputs[0].shape[1] == 1;
  auto t = nchw_tensorFromRGBA(
      spec.inputs[0], in_tex.bytes, in_tex.width, in_tex.height, mw, mh, storage,
      gray ? std::array<float, 3>{0.f, 0.f, 0.f}
           : std::array<float, 3>{255.f * 0.485f, 255.f * 0.456f, 255.f * 0.406f},
      gray ? std::array<float, 3>{1.f, 1.f, 1.f}
           : std::array<float, 3>{255.f * 0.229f, 255.f * 0.224f, 255.f * 0.225f});
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
  outputs.detection.value.clear();
}
}
