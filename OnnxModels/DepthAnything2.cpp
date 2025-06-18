#include "DepthAnything2.hpp"

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Resnet.hpp>

namespace OnnxModels
{

DepthAnythingV2::DepthAnythingV2() noexcept
{
  inputs.image.request_height = 518;
  inputs.image.request_width = 518;
  outputs.image.create(518, 518);
  storage.reserve(518 * 518 * 3);
}

DepthAnythingV2::~DepthAnythingV2() = default;

void DepthAnythingV2::operator()()
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

  {
    const int N = out_tt[0].GetTensorTypeAndShapeInfo().GetElementCount();
    SCORE_ASSERT(N == 518 * 518);
    std::span<const float> res
        = std::span(out_tt[0].GetTensorData<float>(), N);

#pragma omp simd
    for (int i = 0; i < N; i++)
    {
      outputs.image.storage[i] = std::clamp(int(res[i] * 16.f), 0, 255);
    }
  }

  outputs.image.texture.changed = true;
  std::swap(t.storage, this->storage);
}
catch (...)
{
}
}
