#include "RealESRGAN.hpp"

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>

namespace OnnxModels
{
RealESRGANScaler::RealESRGANScaler() noexcept = default;
RealESRGANScaler::~RealESRGANScaler() = default;

void RealESRGANScaler::operator()()
{
  if (!available)
    return;

  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
    return;

  if (!this->ctx)
  {
    this->ctx = std::make_unique<Onnx::OnnxRunContext>(
        this->inputs.model.file.bytes);
  }
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();

  int channels = 3;

  std::vector<int64_t> input_shape
      = {1, channels, in_tex.height, in_tex.width};

  auto t = Onnx::tensorFromARGB(
      input_shape,
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      in_tex.width,
      in_tex.height,
      storage,
      false);
  Ort::Value input_tensor[1] = {std::move(t.value)};
  // Create input tensor shape (NCHW)
  // Get input name
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = ctx.session.GetInputNameAllocated(0, allocator);
  auto output_name = ctx.session.GetOutputNameAllocated(0, allocator);

  Ort::Value output_tensors[1];
  // Run inference
  ctx.session.Run(
      Ort::RunOptions{nullptr},
      spec.input_names_char.data(),
      input_tensor,
      1,
      spec.output_names_char.data(),
      output_tensors,
      1);

  // Get output tensor
  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

  // Convert CHW output back to our texture
  const int out_height = static_cast<int>(output_shape[2]);
  const int out_width = static_cast<int>(output_shape[3]);

  if (out_width < 1 || out_height < 1 || !output_data)
    return;

  outputs.image.create(out_width, out_height);
  auto ptr = outputs.image.texture.bytes;
  for (int y = 0; y < out_height; ++y)
  {
    for (int x = 0; x < out_width; ++x)
    {
      // Get RGB values from CHW output
      float r = output_data[0 * out_height * out_width + y * out_width + x];
      float g = output_data[1 * out_height * out_width + y * out_width + x];
      float b = output_data[2 * out_height * out_width + y * out_width + x];

      // Clip to [0, 1] and convert to 8-bit
      ptr[0] = std::clamp(r, 0.0f, 1.0f) * 255.0f;
      ptr[1] = std::clamp(g, 0.0f, 1.0f) * 255.0f;
      ptr[2] = std::clamp(b, 0.0f, 1.0f) * 255.0f;
      ptr[3] = 255;
      ptr += 4;
    }
  }
  outputs.image.texture.changed = true;
}
}
