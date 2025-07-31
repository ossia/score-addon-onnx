#include "StreamDiffusion.hpp"

#include <Onnx/helpers/Images.hpp>
#include <pybind11/numpy.h>
// // clang-format off
// #include <c10/util/Exception.h>
// #include <torch/csrc/autograd/python_variable.h>
// #include <torch/torch.h>
// // clang-format on
// namespace c10::detail
// {
// void c10::detail::torchCheckFail(
//     char const*,
//     char const*,
//     unsigned int,
//     const std::string&)
// {
//   throw std::runtime_error("oh noes");
// }
//
// }

// OMP_NUM_THREADS=1
namespace PythonModels
{
static auto& interp()
{
  static py::scoped_interpreter guard{};
  py::module sys = py::module::import("sys");
  // FIXME
  sys.attr("path").attr("insert")(
      0, "/home/jcelerier/projets/oss/StreamDiffusion-uv/src");

  sys.attr("path").attr("insert")(
      1,
      "/home/jcelerier/projets/oss/StreamDiffusion-uv/.venv/lib/python3.12/"
      "site-packages");

  py::exec(R"(import os
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
# from streamdiffusion.acceleration.sfast import accelerate_with_stable_fast

)");
  return guard;
}

StreamDiffusionImg2Img::StreamDiffusionImg2Img() noexcept
{
  static auto& guard = interp();
}

StreamDiffusionImg2Img::~StreamDiffusionImg2Img() { }

void tensor_to_image(py::object tens)
{
#if 0
  // Inference
  py::object pytensor = py::eval(
      "postprocess_image("
      "  image=stream(input_image),"
      "  output_type='pt'"
      ")[0]",
      py::globals(),
      locals);

  auto& image_tensor = THPVariable_Unpack(pytensor.ptr());
  /*
  TORCH_CHECK(image_tensor.dim() == 3, "Expected a 3D tensor");
  TORCH_CHECK(image_tensor.size(0) == 3, "Expected tensor shape [3, H, W]");
  TORCH_CHECK(
      image_tensor.dtype() == torch::kUInt8, "Expected tensor dtype uint8");
*/
  at::Tensor contiguous_tensor
      = image_tensor
            .to(torch::kFloat32) // Convert to float32
            .mul(255.0)          // Scale from [0,1] to [0,255]
            .clamp(0, 255)       // Clamp values to valid byte range
            .contiguous();       // Ensure memory layout is contiguous

  auto content = contiguous_tensor.accessor<float, 3>();
  const int ocol = contiguous_tensor.size(0);
  const int oheight = contiguous_tensor.size(1);
  const int owidth = contiguous_tensor.size(2);
  SCORE_ASSERT(oheight == 512);
  SCORE_ASSERT(owidth == 512);
  SCORE_ASSERT(ocol == 3);
  qDebug("A");
  qDebug() << content[0][0][0];
  qDebug("B");
  //   std::string mode = image.attr("mode").cast<std::string>();
  //   if (mode != "RGB")
  //     return;

  // Get our image back
  //py::bytes raw_bytes = image.attr("tobytes")();

  outputs.image.create(width, height);
  float* p_i = content.data();

  auto* p_o = outputs.image.texture.bytes;
  for (int z = 0; z < height * width; ++z)
  {
    {
      qDebug("X");
      p_o[0] = content[0][0][0]; //p_i[0];
      qDebug("X1");
      p_o[1] = 0; //p_i[1];
      p_o[2] = 0; //p_i[2];
      p_o[3] = 255;
      p_o += 4;
      p_i += 3;
    }
  }
#endif
}
void StreamDiffusionImg2Img::operator()()
{
  static constexpr auto width = 512;
  static constexpr auto height = 512;
  using namespace py::literals;
  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
    return;
  if (m_needsModel)
    this->load_model();
  if (m_needsCreate)
    this->create();
  if (!m_created)
    return;
  if (m_needsPrepare)
    this->prepare();
  if (!m_prepared)
    return;

  auto img = Onnx::rescaleImage(
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      QImage::Format_RGBA8888,
      width,
      height);

  // RGBA to PIL
  auto img_array = py::array_t<uint8_t>(
      {width, height, 4}, // shape
      {width * 4, 4, 1},  // strides (row, col, channel)
      img.constBits()     // pointer to data
  );

  // Import PIL and create image in Python
  py::module_ PIL = py::module_::import("PIL.Image");
  py::object rgba_image = PIL.attr("fromarray")(img_array, "RGBA");
  py::object rgb_image = rgba_image.attr("convert")("RGB");

  // Make our image available in python
  auto locals = py::dict("input_image"_a = rgb_image);

  // Warmup. TODO compute correct length.
  if (m_needsTrain)
  {
    py::exec(
        fmt::format(
            R"(
#Warmup >= len(t_index_list) x frame_buffer_size
for _ in range(10):
    stream(input_image)
)",
            inputs.tcount.value),
        py::globals(),
        locals);
    m_needsTrain = false;
  }

  // Inference
  py::bytes raw_bytes = py::eval(
      "postprocess_image(stream(input_image))[0].tobytes()",
      py::globals(),
      locals);

  /*
  auto r = x.unchecked<3>();
  auto p_i = r.data(0, 0, 0);

  outputs.image.create(width, height);
  auto* p_o = outputs.image.texture.bytes;
  for (int pix = 0; pix < height * width; ++pix)
  {
    p_o[0] = p_i[0] * 255.f;
    p_o[1] = p_i[1] * 255.f;
    p_o[2] = p_i[2] * 255.f;
    p_o[3] = 255;
    p_i += 3;
    p_o += 4;
}
*/
  // Get our image back
  std::string_view content{raw_bytes};
  if (content.size() < width * height * 3)
    return;

  outputs.image.create(width, height);
  auto* p_i = content.data();
  auto* p_o = outputs.image.texture.bytes;
  for (int pix = 0; pix < width * height; pix++)
  {
    p_o[0] = p_i[0];
    p_o[1] = p_i[1];
    p_o[2] = p_i[2];
    p_o[3] = 255;
    p_i += 3;
    p_o += 4;
  }
  outputs.image.texture.changed = true;
}

void StreamDiffusionImg2Img::load_model()
{
  py::exec(fmt::format(
      R"(
pipe = StableDiffusionPipeline.from_pretrained("{}").to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)
# TODO:
# pipe.load_lora_weights('FirstLast/RealisticVision-LoRA-libr-0.2', weight_name='pytorch_lora_weights.safetensors')
# pipe.load_lora_weights('/home/jcelerier/projets/oss/StreamDiffusion/models/LoRA/pixels.safetensors')
)",
      inputs.model.value));
  m_needsModel = false;
}

void StreamDiffusionImg2Img::create()
{
  m_created = false;
  try
  {
    std::string count;
    count += std::to_string(inputs.t1.value);
    if (inputs.tcount.value > 1)
    {
      count += ",";
      count += std::to_string(inputs.t2.value);
    }
    if (inputs.tcount.value > 2)
    {
      count += ",";
      count += std::to_string(inputs.t3.value);
    }
    py::exec(fmt::format(
        R"(# Wrap the pipeline in StreamDiffusion
stream = None
stream = StreamDiffusion(
    pipe,
    t_index_list=[ {} ],
    torch_dtype=torch.float16,
    cfg_type="self",
)

# TODO:
# stream.load_lora("/home/jcelerier/projets/oss/StreamDiffusion/models/LoRA/pixels.safetensors", adapter_name='qsdfqsdf')
# stream.pipe.set_adapters(["lcm", "qsdfqsdf"], adapter_weights=[1.0, 1.0])

stream.load_lcm_lora("{}")
stream.fuse_lora(True, True, 1.0, False)

stream.vae = AutoencoderTiny.from_pretrained("{}").to(device=pipe.device, dtype=pipe.dtype)

stream = accelerate_with_tensorrt(
    stream, "engines", max_batch_size={},
)
)",
        count,
        inputs.lcm.value,
        inputs.vae.value,
        inputs.tcount.value));
    m_needsCreate = false;
    m_created = true;
  }
  catch (const std::exception& e)
  {
    qDebug() << e.what();
  }
}

void StreamDiffusionImg2Img::prepare()
{
  m_prepared = false;
  try
  {
    py::exec(fmt::format(
        R"(
stream.prepare(
  prompt="{}",
  negative_prompt="{}",
  num_inference_steps={},
  guidance_scale={},
  seed={}
)
)",
        this->inputs.prompt.value,
        this->inputs.negative_prompt.value,
        this->inputs.steps.value,
        this->inputs.guidance.value,
        this->inputs.seed.value));
    m_needsPrepare = false;
    m_needsTrain = true;
    m_prepared = true;
  }
  catch (const std::exception& e)
  {
    qDebug() << e.what();
  }
}
}
