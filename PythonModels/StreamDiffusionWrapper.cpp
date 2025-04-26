#include "StreamDiffusionWrapper.hpp"

#include <Onnx/helpers/Images.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace PythonModels
{

int python_interpreter_global_instance()
{
  static py::scoped_interpreter guard{};
  return 1;
}

StreamDiffusionWrapper::StreamDiffusionWrapper()
{
  static int guard = python_interpreter_global_instance();
  static int path = setup_path();
  static int imports = setup_imports();
}

int StreamDiffusionWrapper::setup_path()
{

  py::module sys = py::module::import("sys");
  // FIXME
  sys.attr("path").attr("insert")(
      0, "/home/jcelerier/projets/oss/StreamDiffusion/src");

  sys.attr("path").attr("insert")(
      1,
      "/home/jcelerier/projets/oss/StreamDiffusion/.venv/lib/python3.13/"
      "site-packages");
  return 1;
}

int StreamDiffusionWrapper::setup_imports()
{
  py::exec(R"(import os
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
# from streamdiffusion.acceleration.sfast import accelerate_with_stable_fast

)");
  return 1;
}

void StreamDiffusionWrapper::load_model()
{
  m_hasModel = false;
  try
  {
    py::exec(fmt::format(
        R"(
pipe = StableDiffusionPipeline.from_pretrained("{}").to(
    device=torch.device("cuda"),
    dtype=torch.float16
)
# TODO:
# pipe.load_lora_weights('FirstLast/RealisticVision-LoRA-libr-0.2', weight_name='pytorch_lora_weights.safetensors')
# pipe.load_lora_weights('/home/jcelerier/projets/oss/StreamDiffusion/models/LoRA/pixels.safetensors')
)",
        m_model));
    m_needsModel = false;
    m_hasModel = true;
  }
  catch (const std::exception& e)
  {
    qDebug() << e.what();
    return;
  }
}

void StreamDiffusionWrapper::create()
{
  m_created = false;

  if (m_temps.empty())
    return;
  try
  {
    std::string count;
    count += std::to_string(m_temps[0]);
    if (m_temps.size() > 1)
    {
      count += ",";
      count += std::to_string(m_temps[1]);
    }
    if (m_temps.size() > 2)
    {
      count += ",";
      count += std::to_string(m_temps[2]);
    }
    py::exec(fmt::format(
        R"(# Wrap the pipeline in StreamDiffusion
stream = None
stream = StreamDiffusion(
    pipe,
    t_index_list=[ {} ],
    torch_dtype=torch.float16,
    cfg_type="self",
    width={},
    height={}
)

# TODO:
# stream.load_lora("/home/jcelerier/projets/oss/StreamDiffusion/models/LoRA/pixels.safetensors", adapter_name='qsdfqsdf')
# stream.pipe.set_adapters(["lcm", "qsdfqsdf"], adapter_weights=[1.0, 1.0])

stream.load_lcm_lora("{}")
stream.fuse_lora(True, True, 1.0, False)

stream.vae = AutoencoderTiny.from_pretrained("{}").to(device=pipe.device, dtype=pipe.dtype)

resolutiondict = {{'engine_build_options' : {{ 'opt_image_width': {}, 'opt_image_height': {} }} }}
stream = accelerate_with_tensorrt(
    stream, "engines_{}_{}", max_batch_size={},engine_build_options=resolutiondict
)
)",
        count,
        m_width,
        m_height,
        m_lcm,
        m_vae,
        m_width,
        m_height,
        m_width,
        m_height,
        m_temps.size()));
    m_needsCreate = false;
    m_created = true;
  }
  catch (const std::exception& e)
  {
    qDebug() << e.what();
  }
}

void StreamDiffusionWrapper::prepare()
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
        m_prompt_positive,
        m_prompt_negative,
        m_steps,
        m_guidance,
        m_seed));
    m_needsPrepare = false;
    m_needsTrain = true;
    m_prepared = true;
  }
  catch (const std::exception& e)
  {
    qDebug() << e.what();
  }
}

void StreamDiffusionWrapper::img2img(
    unsigned char* in_bytes,
    int in_width,
    int in_height,
    unsigned char* out_bytes)
{
  try
  {

    using namespace py::literals;
    auto img = Onnx::rescaleImage(
        in_bytes,
        in_width,
        in_height,
        QImage::Format_RGBA8888,
        m_width,
        m_height);
    img = img.convertToFormat(QImage::Format::Format_RGB888);

    // RGBA to PIL
    auto img_array = py::array_t<uint8_t>(
        {m_width, m_height, 3}, // shape
        {m_width * 3, 3, 1},    // strides (row, col, channel)
        img.constBits()         // pointer to data
    );

    // Import PIL and create image in Python
    py::module_ PIL = py::module_::import("PIL.Image");
    py::object rgb_image = PIL.attr("fromarray")(img_array, "RGB");

    // Make our image available in python
    auto locals = py::dict("input_image"_a = rgb_image);

    // Warmup. TODO compute correct length.
    if (m_needsTrain)
    {
      py::exec(
          fmt::format(
              R"(
#Warmup >= len(t_index_list) x frame_buffer_size
for _ in range({}):
    stream(input_image)
)",
              this->m_temps.size()),
          py::globals(),
          locals);
      m_needsTrain = false;
    }

    // Inference
    py::bytes raw_bytes = py::eval(
        "postprocess_image(stream(input_image))[0].tobytes()",
        py::globals(),
        locals);

    // Get our image back
    std::string_view content{raw_bytes};

    auto* p_i = content.data();
    auto* p_o = out_bytes;
    if (std::ssize(content) < m_width * m_height * 3)
      return;

    for (int pix = 0; pix < m_width * m_height; pix++)
    {
      p_o[0] = p_i[0];
      p_o[1] = p_i[1];
      p_o[2] = p_i[2];
      p_o[3] = 255;
      p_i += 3;
      p_o += 4;
    }
  }
  catch (const std::exception& e)
  {
    qDebug() << e.what();
    m_prepared = false;
  }
}

bool StreamDiffusionWrapper::prepare_inference()
{
  if (m_needsModel)
    this->load_model();
  if (!m_hasModel)
    return false;

  if (m_needsCreate)
    this->create();
  if (!m_created)
    return false;

  if (m_needsPrepare)
    this->prepare();
  if (!m_prepared)
    return false;

  return true;
}
}
