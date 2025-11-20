#include "StreamDiffusionWrapper.hpp"

#include <Library/LibrarySettings.hpp>

#include <ossia/detail/fmt.hpp>

#include <QDebug>
#include <QDir>
#include <QProcess>

#include <Onnx/helpers/Images.hpp>
#include <PythonModels/PromptParser.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/subinterpreter.h>
namespace py = pybind11;

namespace PythonModels
{

QString uvPath()
{
    return "/home/jcelerier/.local/bin";
  return score::AppContext()
             .settings<Library::Settings::Model>()
             .getPackagesPath()
         + QDir::separator() + "uv";
}
QString StreamDiffusionPath()
{
  return "/home/jcelerier/projets/oss/StreamDiffusion";
  return score::AppContext()
             .settings<Library::Settings::Model>()
             .getPackagesPath()
         + QDir::separator() + "streamdiffusion";
}

int python_interpreter_global_instance()
{
  static py::scoped_interpreter guard{};
  return 1;
}

StreamDiffusionWrapper::StreamDiffusionWrapper() { }

int StreamDiffusionWrapper::init()
{

  return 1;
}
int StreamDiffusionWrapper::setup_path()
{
  py::module sys = py::module::import("sys");
  // FIXME
  auto sd_path = StreamDiffusionPath();
  if (QDir{sd_path}.isEmpty())
    throw std::runtime_error("StreamDiffusion not downloaded!");

  QProcessEnvironment penv = QProcessEnvironment::systemEnvironment();

  if (!QFile::exists("/usr/bin/uv"))
  {
    auto uv_path = uvPath();
    if (QDir{uv_path}.isEmpty())
      throw std::runtime_error("uv not downloaded!");

    auto path = penv.value("PATH");
#if defined(_WIN32)
#define PATH_SEPARATOR_CHAR ";"
#else
#define PATH_SEPARATOR_CHAR ":"
#endif

  path.prepend(uv_path + PATH_SEPARATOR_CHAR);
  penv.insert("PATH", path);
  }

  // uv venv
  // source .venv/bin/activate
  // uv venv sync

  QProcess uv_venv;
  uv_venv.setProgram("uv");
  uv_venv.setArguments({"venv", "--python", "3.13"});
  uv_venv.setProcessEnvironment(penv);
  uv_venv.setWorkingDirectory(sd_path);
  uv_venv.start();

  QProcess uv_sync;
  uv_venv.setProgram("uv");
  uv_venv.setArguments({"sync"});
  uv_venv.setProcessEnvironment(penv);
  uv_venv.setWorkingDirectory(sd_path);
  uv_venv.start();

  sys.attr("path").attr("insert")(0, QString(sd_path + "/src").toStdString());
  sys.attr("path").attr("insert")(
      1,
      "/home/jcelerier/projets/oss/StreamDiffusion/src/streamdiffusion/cuda");
  sys.attr("path").attr("insert")(
      2,
      QString(sd_path + "/.venv/lib/python3.13/site-packages").toStdString());

  return 1;
}

int StreamDiffusionWrapper::setup_imports()
{
  py::exec(R"(import os)");
  py::exec("import torch");
  py::exec(
      "from diffusers import AutoencoderTiny, StableDiffusionPipeline, "
      "StableDiffusionXLPipeline, UNet2DConditionModel");
  py::exec("from diffusers.utils import load_image");

  py::exec("from streamdiffusion import StreamDiffusion");
  py::exec("from streamdiffusion.image_utils import postprocess_image");
  py::exec(
      "from streamdiffusion.acceleration.tensorrt import "
      "accelerate_with_tensorrt");
  py::exec(
      "#from streamdiffusion.acceleration.sfast import "
      "accelerate_with_stable_fast");

  return 1;
}

void StreamDiffusionWrapper::load_model()
{
  m_hasModel = false;
  try
  {
    py::exec(fmt::format(
        R"(
sdxl = "xl" in "{0}"
turbo = "turbo" in "{0}"
if(sdxl):
    # if(not turbo):
    #     unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")

    pipe = StableDiffusionXLPipeline.from_pretrained("{0}", unet=unet).to(
        device=torch.device("cuda"),
        dtype=torch.float16
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained("{0}").to(
        device=torch.device("cuda"),
        dtype=torch.float16
    )
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
    using namespace std::literals;
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
    py::exec(R"(
from streamdiffusion.cuda import is_cuda_available

print("Loading cuda")
if not is_cuda_available():
  print("ERROR: CUDA operations library is not available!")
  raise RuntimeError("CUDA library not loaded")
else:
  print("SUCCESS: CUDA library loaded and available")
  )");
    py::exec(
        fmt::format(
            R"(# Wrap the pipeline in StreamDiffusion
print("Loading SteamDiffusion")

stream = None
stream = StreamDiffusion(
    pipe,
    t_index_list=[ {0} ],
    torch_dtype=torch.float16,
    cfg_type="{1}",
    do_add_noise={2},
    width={3},
    height={4},
    use_denoising_batch={5},
    use_cuda_native=True
)
print("Loaded!")
print(stream)

# if(sdxl and turbo):
#     stream.load_lcm_lora("openskyml/lcm-lora-sdxl-turbo")
#     stream.fuse_lora()
# elif(sdxl and not turbo):
#     stream.load_lcm_lora("latent-consistency/lcm-sdxl")
#     stream.fuse_lora()
# elif(not sdxl and turbo):
#     print("no lora to load")
# elif((not sdxl) and (not turbo)):
#     stream.load_lcm_lora("latent-consistency/lcm-lora-sdv1-5")
#     stream.fuse_lora()

#
# sd-turbo does not need lcm-lora

# if hasattr(stream.unet, 'config'):
#    stream.unet.config.addition_embed_type = None
# if device_ids is not None:
#     stream.unet = torch.nn.DataParallel(
#         stream.unet, device_ids=device_ids
#     )

)",
            count,
            m_cfg,
            m_add_noise ? "True"sv : "False"sv,
            m_width,
            m_height,
            m_denoising_batch ? "True"sv : "False"sv));

    qDebug()<<m_lora_weight;
    if (!m_sd_turbo)
    {
      for (const auto& [k, v] : m_loras)
      {
        if (!v.empty())
        {
          py::exec(fmt::format(
              R"(
stream.load_lora('{}',  weight_name='{}')
#stream.fuse_lora(True, True, 1.0, False)
stream.fuse_lora(fuse_unet=True, fuse_text_encoder=True, lora_scale={}, safe_fusing=False)
stream.pipe.unload_lora_weights()
)",
              k,
              v, m_lora_weight * 2.5));
          qDebug()<<"Loading lora: " << k.c_str() << v.c_str() << m_lora_weight;
        }
        else
        {
          py::exec(fmt::format(
              R"(
stream.load_lora('{}')
stream.fuse_lora(fuse_unet=True, fuse_text_encoder=True, lora_scale={}, safe_fusing=False)
stream.pipe.unload_lora_weights()
)",
              k, m_lora_weight * 2.5));
        }
      }
    }

    py::exec(
        fmt::format(
            R"(
stream.vae = AutoencoderTiny.from_pretrained("{2}").to(device=pipe.device, dtype=pipe.dtype)

resolutiondict = {{ 'opt_image_width': {0}, 'opt_image_height': {1}, 'opt_batch_size': 2 }} 
#if((not sdxl) and (not turbo)):
# stream = accelerate_with_tensorrt(stream, "engines_test_sd_turbo_{0}_{1}", max_batch_size={3},engine_build_options=resolutiondict)
# stream.pipe.enable_xformers_memory_efficient_attention()
)",
            m_width,
            m_height,
            m_vae,
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
  seed={},
  delta={}
)
)",
        m_prompt_positive,
        m_prompt_negative,
        m_steps,
        m_guidance,
        m_seed,
        m_delta));
    m_needsPrepare = false;
    m_needsTrain = true;
    m_needsUpdatePrompt = true;
    m_prepared = true;
  }
  catch (const std::exception& e)
  {
    qDebug() << e.what();
  }
}

void StreamDiffusionWrapper::update_prompt()
{
  try
  {
    if (auto weights = parse_input_string(m_prompt_positive))
    {
      std::string prompt;
      for (const auto& [k, v] : *weights)
      {
        prompt += fmt::format("({}, {}), ", quote_string_for_python(k), v);
      }
      if (prompt.ends_with(','))
        prompt.pop_back();
      py::exec(
          fmt::format("stream.update_prompts(weighted_prompts=[{}])", prompt));
    }
    else
    {
      py::exec(fmt::format(
          "stream.update_prompts(weighted_prompts=[({}, 1.0)])",
          quote_string_for_python(m_prompt_positive)));
    }

    m_needsUpdatePrompt = false;
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

    // RGBA to PIL
    auto img_array = py::array_t<uint8_t>(
        {m_width, m_height, 3}, // shape
        {m_width * 3, 3, 1},    // strides (row, col, channel)
        in_bytes                // pointer to data
    );

    // Import PIL and create image in Python
    py::module_ PIL = py::module_::import("PIL.Image");
    py::object rgb_image = PIL.attr("fromarray")(img_array, "RGB");

    // Make our image available in python
    auto locals = py::dict("input_image"_a = rgb_image);

    // Warmup.
    if (m_needsTrain)
    {
      py::exec(
          fmt::format(
              R"(
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
    const auto vv = std::string_view(raw_bytes);
    memcpy(out_bytes, vv.data(), vv.size());
    // read_image_pil(raw_bytes, out_bytes);
  }
  catch (const std::exception& e)
  {
    qDebug() << e.what();
    m_prepared = false;
  }
}

void StreamDiffusionWrapper::txt2img(unsigned char* out_bytes)
{
  try
  {
    using namespace py::literals;

    // Warmup.
    if (m_needsTrain)
    {
      py::exec(
          fmt::format(
              R"(
for _ in range({}):
    stream()
)",
              this->m_temps.size()),
          py::globals());
      m_needsTrain = false;
    }

    // Inference
    py::bytes raw_bytes;
    if (m_model.find("turbo") != std::string::npos)
    {
      raw_bytes = py::eval(
          "postprocess_image(stream.txt2img_sd_turbo(1))[0].tobytes()",
          py::globals());
    }
    else
    {
      raw_bytes = py::eval(
          "postprocess_image(stream.txt2img(1))[0].tobytes()", py::globals());
    }

    // Get our image back
    read_image_pil(raw_bytes, out_bytes);
  }
  catch (const std::exception& e)
  {
    qDebug() << e.what();
    m_prepared = false;
  }
}

bool StreamDiffusionWrapper::prepare_inference()
{
  std::lock_guard _{m_mtx};
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

  if (m_needsUpdatePrompt)
    this->update_prompt();
  return true;
}

void StreamDiffusionWrapper::read_image_pil(
    std::string_view content,
    unsigned char* out_bytes)
{
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
}
