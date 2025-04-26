#include "StreamDiffusion_img2img.hpp"

namespace PythonModels
{

StreamDiffusionImg2Img::StreamDiffusionImg2Img() noexcept
{
#if 1
  m_wrapper.init();
#endif
}

StreamDiffusionImg2Img::~StreamDiffusionImg2Img() { }

void StreamDiffusionImg2Img::operator()()
{
  if (inputs.manual && !inputs.trigger.value.has_value())
    return;
  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
    return;
  if (!m_wrapper.prepare_inference())
    return;

  auto& out_img = outputs.image;
  auto [w, h] = inputs.size.value;
  out_img.create(w, h);

  m_wrapper.img2img(
      in_tex.bytes, in_tex.width, in_tex.height, out_img.texture.bytes);

  out_img.texture.changed = true;
}
}
