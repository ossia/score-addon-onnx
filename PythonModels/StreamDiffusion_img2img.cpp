#include "StreamDiffusion_img2img.hpp"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/subinterpreter.h>

namespace PythonModels
{

StreamDiffusionImg2Img::StreamDiffusionImg2Img() noexcept
{
#if 1
  namespace py = pybind11;
  in_vec.reserve(512 * 512 * 3);
  out_vec.reserve(512 * 512 * 3);
  std::atomic_bool ready{};
  m_t = std::thread(
      [this, &ready]
      {
        m_running = true;
        static int guard = python_interpreter_global_instance();
        m_wrapper = std::make_shared<StreamDiffusionWrapper>();
        static int path = m_wrapper->setup_path();
        static int imports = m_wrapper->setup_imports();
        m_wrapper->init();
        ready = true;
        int lq = 0;
        while (m_running)
        {
          if (queue <= 1)
            continue;
          lq = queue;
          auto t0 = std::chrono::high_resolution_clock::now();
          {
            std::lock_guard _{m_param_mutex};
            if (!m_wrapper->prepare_inference())
              continue;
            m_wrapper->img2img(in_vec.data(), 512, 512, out_vec.data());
          }

          auto t1 = std::chrono::high_resolution_clock::now();
          qDebug()
              << (1e9
                  / std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t1 - t0)
                        .count())
              << "hz";
        }
        m_wrapper.reset();
      });
  while (!ready)
    std::this_thread::yield();
#endif
}

StreamDiffusionImg2Img::~StreamDiffusionImg2Img()
{
  m_running = false;
  m_t.join();
}

void StreamDiffusionImg2Img::operator()()
{
  if (inputs.manual && !inputs.trigger.value.has_value())
    return;
  auto& in_tex = inputs.image.texture;
  // if (!in_tex.changed)
  //   return;
  if (in_tex.changed)
    queue++;

  auto& out_img = outputs.image;
  auto [w, h] = inputs.size.value;
  out_img.create(w, h);

  std::memcpy(in_vec.data(), in_tex.bytes, 512 * 512 * 3);

  // m_wrapper.img2img(
  //     in_tex.bytes, in_tex.width, in_tex.height, out_img.texture.bytes);

  std::memcpy(out_img.texture.bytes, out_vec.data(), 512 * 512 * 3);

  out_img.texture.changed = true;
}
}
