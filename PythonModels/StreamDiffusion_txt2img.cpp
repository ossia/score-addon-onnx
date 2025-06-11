#include "StreamDiffusion_txt2img.hpp"

#include <ossia/detail/triple_buffer.hpp>

#include <QDebug>
#include <QElapsedTimer>

#include <thread>
namespace PythonModels
{
struct sd_thread
{
  StreamDiffusionWrapper& m_wrapper;
  int width, height;
  ossia::triple_buffer<std::vector<unsigned char>> bytes;

  sd_thread(StreamDiffusionWrapper& w, int width, int height)
      : m_wrapper{w}
      , width{width}
      , height{height}
      , bytes{std::vector<unsigned char>(width * height * 4)}
  {
    produce.resize(width * height * 4);
    consume.resize(width * height * 4);
    t = std::thread{[this]
                    {
                      m_wrapper.init();
                      for (;;)
                        this->run();
                    }};
  }
  void run()
  {
    tm.restart();
    if (!m_wrapper.prepare_inference())
      return;

    produce.resize(width * height * 4);
    m_wrapper.txt2img(produce.data());
    bytes.produce(produce);
    qDebug() << int(1e9 / double(tm.nsecsElapsed())) << "fps";
  }

  void read(unsigned char* res)
  {
    if (bytes.consume(consume))
    {
      if (std::ssize(consume) == width * height * 4)
        std::copy_n(consume.data(), consume.size(), res);
    }
  }
  std::thread t;
  std::vector<unsigned char> produce;
  std::vector<unsigned char> consume;
  QElapsedTimer tm;
};

StreamDiffusionTxt2Img::StreamDiffusionTxt2Img() noexcept
{
#if 1
  m_wrapper.init();
#endif
  m_wrapper.set_cfg("none");
}

StreamDiffusionTxt2Img::~StreamDiffusionTxt2Img() { }

void StreamDiffusionTxt2Img::operator()()
{
  if (inputs.manual && !inputs.trigger.value.has_value())
    return;

#if 1
  if (!m_wrapper.prepare_inference())
    return;

  inputs.trigger.value.reset();
  auto& out_img = outputs.image;
  auto [w, h] = inputs.size.value;
  out_img.create(w, h);

  m_wrapper.txt2img(out_img.texture.bytes);

  out_img.texture.changed = true;
#else
  auto [w, h] = inputs.size.value;
  static auto th = sd_thread{m_wrapper, w, h};
  auto& out_img = outputs.image;
  out_img.create(w, h);
  SCORE_ASSERT(out_img.texture.bytesize() == w * h * 4);
  th.read(out_img.texture.bytes);
  out_img.texture.changed = true;
#endif
}
}
