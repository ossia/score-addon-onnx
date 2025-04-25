#pragma once
#include <boost/container/vector.hpp>

#include <QDebug>
#include <QImage>
#include <QPainter>
#include <QPen>

#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxBase.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <vector>

namespace Onnx
{

struct FloatTensor
{
  boost::container::vector<float> storage;
  Ort::Value value;
};

inline QImage rescaleImage(
    const unsigned char* source_bits,
    int source_w,
    int source_h,
    QImage::Format sourceFormat,
    int model_w,
    int model_h)
{
  QImage img = QImage(source_bits, source_w, source_h, sourceFormat);
  if (source_w == model_w && source_h == model_h)
    return img;

  img = img.scaled(
      model_w,
      model_h,
      Qt::AspectRatioMode::KeepAspectRatioByExpanding,
      Qt::SmoothTransformation);
  if (model_w == img.width() && model_h == img.height())
    return img;

  return img.copy(0, 0, model_w, model_h);
}

inline FloatTensor tensorFromARGB(
    ModelSpec::Port& port,
    const unsigned char* source_bits,
    int source_w,
    int source_h,
    int model_w,
    int model_h,
    boost::container::vector<float>& input_tensor_values,
    bool normalize_resnet = false)
{
  auto& input_shape = port.shape;
  QImage img = rescaleImage(
      source_bits,
      source_w,
      source_h,
      QImage::Format_ARGB32,
      model_w,
      model_h);
  img = img.rgbSwapped();
  img = img.convertToFormat(QImage::Format_RGB888);

  // FIXME pass storage as input instead
  input_tensor_values.resize(
      3 * model_w * model_h, boost::container::default_init);

  auto ptr = (unsigned char*)img.constBits();
  auto dst = input_tensor_values.data();
  auto dst_r = dst;
  auto dst_g = dst_r + model_w * model_h;
  auto dst_b = dst_g + model_w * model_h;
  if (normalize_resnet)
  {
    nhwc_to_nchw<float>(
        model_w,
        model_h,
        3 * model_w,
        ptr,
        dst_r,
        dst_g,
        dst_b,
        {255.f * 0.485f, 255.f * 0.456f, 255.f * 0.406f},
        {255.f * 0.229f, 255.f * 0.224f, 255.f * 0.225f});
  }
  else
  {
    nhwc_to_nchw<float>(
        model_w,
        model_h,
        3 * model_w,
        ptr,
        dst_r,
        dst_g,
        dst_b,
        {0., 0., 0.},
        {255., 255., 255.});
  }

  FloatTensor f{
      .storage = {},
      .value = vec_to_tensor<float>(input_tensor_values, input_shape)};
  f.storage = std::move(input_tensor_values);
  return f;
}

// When coming from a GL_RGBA texture
inline FloatTensor tensorFromRGBA(
    ModelSpec::Port& port,
    const unsigned char* source_bits,
    int source_w,
    int source_h,
    int model_w,
    int model_h,
    boost::container::vector<float>& input_tensor_values,
    bool normalize_resnet = false)
{
  auto& input_shape = port.shape;
  QImage img = rescaleImage(
      source_bits,
      source_w,
      source_h,
      QImage::Format_RGBA8888,
      model_w,
      model_h);
  img = img.convertToFormat(QImage::Format_RGB888);

  input_tensor_values.resize(
      3 * model_w * model_h, boost::container::default_init);

  auto ptr = (unsigned char*)img.constBits();
  auto dst = input_tensor_values.data();
  auto dst_r = dst;
  auto dst_g = dst_r + model_w * model_h;
  auto dst_b = dst_g + model_w * model_h;
  if (normalize_resnet)
  {
    nhwc_to_nchw<float>(
        model_w,
        model_h,
        3 * model_w,
        ptr,
        dst_r,
        dst_g,
        dst_b,
        {255.f * 0.485f, 255.f * 0.456f, 255.f * 0.406f},
        {255.f * 0.229f, 255.f * 0.224f, 255.f * 0.225f});
  }
  else
  {
    nhwc_to_nchw<float>(
        model_w,
        model_h,
        3 * model_w,
        ptr,
        dst_r,
        dst_g,
        dst_b,
        {0., 0., 0.},
        {255., 255., 255.});
  }

  FloatTensor f{
      .storage = {},
      .value = vec_to_tensor<float>(input_tensor_values, input_shape)};
  f.storage = std::move(input_tensor_values);
  return f;
}

// When coming from a GL_RGBA texture
inline FloatTensor nhwc_rgb_tensorFromRGBA(
    ModelSpec::Port& port,
    const unsigned char* source_bits,
    int source_w,
    int source_h,
    int model_w,
    int model_h,
    boost::container::vector<float>& input_tensor_values)
{
  auto& input_shape = port.shape;
  auto img = rescaleImage(
      source_bits,
      source_w,
      source_h,
      QImage::Format_RGBA8888,
      model_w,
      model_h);

  input_tensor_values.resize(
      3 * model_w * model_h, boost::container::default_init);

  auto ptr = (unsigned char*)img.constBits();
  auto dst = input_tensor_values.data();
  for (int src_i = 0, dst_i = 0; src_i < 3 * model_w * model_h;)
  {
    dst[dst_i] = ptr[src_i] / 255.f;
    dst[dst_i + 1] = ptr[src_i + 1] / 255.f;
    dst[dst_i + 2] = ptr[src_i + 2] / 255.f;

    src_i += 4;
    dst_i += 3;
  }

  FloatTensor f{
      .storage = {},
      .value = vec_to_tensor<float>(input_tensor_values, input_shape)};
  f.storage = std::move(input_tensor_values);
  return f;
}

inline QImage drawRects(const unsigned char* input, int w, int h, auto& rects)
{
  QImage img(input, w, h, QImage::Format_RGBA8888);

  {
    QPainter p(&img);
    p.setPen(QPen(Qt::white, 4));
    p.setBrush(Qt::NoBrush);
    for (const auto& rect : rects)
    {
      p.drawRect(
          rect.geometry.x, rect.geometry.y, rect.geometry.w, rect.geometry.h);
      if constexpr (requires { rect.name; })
      {
        p.drawText(
            rect.geometry.x,
            rect.geometry.y,
            QString::fromStdString(rect.name));
      }

      if constexpr (requires { rect.keypoints; })
      {
        for (auto [i, x, y] : rect.keypoints)
        {
          p.drawEllipse(QPoint(x, y), 5, 5);
        }
      }
    }
  }
  return img;
}

inline QImage drawKeypoints(
    const unsigned char* input,
    int w,
    int h,
    float min_confidence,
    const auto& keypoints)
{
  QImage img(input, w, h, QImage::Format_RGBA8888);

  {
    QPainter p(&img);
    p.setPen(QPen(Qt::white, 4));
    p.setBrush(Qt::NoBrush);
    for (const auto& kp : keypoints)
    {
      //if (kp.confidence() >= min_confidence)
        p.drawEllipse(QPoint(kp.x, kp.y), 5, 5);
    }
  }

  return img;
}
}
