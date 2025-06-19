#pragma once
#include <boost/container/vector.hpp>
#include <boost/dynamic_bitset.hpp>

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

inline FloatTensor nchw_tensorFromARGB(
    ModelSpec::Port& port,
    const unsigned char* source_bits,
    int source_w,
    int source_h,
    int model_w,
    int model_h,
    boost::container::vector<float>& input_tensor_values,
    std::array<float, 3> mean,
    std::array<float, 3> std)
{
  auto& input_shape = port.shape;
  QImage img = QImage(source_bits, source_w, source_h, QImage::Format_ARGB32);
  img = std::move(img).scaled(
      model_w,
      model_h,
      Qt::AspectRatioMode::KeepAspectRatioByExpanding,
      Qt::SmoothTransformation);
  if (model_w != img.width() || model_h != img.height())
    img = img.copy(0, 0, model_w, model_h);
  img = std::move(img).convertToFormat(QImage::Format_RGB888);

  input_tensor_values.resize(
      3 * model_w * model_h, boost::container::default_init);

  auto ptr = (unsigned char*)img.constBits();
  auto dst = input_tensor_values.data();
  auto dst_r = dst;
  auto dst_g = dst_r + model_w * model_h;
  auto dst_b = dst_g + model_w * model_h;

  nhwc_to_nchw<float>(
      model_w,
      model_h,
      img.bytesPerLine(),
      ptr,
      dst_r,
      dst_g,
      dst_b,
      mean,
      std);

  FloatTensor f{
      .storage = {},
      .value = vec_to_tensor<float>(input_tensor_values, input_shape)};
  f.storage = std::move(input_tensor_values);
  return f;
}

// When coming from a GL_RGBA texture
inline FloatTensor nchw_tensorFromRGBA(
    ModelSpec::Port& port,
    const unsigned char* source_bits,
    int source_w,
    int source_h,
    int model_w,
    int model_h,
    boost::container::vector<float>& input_tensor_values,
    std::array<float, 3> mean,
    std::array<float, 3> std)
{
  auto& input_shape = port.shape;
  QImage img
      = QImage(source_bits, source_w, source_h, QImage::Format_RGBA8888);
  img = img.scaled(
      model_w,
      model_h,
      Qt::AspectRatioMode::KeepAspectRatioByExpanding,
      Qt::SmoothTransformation);
  if (model_w != img.width() || model_h != img.height())
    img = img.copy(0, 0, model_w, model_h);
  img = std::move(img).convertToFormat(QImage::Format_RGB888);

  // FIXME pass storage as input instead
  input_tensor_values.resize(
      3 * model_w * model_h, boost::container::default_init);

  auto ptr = (unsigned char*)img.constBits();
  auto dst = input_tensor_values.data();
  auto dst_r = dst;
  auto dst_g = dst_r + model_w * model_h;
  auto dst_b = dst_g + model_w * model_h;
  nhwc_to_nchw<float>(
      model_w,
      model_h,
      img.bytesPerLine(),
      ptr,
      dst_r,
      dst_g,
      dst_b,
      mean,
      std);

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
  QImage img
      = QImage(source_bits, source_w, source_h, QImage::Format_RGBA8888);
  img = img.scaled(
      model_w,
      model_h,
      Qt::AspectRatioMode::KeepAspectRatioByExpanding,
      Qt::SmoothTransformation);
  if (model_w != img.width() || model_h != img.height())
    img = img.copy(0, 0, model_w, model_h);

  input_tensor_values.resize(
      3 * model_w * model_h, boost::container::default_init);

  // FIXME does not work if img.bytesPerLine() != 4 * model_w;
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

// Function to draw segmentation results on a QImage
inline QImage drawBlobAndSegmentation(
    const unsigned char* input,
    int w,
    int h,
    const auto& segmentations)
{
  QImage img(input, w, h, QImage::Format_RGBA8888);
  QImage maskOverlay(w, h, QImage::Format_ARGB32_Premultiplied);
  maskOverlay.fill(Qt::transparent);

  QPainter painter(&maskOverlay);
  painter.setRenderHint(QPainter::Antialiasing);

  int k = 120;
  for (const auto& seg : segmentations)
  {
    QColor color = QColor::fromHsv(k, 200, 220);
    color.setAlpha(100);
    auto r = color.red();
    auto g = color.green();
    auto b = color.blue();
    painter.setBrush(color);
    painter.setPen(color);
    auto data = img.bits();
    const boost::dynamic_bitset<>& mask = seg.mask;
#pragma omp simd
    for (int y = 0; y < h; y++)
    {
      for (int x = 0; x < w; x++)
      {
        if (mask[y * w + x])
        {
          auto* pixel = data + (y * w + x) * 4;
          pixel[0] = std::min(pixel[0] + r * 0.25, 255.);
          pixel[1] = std::min(pixel[1] + g * 0.25, 255.);
          pixel[2] = std::min(pixel[2] + b * 0.25, 255.);
        }
      }
    }

    k += 3759;
    k = k % 359;
  }

  QPainter finalPainter(&img);
  finalPainter.drawImage(0, 0, maskOverlay);

  for (const auto& seg : segmentations)
  {
    finalPainter.setBrush(Qt::NoBrush);
    finalPainter.setPen(QPen(Qt::white, 2));
    finalPainter.drawRect(
        seg.geometry.x, seg.geometry.y, seg.geometry.w, seg.geometry.h);

    QString label = QString("%1: %2")
                        .arg(QString::fromStdString(seg.name))
                        .arg(seg.confidence, 0, 'f', 2);

    QPointF textPos(seg.geometry.x, seg.geometry.y - 5);
    finalPainter.setPen(Qt::white);
    finalPainter.drawText(textPos, label);
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
      p.setPen(QPen(QColor::fromRgbF(1., 1., 1., kp.confidence()), 4));
      //if (kp.confidence() >= min_confidence)
        p.drawEllipse(QPoint(kp.x, kp.y), 5, 5);
    }
  }

  return img;
}
}
