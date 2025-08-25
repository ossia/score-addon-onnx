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
  QImage img
      = QImage(source_bits, source_w, source_h, QImage::Format_RGBA8888);
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

// GAN-specific image utilities
inline QImage normalizeToImage(
    const float* data,
    const std::vector<int64_t>& shape,
    float min_val,
    float max_val)
{
  if (shape.size() != 4)
    return QImage();

  int channels = static_cast<int>(shape[1]);
  int height = static_cast<int>(shape[2]);
  int width = static_cast<int>(shape[3]);

  if (channels != 3)
    return QImage();

  // Find actual min/max for better normalization
  float actual_min = data[0], actual_max = data[0];
  size_t total_pixels = channels * height * width;
  for (size_t i = 0; i < total_pixels; ++i) {
    actual_min = std::min(actual_min, data[i]);
    actual_max = std::max(actual_max, data[i]);
  }

  QImage image(width, height, QImage::Format_RGB888);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Get RGB values from NCHW tensor
      float r = data[0 * height * width + y * width + x];
      float g = data[1 * height * width + y * width + x];
      float b = data[2 * height * width + y * width + x];

      // Normalize to [0,255]
      int red = static_cast<int>(std::clamp(
          (r - actual_min) / (actual_max - actual_min) * 255.0f,
          0.0f,
          255.0f));
      int green = static_cast<int>(std::clamp(
          (g - actual_min) / (actual_max - actual_min) * 255.0f,
          0.0f,
          255.0f));
      int blue = static_cast<int>(std::clamp(
          (b - actual_min) / (actual_max - actual_min) * 255.0f,
          0.0f,
          255.0f));

      image.setPixel(x, y, qRgb(red, green, blue));
    }
  }

  return image;
}

inline QImage nchwToQImage(const float* data, int width, int height, int channels)
{
  if (channels != 3)
    return QImage();

  QImage image(width, height, QImage::Format_RGB888);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float r = data[0 * height * width + y * width + x];
      float g = data[1 * height * width + y * width + x];
      float b = data[2 * height * width + y * width + x];

      // Assume values are in [0,1] range
      int red = static_cast<int>(std::clamp(r * 255.0f, 0.0f, 255.0f));
      int green = static_cast<int>(std::clamp(g * 255.0f, 0.0f, 255.0f));
      int blue = static_cast<int>(std::clamp(b * 255.0f, 0.0f, 255.0f));

      image.setPixel(x, y, qRgb(red, green, blue));
    }
  }

  return image;
}

// Convert QImage to tensor data in specified format
inline std::vector<float> imageToTensor(
    const QImage& image,
    const std::string& tensor_format,
    float input_mean = 0.0f,
    float input_std = 1.0f)
{
  if (image.isNull())
    return {};

  // Convert to RGB format if necessary
  QImage rgb_image = image.convertToFormat(QImage::Format_RGB888);
  int width = rgb_image.width();
  int height = rgb_image.height();
  std::vector<float> tensor_data(height * width * 3);

  if (tensor_format == "NCHW") {
    // NCHW format: [channels, height, width]
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        QRgb pixel = rgb_image.pixel(x, y);

        // Normalize based on config
        float r = (qRed(pixel) / 255.0f - input_mean) / input_std;
        float g = (qGreen(pixel) / 255.0f - input_mean) / input_std;
        float b = (qBlue(pixel) / 255.0f - input_mean) / input_std;

        // NCHW format: tensor[c][y][x]
        tensor_data[0 * height * width + y * width + x] = r; // R channel
        tensor_data[1 * height * width + y * width + x] = g; // G channel
        tensor_data[2 * height * width + y * width + x] = b; // B channel
      }
    }
  } else {
    // NHWC format: [height, width, channels]
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        QRgb pixel = rgb_image.pixel(x, y);

        // Normalize to [0,1] range
        float r = qRed(pixel) / 255.0f;
        float g = qGreen(pixel) / 255.0f;
        float b = qBlue(pixel) / 255.0f;

        // NHWC format: tensor[y][x][c]
        size_t idx = y * width * 3 + x * 3;
        tensor_data[idx + 0] = r;
        tensor_data[idx + 1] = g;
        tensor_data[idx + 2] = b;
      }
    }
  }

  return tensor_data;
}
}
