#pragma once
#include <boost/container/vector.hpp>
#include <boost/dynamic_bitset.hpp>

#include <Onnx/helpers/CoreTypes.hpp>
#include <Onnx/helpers/CtxOverlay.hpp>
#include <Onnx/helpers/ImageBuffer.hpp>
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxBase.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace Onnx
{

inline FloatTensor nchw_tensorFromARGB(
    const ModelSpec::Port& port,
    const unsigned char* source_bits,
    int source_w,
    int source_h,
    int model_w,
    int model_h,
    boost::container::vector<float>& input_tensor_values,
    std::array<float, 3> mean,
    std::array<float, 3> std)
{
  // One image: force the batch dim to 1. The model's declared shape may carry a
  // dynamic (-1) or >1 batch, which would make the tensor's element count differ
  // from the single-image buffer we fill below (ORT OOB read / create failure).
  std::vector<std::int64_t> input_shape = port.shape;
  if(!input_shape.empty())
    input_shape[0] = 1;
  auto rgba = resize_fill_crop_rgba(source_bits, source_w, source_h, model_w, model_h);
  auto rgb = rgba_to_rgb(rgba.data(), model_w, model_h);

  input_tensor_values.resize(
      3 * model_w * model_h, boost::container::default_init);

  auto ptr = rgb.data();
  auto dst = input_tensor_values.data();
  auto dst_r = dst;
  auto dst_g = dst_r + model_w * model_h;
  auto dst_b = dst_g + model_w * model_h;

  nhwc_to_nchw<float>(
      model_w,
      model_h,
      model_w * 3,
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
    const ModelSpec::Port& port,
    const unsigned char* source_bits,
    int source_w,
    int source_h,
    int model_w,
    int model_h,
    boost::container::vector<float>& input_tensor_values,
    std::array<float, 3> mean,
    std::array<float, 3> std)
{
  // One image: force the batch dim to 1. The model's declared shape may carry a
  // dynamic (-1) or >1 batch, which would make the tensor's element count differ
  // from the single-image buffer we fill below (ORT OOB read / create failure).
  std::vector<std::int64_t> input_shape = port.shape;
  if(!input_shape.empty())
    input_shape[0] = 1;
  auto rgba = resize_fill_crop_rgba(source_bits, source_w, source_h, model_w, model_h);
  auto rgb = rgba_to_rgb(rgba.data(), model_w, model_h);

  // FIXME pass storage as input instead
  input_tensor_values.resize(
      3 * model_w * model_h, boost::container::default_init);

  auto ptr = rgb.data();
  auto dst = input_tensor_values.data();
  auto dst_r = dst;
  auto dst_g = dst_r + model_w * model_h;
  auto dst_b = dst_g + model_w * model_h;
  nhwc_to_nchw<float>(
      model_w,
      model_h,
      model_w * 3,
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
    const ModelSpec::Port& port,
    const unsigned char* source_bits,
    int source_w,
    int source_h,
    int model_w,
    int model_h,
    boost::container::vector<float>& input_tensor_values)
{
  // One image: force the batch dim to 1. The model's declared shape may carry a
  // dynamic (-1) or >1 batch, which would make the tensor's element count differ
  // from the single-image buffer we fill below (ORT OOB read / create failure).
  std::vector<std::int64_t> input_shape = port.shape;
  if(!input_shape.empty())
    input_shape[0] = 1;
  auto rgba = resize_fill_crop_rgba(source_bits, source_w, source_h, model_w, model_h);

  input_tensor_values.resize(
      3 * model_w * model_h, boost::container::default_init);

  auto ptr = rgba.data();
  auto dst = input_tensor_values.data();
  for (int src_i = 0, dst_i = 0; dst_i < 3 * model_w * model_h;)
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

// Composite detection rectangles (and optional names / keypoints) onto a copy
// of the RGBA8888 input frame using the Qt-free ctx overlay. Returns the
// rasterized RGBA buffer; the caller memcpy's it into its output texture.
inline ImageData drawRects(const unsigned char* input, int w, int h, auto& rects)
{
  ImageData img;
  img.width = w;
  img.height = h;
  img.pixels.assign(
      input, input + static_cast<std::size_t>(w) * h * 4);

  {
    OnnxModels::Overlay ov(img.pixels.data(), w, h);
    ov.color(Onnx::Rgba{1.f, 1.f, 1.f, 1.f});
    ov.lineWidth(4.f);
    for (const auto& rect : rects)
    {
      // Clamp coordinates to valid range
      int rx = std::clamp(static_cast<int>(rect.geometry.x), 0, w);
      int ry = std::clamp(static_cast<int>(rect.geometry.y), 0, h);
      int rw = std::clamp(static_cast<int>(rect.geometry.w), 0, w - rx);
      int rh = std::clamp(static_cast<int>(rect.geometry.h), 0, h - ry);
      ov.strokeRect(Onnx::Rect{
          static_cast<float>(rx), static_cast<float>(ry),
          static_cast<float>(rw), static_cast<float>(rh)});
      if constexpr (requires { rect.name; })
      {
        ov.text(
            14.f,
            Onnx::Vec2{
                static_cast<float>(std::max(0, rx)),
                static_cast<float>(std::max(10, ry))},
            rect.name.c_str());
      }

      if constexpr (requires { rect.keypoints; })
      {
        for (auto [i, x, y] : rect.keypoints)
        {
          if (x >= 0 && x < w && y >= 0 && y < h)
            ov.fillCircle(
                Onnx::Vec2{static_cast<float>(x), static_cast<float>(y)}, 5.f);
        }
      }
    }
  } // Overlay dtor rasterizes into img.pixels
  return img;
}

// Function to draw segmentation results onto a copy of the RGBA8888 input.
inline ImageData drawBlobAndSegmentation(
    const unsigned char* input,
    int w,
    int h,
    const auto& segmentations)
{
  ImageData img;
  img.width = w;
  img.height = h;
  img.pixels.assign(
      input, input + static_cast<std::size_t>(w) * h * 4);

  // 1) Tint the masked pixels directly in the RGBA buffer (was a premultiplied
  //    ARGB overlay composited with QPainter).
  int k = 120;
  for (const auto& seg : segmentations)
  {
    const Onnx::Rgba c = Onnx::hsv((k % 360) / 360.f, 200.f / 255.f, 220.f / 255.f);
    const double r = c.r * 255.0, g = c.g * 255.0, b = c.b * 255.0;
    auto data = img.pixels.data();
    const boost::dynamic_bitset<>& mask = seg.mask;
    const size_t mask_size = mask.size();
#pragma omp simd
    for (int y = 0; y < h; y++)
    {
      for (int x = 0; x < w; x++)
      {
        size_t idx = y * w + x;
        if (idx < mask_size && mask[idx])
        {
          auto* pixel = data + idx * 4;
          pixel[0] = static_cast<unsigned char>(std::min(pixel[0] + r * 0.25, 255.));
          pixel[1] = static_cast<unsigned char>(std::min(pixel[1] + g * 0.25, 255.));
          pixel[2] = static_cast<unsigned char>(std::min(pixel[2] + b * 0.25, 255.));
        }
      }
    }

    k += 3759;
    k = k % 359;
  }

  // 2) Stroke boxes + labels with the ctx overlay.
  {
    OnnxModels::Overlay ov(img.pixels.data(), w, h);
    for (const auto& seg : segmentations)
    {
      ov.color(Onnx::Rgba{1.f, 1.f, 1.f, 1.f});
      ov.lineWidth(2.f);

      // Clamp coordinates to valid range
      int rx = std::clamp(static_cast<int>(seg.geometry.x), 0, w);
      int ry = std::clamp(static_cast<int>(seg.geometry.y), 0, h);
      int rw = std::clamp(static_cast<int>(seg.geometry.w), 0, w - rx);
      int rh = std::clamp(static_cast<int>(seg.geometry.h), 0, h - ry);
      ov.strokeRect(Onnx::Rect{
          static_cast<float>(rx), static_cast<float>(ry),
          static_cast<float>(rw), static_cast<float>(rh)});

      char label[256];
      std::snprintf(
          label, sizeof(label), "%s: %.2f", seg.name.c_str(),
          static_cast<double>(seg.confidence));
      ov.text(
          14.f,
          Onnx::Vec2{
              static_cast<float>(std::max(0, rx)),
              static_cast<float>(std::max(10, ry - 5))},
          label);
    }
  }

  return img;
}

inline ImageData drawKeypoints(
    const unsigned char* input,
    int w,
    int h,
    float min_confidence,
    const auto& keypoints)
{
  ImageData img;
  img.width = w;
  img.height = h;
  img.pixels.assign(
      input, input + static_cast<std::size_t>(w) * h * 4);

  {
    OnnxModels::Overlay ov(img.pixels.data(), w, h);
    for (const auto& kp : keypoints)
    {
      if (kp.x >= 0 && kp.x < w)
        if (kp.y >= 0 && kp.y < h)
        {
          float conf = std::clamp(static_cast<float>(kp.confidence()), 0.0f, 1.0f);
          ov.color(Onnx::Rgba{1.f, 1.f, 1.f, conf});
          //if (kp.confidence() >= min_confidence)
          ov.fillCircle(
              Onnx::Vec2{
                  static_cast<float>(kp.x), static_cast<float>(kp.y)},
              5.f);
        }
    }
  }

  return img;
}

// Set one RGBA8888 pixel (alpha forced opaque) in an ImageData buffer.
inline void
setRgb(ImageData& img, int x, int y, int r, int g, int b)
{
  unsigned char* p
      = img.pixels.data() + (static_cast<std::size_t>(y) * img.width + x) * 4;
  p[0] = static_cast<unsigned char>(r);
  p[1] = static_cast<unsigned char>(g);
  p[2] = static_cast<unsigned char>(b);
  p[3] = 255;
}

// GAN-specific image utilities. Produce an RGBA8888 ImageData (was QImage
// RGB888); the alpha channel is opaque.
inline ImageData normalizeToImage(
    const float* data,
    const std::vector<int64_t>& shape,
    float min_val,
    float max_val)
{
  if (shape.size() != 4)
    return {};

  int channels = static_cast<int>(shape[1]);
  int height = static_cast<int>(shape[2]);
  int width = static_cast<int>(shape[3]);

  if (channels != 3)
    return {};

  // Find actual min/max for better normalization
  float actual_min = data[0], actual_max = data[0];
  size_t total_pixels = channels * height * width;
  for (size_t i = 0; i < total_pixels; ++i) {
    actual_min = std::min(actual_min, data[i]);
    actual_max = std::max(actual_max, data[i]);
  }

  ImageData image;
  image.width = width;
  image.height = height;
  image.pixels.resize(static_cast<std::size_t>(width) * height * 4);

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

      setRgb(image, x, y, red, green, blue);
    }
  }

  return image;
}

inline ImageData
nchwToQImage(const float* data, int width, int height, int channels)
{
  if (channels != 3)
    return {};

  ImageData image;
  image.width = width;
  image.height = height;
  image.pixels.resize(static_cast<std::size_t>(width) * height * 4);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float r = data[0 * height * width + y * width + x];
      float g = data[1 * height * width + y * width + x];
      float b = data[2 * height * width + y * width + x];

      // Assume values are in [0,1] range
      int red = static_cast<int>(std::clamp(r * 255.0f, 0.0f, 255.0f));
      int green = static_cast<int>(std::clamp(g * 255.0f, 0.0f, 255.0f));
      int blue = static_cast<int>(std::clamp(b * 255.0f, 0.0f, 255.0f));

      setRgb(image, x, y, red, green, blue);
    }
  }

  return image;
}

// Convert an RGBA8888 ImageData to tensor data in the specified format (the
// alpha channel is dropped).
inline std::vector<float> imageToTensor(
    const ImageData& image,
    const std::string& tensor_format,
    float input_mean = 0.0f,
    float input_std = 1.0f)
{
  if (image.empty())
    return {};

  int width = image.width;
  int height = image.height;
  const unsigned char* src = image.pixels.data();
  std::vector<float> tensor_data(height * width * 3);

  if (tensor_format == "NCHW") {
    // NCHW format: [channels, height, width]
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const unsigned char* px
            = src + (static_cast<std::size_t>(y) * width + x) * 4;

        // Normalize based on config
        float r = (px[0] / 255.0f - input_mean) / input_std;
        float g = (px[1] / 255.0f - input_mean) / input_std;
        float b = (px[2] / 255.0f - input_mean) / input_std;

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
        const unsigned char* px
            = src + (static_cast<std::size_t>(y) * width + x) * 4;

        // Normalize to [0,1] range
        float r = px[0] / 255.0f;
        float g = px[1] / 255.0f;
        float b = px[2] / 255.0f;

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
