#pragma once
// Preprocessing helpers and decoding of image-model outputs to the Image /
// Mask / Depth / Data outlets, shared by the Image Processor and the Video
// Processor (both have the same
// outputs.{image,mask,depth,data} members).
#include <OnnxModels/ImageEnums.hpp>

#include <Onnx/helpers/ImageModelRole.hpp>
#include <Onnx/helpers/ImageOps.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/TensorToTexture.hpp>

#include <onnxruntime_cxx_api.h>

#include <boost/container/vector.hpp>

#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <vector>

namespace OnnxModels::imgdec
{
using Onnx::ImageModelKind;
using Onnx::TensorElemType;

// A model with a 4-channel image input (RGB + a prior mask, e.g. PINTO's hair
// segmenter): the samplers build 3 channels, add a 4th one at zero. `shape`
// is the 3-channel tensor shape ({1,3,H,W} or {1,H,W,3}) and is updated.
template <typename Buffer>
void addZeroChannel(Buffer& t, std::vector<int64_t>& shape)
{
  if(shape.size() != 4)
    return;
  if(shape[1] == 3) // NCHW: one more plane
  {
    const std::size_t hw = (std::size_t)(shape[2] * shape[3]);
    t.resize(4 * hw);
    std::fill(t.begin() + 3 * hw, t.end(), 0.f);
    shape[1] = 4;
  }
  else if(shape[3] == 3) // NHWC: RGB -> RGB0, repacked from the end
  {
    const std::size_t hw = (std::size_t)(shape[1] * shape[2]);
    t.resize(4 * hw);
    for(std::size_t p = hw; p-- > 0;)
    {
      t[p * 4 + 3] = 0.f;
      t[p * 4 + 2] = t[p * 3 + 2];
      t[p * 4 + 1] = t[p * 3 + 1];
      t[p * 4 + 0] = t[p * 3 + 0];
    }
    shape[3] = 4;
  }
}

// (mean, invstd) in the 0..255 sample domain expected by ImageOps' samplers.
inline void normConstants(InputNormalization n, float mean[3], float invstd[3])
{
  float m[3], s[3];
  switch(n)
  {
    case InputNormalization::ImageNet:
      m[0] = 255.f * 0.485f; m[1] = 255.f * 0.456f; m[2] = 255.f * 0.406f;
      s[0] = 255.f * 0.229f; s[1] = 255.f * 0.224f; s[2] = 255.f * 0.225f;
      break;
    case InputNormalization::Centered:
      m[0] = m[1] = m[2] = 127.5f;
      s[0] = s[1] = s[2] = 127.5f;
      break;
    case InputNormalization::None:
      m[0] = m[1] = m[2] = 0.f;
      s[0] = s[1] = s[2] = 1.f;
      break;
    case InputNormalization::DivBy255:
    default:
      m[0] = m[1] = m[2] = 0.f;
      s[0] = s[1] = s[2] = 255.f;
      break;
  }
  for(int i = 0; i < 3; ++i)
  {
    mean[i] = m[i];
    invstd[i] = (s[i] != 0.f) ? 1.f / s[i] : 1.f;
  }
}

inline Onnx::TensorLayout toTensorLayout(Onnx::ImgLayout l, ChannelOrder ord)
{
  switch(l)
  {
    case Onnx::ImgLayout::NhwcRgb:
      return Onnx::TensorLayout::NhwcRgb;
    case Onnx::ImgLayout::NchwGray:
      return Onnx::TensorLayout::NchwGray;
    case Onnx::ImgLayout::NchwRgb:
    default:
      return (ord == ChannelOrder::BGR) ? Onnx::TensorLayout::NchwBgr
                                        : Onnx::TensorLayout::NchwRgb;
  }
}

// A free (dynamic) model dimension: the requested size rounded to the
// model's stride.
inline int snapDim(int v, int stride)
{
  if(stride <= 1)
    return std::max(1, v);
  int r = ((v + stride / 2) / stride) * stride;
  return std::max(stride, r);
}

// The ModelIO classifyImage reads, with the image input first: it takes input
// 0 as the image (FILM declares its `time` scalar before the two frames).
inline Onnx::ModelIO toImageIO(const Onnx::ModelSpec& s, int image_in = 0)
{
  Onnx::ModelIO io;
  io.inputs.reserve(s.inputs.size());
  io.outputs.reserve(s.outputs.size());
  if(image_in >= 0 && image_in < (int)s.inputs.size())
    io.inputs.push_back({s.inputs[image_in].name, s.inputs[image_in].shape});
  for(int i = 0; i < (int)s.inputs.size(); ++i)
    if(i != image_in)
      io.inputs.push_back({s.inputs[i].name, s.inputs[i].shape});
  for(const auto& p : s.outputs)
    io.outputs.push_back({p.name, p.shape});
  return io;
}

// The ORT input tensor over the preprocessed floats, in the model's declared
// element type (fp16, or uint8 for 0..255 inputs). The staging buffers must
// outlive the returned Value.
inline Ort::Value buildInputTensor(
    const boost::container::vector<float>& input, const std::vector<int64_t>& ishape,
    TensorElemType in_dt, std::vector<uint16_t>& half_buf, std::vector<uint8_t>& u8_buf)
{
  const std::size_t n = input.size();
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  if(in_dt == TensorElemType::Float16)
  {
    half_buf.resize(n);
    for(std::size_t i = 0; i < n; ++i)
      half_buf[i] = Onnx::floatToHalf(input[i]);
    return Ort::Value::CreateTensor(
        mem, half_buf.data(), n * sizeof(uint16_t), ishape.data(), ishape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  }
  if(in_dt == TensorElemType::Uint8)
  {
    u8_buf.resize(n);
    for(std::size_t i = 0; i < n; ++i)
      u8_buf[i] = (uint8_t)std::lround(std::clamp(input[i], 0.f, 255.f));
    return Ort::Value::CreateTensor<uint8_t>(
        mem, u8_buf.data(), n, ishape.data(), ishape.size());
  }
  return Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(input.data()), n, ishape.data(), ishape.size());
}

// A declared shape with its dynamic dims (<= 0) at 1, rank kept: a scalar
// input declared [] takes a rank-0 tensor.
inline std::vector<int64_t> concreteShape(const std::vector<int64_t>& s)
{
  std::vector<int64_t> r = s;
  for(auto& d : r)
    if(d <= 0)
      d = 1;
  return r;
}

// Heavy models run on the worker, so that a slow inference does not freeze
// the render thread.
inline bool isHeavyModel(std::size_t model_bytes, int64_t in_pixels)
{
  return model_bytes > 64u * 1024 * 1024 || in_pixels > (int64_t)512 * 512;
}

inline Onnx::WriteMode resolveWriteMode(OutputMode m, ImageModelKind kind)
{
  switch(m)
  {
    case OutputMode::DirectClamp:
      return Onnx::WriteMode::DirectClamp;
    case OutputMode::MinMaxNormalize:
      return Onnx::WriteMode::MinMaxNormalize;
    case OutputMode::Denormalize:
      return Onnx::WriteMode::Denormalize;
    case OutputMode::Passthrough:
      return Onnx::WriteMode::Passthrough;
    case OutputMode::Half255:
      return Onnx::WriteMode::Half255;
    case OutputMode::Sigmoid:
      return Onnx::WriteMode::Sigmoid;
    case OutputMode::Auto:
    default:
      // image -> assume [0,1]; a mask already in [0,1] (an alpha matte, a
      // sigmoid) is kept as is, any other range is stretched: stretching a
      // nearly empty matte would turn a few weak pixels into white specks.
      return (kind == ImageModelKind::ImageToImage)
                 ? Onnx::WriteMode::DirectClamp
                 : Onnx::WriteMode::AutoRange;
  }
}

inline ImageModelKind applyTaskOverride(ImageModelKind k, TaskMode t)
{
  switch(t)
  {
    case TaskMode::Image: return ImageModelKind::ImageToImage;
    case TaskMode::Mask:  return ImageModelKind::ImageToMask;
    case TaskMode::Depth: return ImageModelKind::ImageToDepth;
    case TaskMode::Data:  return ImageModelKind::ImageToData;
    case TaskMode::Auto:
    default:              return k;
  }
}

// Decoded result staged in heap buffers (so the per-pixel decode also runs off
// the worker thread); applyDecoded() then just creates the texture + memcpy.
struct DecodedOutput
{
  enum Target
  {
    None,
    Image,
    Mask,
    Depth,
    Data
  } target = None;
  int w = 0, h = 0;
  std::vector<uint8_t> rgba; // Image: w*h*4
  std::vector<uint8_t> r8;   // Mask / Depth-visual: w*h
  std::vector<float> r32f;   // Depth raw: w*h
  std::vector<float> data;   // Data: flattened
};

// A single-channel spatial result is a mask, even if classifyImage guessed
// image: a fully-dynamic declared output shape (e.g. isnet / isnetis) hides the
// channel count at load time, but the real result tensor reveals it here.
inline ImageModelKind effectiveKind(ImageModelKind kind, const Onnx::OutSpec& os)
{
  // Two channels are a background / foreground segmentation (hand / human
  // segmenters), shown as a mask of the foreground.
  if(os.spatial && (os.channels == 1 || os.channels == 2)
     && kind == ImageModelKind::ImageToImage)
    return ImageModelKind::ImageToMask;
  return kind;
}

// The outlet a result of this kind and shape is written to.
inline DecodedOutput::Target targetOf(ImageModelKind kind, const Onnx::OutSpec& os)
{
  kind = effectiveKind(kind, os);
  if(kind == ImageModelKind::ImageToData || !os.spatial)
    return DecodedOutput::Data;
  switch(kind)
  {
    case ImageModelKind::ImageToImage:
    case ImageModelKind::LatentToImage:
      return DecodedOutput::Image;
    case ImageModelKind::ImageToMask:
      return DecodedOutput::Mask;
    case ImageModelKind::ImageToDepth:
      return DecodedOutput::Depth;
    default:
      return DecodedOutput::None;
  }
}

inline DecodedOutput decodeOutput(
    Ort::Value& res, ImageModelKind kind, Onnx::WriteMode wm,
    std::vector<float>& scratch)
{
  DecodedOutput d;
  const auto info = res.GetTensorTypeAndShapeInfo();
  const auto oshape = info.GetShape();
  const int64_t ocount = (int64_t)info.GetElementCount();
  const TensorElemType odt = Onnx::fromOrtElementType(info.GetElementType());
  const void* raw = res.GetTensorData<uint8_t>();
  const Onnx::OutSpec os = Onnx::makeOutSpec(oshape);
  const float* f = Onnx::toFloat(raw, ocount, odt, scratch);

  kind = effectiveKind(kind, os);

  if(kind == ImageModelKind::ImageToData || !os.spatial)
  {
    d.target = DecodedOutput::Data;
    d.data.assign(f, f + ocount);
    return d;
  }
  d.w = os.w;
  d.h = os.h;
  switch(kind)
  {
    case ImageModelKind::ImageToImage:
    case ImageModelKind::LatentToImage:
      d.target = DecodedOutput::Image;
      d.rgba.resize((size_t)os.w * os.h * 4);
      Onnx::writeRgb(f, os, wm, d.rgba.data());
      break;
    case ImageModelKind::ImageToMask:
      d.target = DecodedOutput::Mask;
      d.r8.resize((size_t)os.w * os.h);
      Onnx::writeMask(f, os, wm, d.r8.data());
      break;
    case ImageModelKind::ImageToDepth:
      d.target = DecodedOutput::Depth;
      d.r8.resize((size_t)os.w * os.h);
      Onnx::writeMask(f, os, Onnx::WriteMode::MinMaxNormalize, d.r8.data());
      d.r32f.resize((size_t)os.w * os.h);
      Onnx::writeMaskF(f, os, d.r32f.data());
      break;
    default:
      break;
  }
  return d;
}

template <typename Node>
void applyDecoded(Node& self, DecodedOutput& d)
{
  // A degenerate output shape (0-sized spatial dim) would create a 0x0 texture
  // and memcpy from an empty buffer; only the Data path is meaningful then.
  if(d.target != DecodedOutput::Data && (d.w <= 0 || d.h <= 0))
    return;
  switch(d.target)
  {
    case DecodedOutput::Image:
      self.outputs.image.create(d.w, d.h);
      std::memcpy(self.outputs.image.texture.bytes, d.rgba.data(), d.rgba.size());
      self.outputs.image.texture.changed = true;
      break;
    case DecodedOutput::Mask:
      self.outputs.mask.create(d.w, d.h);
      std::memcpy(self.outputs.mask.texture.bytes, d.r8.data(), d.r8.size());
      self.outputs.mask.texture.changed = true;
      break;
    case DecodedOutput::Depth:
      // No visual when another output already owns the Mask outlet.
      if(!d.r8.empty())
      {
        self.outputs.mask.create(d.w, d.h);
        std::memcpy(self.outputs.mask.texture.bytes, d.r8.data(), d.r8.size());
        self.outputs.mask.texture.changed = true;
      }
      self.outputs.depth.create(d.w, d.h);
      std::memcpy(
          self.outputs.depth.texture.bytes, d.r32f.data(),
          d.r32f.size() * sizeof(float));
      self.outputs.depth.texture.changed = true;
      break;
    case DecodedOutput::Data:
      self.outputs.data.value = std::move(d.data);
      break;
    default:
      break;
  }
}

using DecodedOutputs = boost::container::small_vector<DecodedOutput, 4>;

// Decode the Output Index result with the user's Task / Pixel Mapping, then
// route every other model output by its own role to an outlet the first one
// left free (e.g. Depth Anything 3 metric: depth -> Depth, sky -> Mask).
// Secondary outputs use the Auto pixel mapping, and are only decoded when their
// outlet is still free, so an N-mask model (u2net) doesn't pay for N decodes.
inline DecodedOutputs decodeAll(
    std::span<Ort::Value> outs, int primary, ImageModelKind kind, Onnx::WriteMode wm,
    const std::vector<ImageModelKind>& out_kinds, std::vector<float>& scratch)
{
  DecodedOutputs res;
  const int nout = (int)outs.size();
  if(primary < 0 || primary >= nout || !outs[primary])
    return res;

  bool taken[DecodedOutput::Data + 1]{};
  auto claim = [&](const DecodedOutput& d) {
    taken[d.target] = true;
    if(d.target == DecodedOutput::Depth && !d.r8.empty())
      taken[DecodedOutput::Mask] = true;
  };

  res.push_back(decodeOutput(outs[primary], kind, wm, scratch));
  claim(res.back());

  for(int j = 0; j < nout; ++j)
  {
    if(j == primary || !outs[j] || j >= (int)out_kinds.size())
      continue;
    const Onnx::OutSpec os
        = Onnx::makeOutSpec(outs[j].GetTensorTypeAndShapeInfo().GetShape());
    const auto k = effectiveKind(out_kinds[j], os);
    const auto target = targetOf(k, os);
    if(target == DecodedOutput::None || taken[target])
      continue;

    auto d = decodeOutput(outs[j], k, resolveWriteMode(OutputMode::Auto, k), scratch);
    if(d.target == DecodedOutput::Depth && taken[DecodedOutput::Mask])
      d.r8.clear();
    claim(d);
    res.push_back(std::move(d));
  }
  return res;
}

template <typename Node>
void applyDecoded(Node& self, DecodedOutputs& ds)
{
  for(auto& d : ds)
    applyDecoded(self, d);
}
}
