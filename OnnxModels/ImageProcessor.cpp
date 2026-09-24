#include "ImageProcessor.hpp"

#include <OnnxModels/ImageDecode.hpp>
#include <OnnxModels/JobPool.hpp>

#include <Onnx/helpers/ImageOps.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <onnxruntime_cxx_api.h>

#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <cstring>
#include <random>
#include <span>

namespace OnnxModels
{
using Onnx::ImageModelKind;
using Onnx::ImgLayout;
using Onnx::TensorElemType;

namespace
{
using namespace imgdec;

// float32 -> IEEE half (binary16), round-to-nearest-even-ish. Only used for the
// rare fp16-input model; image data tolerates the simplified rounding.
uint16_t floatToHalf(float f) noexcept
{
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if(exp <= 0)
  {
    if(exp < -10)
      return (uint16_t)sign; // too small -> +/-0
    mant |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t h = mant >> shift;
    if((mant >> (shift - 1)) & 1u)
      h += 1; // round
    return (uint16_t)(sign | h);
  }
  if(exp >= 0x1F)
    return (uint16_t)(sign | 0x7C00u); // overflow -> inf
  uint32_t h = ((uint32_t)exp << 10) | (mant >> 13);
  if((mant >> 12) & 1u)
    h += 1; // round to nearest
  return (uint16_t)(sign | h);
}

// (mean, invstd) in the 0..255 sample domain expected by ImageOps' samplers.
void normConstants(InputNormalization n, float mean[3], float invstd[3])
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

Onnx::TensorLayout toTensorLayout(ImgLayout l, ChannelOrder ord)
{
  switch(l)
  {
    case ImgLayout::NhwcRgb:
      return Onnx::TensorLayout::NhwcRgb;
    case ImgLayout::NchwGray:
      return Onnx::TensorLayout::NchwGray;
    case ImgLayout::NchwRgb:
    default:
      return (ord == ChannelOrder::BGR) ? Onnx::TensorLayout::NchwBgr
                                        : Onnx::TensorLayout::NchwRgb;
  }
}

int snapDim(int v, int stride)
{
  if(stride <= 1)
    return std::max(1, v);
  int r = ((v + stride / 2) / stride) * stride;
  return std::max(stride, r);
}

// classifyImage reads input 0 as the image: put the model's image input first
// (FILM declares its `time` scalar before the two frames).
Onnx::ModelIO toImageIO(const Onnx::ModelSpec& s, int image_in = 0)
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

// Build the ORT input tensor from the preprocessed float buffer, converting to
// the model's declared element type. half/u8 staging buffers must outlive the
// returned Value (caller keeps them alive through inference).
Ort::Value buildInputTensor(
    const boost::container::vector<float>& input,
    const std::vector<int64_t>& ishape, TensorElemType in_dt,
    std::vector<uint16_t>& half_buf, std::vector<uint8_t>& u8_buf)
{
  const int64_t n = (int64_t)input.size();
  if(in_dt == TensorElemType::Float16)
  {
    half_buf.resize(n);
    for(int64_t i = 0; i < n; ++i)
      half_buf[i] = floatToHalf(input[i]);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor(
        mem, half_buf.data(), (size_t)n * sizeof(uint16_t), ishape.data(),
        ishape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  }
  if(in_dt == TensorElemType::Uint8)
  {
    u8_buf.resize(n);
    for(int64_t i = 0; i < n; ++i)
    {
      const float v = input[i];
      u8_buf[i] = (uint8_t)(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
    }
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<uint8_t>(
        mem, u8_buf.data(), (size_t)n, ishape.data(), ishape.size());
  }
  return Onnx::vec_to_tensor<float>(
      const_cast<boost::container::vector<float>&>(input), ishape);
}

// Heavy models run async so a slow inference can't freeze the render thread.
bool isHeavyModel(std::size_t model_bytes, int64_t in_pixels)
{
  return model_bytes > 64u * 1024 * 1024 || in_pixels > (int64_t)512 * 512;
}

// --- multi-input (image-pair / parametric) support ---

Onnx::ArchIO toArchIO(const Onnx::ModelSpec& s)
{
  Onnx::ArchIO io;
  io.inputs.reserve(s.inputs.size());
  io.outputs.reserve(s.outputs.size());
  for(const auto& p : s.inputs)
    io.inputs.push_back({p.name, p.shape, p.elem_type});
  for(const auto& p : s.outputs)
    io.outputs.push_back({p.name, p.shape, p.elem_type});
  return io;
}

std::vector<int64_t> concreteShape(const std::vector<int64_t>& s)
{
  std::vector<int64_t> r = s.empty() ? std::vector<int64_t>{1} : s;
  for(auto& d : r)
    if(d <= 0)
      d = 1;
  return r;
}

// A control input bound to Param 1/2: a Scalar, or a single-value "latent"
// such as FILM's time [?,1] (rank <= 2, no concrete dim above 1).
bool isScalarLike(Onnx::PortArchetype a, const std::vector<int64_t>& shape)
{
  if(a == Onnx::PortArchetype::Scalar)
    return true;
  return a == Onnx::PortArchetype::Latent && shape.size() <= 2
         && std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d <= 1; });
}

// Preprocess an RGBA texture into `dst` (float) per an input's declared layout +
// the node's norm/resize settings; returns the NCHW/NHWC/gray tensor shape.
std::vector<int64_t> preprocessTexture(
    const unsigned char* bytes, int tw, int th,
    const std::vector<int64_t>& declared, InputNormalization norm,
    ChannelOrder order, ResizeMode resize, int res_x, int res_y, int stride,
    boost::container::vector<float>& dst)
{
  const auto li = Onnx::detail::resolveLayout(declared);
  int mw = li.w, mh = li.h;
  if(mw <= 0)
    mw = snapDim(res_x, stride);
  if(mh <= 0)
    mh = snapDim(res_y, stride);
  const auto layout = toTensorLayout(li.layout, order);
  const int channels = (layout == Onnx::TensorLayout::NchwGray) ? 1 : 3;
  dst.resize((std::size_t)channels * mw * mh, boost::container::default_init);

  float mean[3], invstd[3];
  normConstants(norm, mean, invstd);
  const Onnx::ImageView src{bytes, tw, th, 4, 0};
  if(resize == ResizeMode::Letterbox)
  {
    Onnx::letterboxToTensor(
        layout, src, mw, mh, true, 0, mean, invstd, dst.data());
  }
  else
  {
    const float sw = (float)tw, sh = (float)th;
    Onnx::Affine a;
    if(resize == ResizeMode::Stretch)
      a = Onnx::Affine{sw / mw, 0, 0, 0, sh / mh, 0};
    else
    {
      const float step = std::min(sw / mw, sh / mh);
      a = Onnx::Affine{
          step, 0, (sw - step * mw) * 0.5f, 0, step, (sh - step * mh) * 0.5f};
    }
    Onnx::sampleAffineToTensor(
        layout, src, a, mw, mh, mean, invstd, dst.data(), Onnx::prof::WarpDet);
  }

  std::vector<int64_t> shape;
  if(layout == Onnx::TensorLayout::NhwcRgb)
    shape = {1, mh, mw, 3};
  else if(layout == Onnx::TensorLayout::NchwGray)
    shape = {1, 1, mh, mw};
  else
    shape = {1, 3, mh, mw};
  if(li.channels == 4)
    addZeroChannel(dst, shape);
  return shape;
}
} // namespace

ImageProcessor::ImageProcessor() noexcept
{
  // halp's texture struct has no member initializers: an Aux that no host
  // ever filled in would otherwise read as a garbage connected texture.
  inputs.image.texture = {};
  inputs.aux.texture = {};
  inputs.image.request_width = 512;
  inputs.image.request_height = 512;
  storage.reserve(512 * 512 * 3);
}

ImageProcessor::~ImageProcessor() = default;

void ImageProcessor::reloadModel()
{
  ctx = std::make_shared<Onnx::OnnxRunContext>(
      inputs.model.file.bytes, inputs.model.file.filename);
  spec = ctx->readModelSpec();

  // Resolve multi-input roles: 1st image -> In, 2nd image -> Aux, scalar inputs
  // -> Param 1/2. Single-input models keep multi_input=false (unchanged path).
  march = Onnx::classifyModel(toArchIO(spec));
  image_input_index = -1;
  aux_input_index = -1;
  param_in[0] = param_in[1] = -1;
  int img_seen = 0, param_seen = 0;
  for(int i = 0; i < (int)march.inputs.size(); ++i)
  {
    const auto a = march.inputs[i].arch;
    if(a == Onnx::PortArchetype::Image)
    {
      if(img_seen == 0)
        image_input_index = i;
      else if(img_seen == 1)
        aux_input_index = i;
      ++img_seen;
    }
    else if(isScalarLike(a, spec.inputs[i].shape) && param_seen < 2)
    {
      param_in[param_seen++] = i;
    }
  }
  has_image_input = image_input_index >= 0;
  if(image_input_index < 0)
    image_input_index = 0;
  multi_input = spec.inputs.size() > 1;

  role = Onnx::classifyImage(
      toImageIO(spec, image_input_index), inputs.output_index.value);
  {
    const auto io = toImageIO(spec, image_input_index);
    out_kinds.clear();
    for(int j = 0; j < (int)spec.outputs.size(); ++j)
      out_kinds.push_back(Onnx::classifyImage(io, j).kind);
  }

  // Invalidate the latent-generation cache: a different GAN may produce an
  // identical seed-derived latent, and without this the `latent == last_latent`
  // short-circuit would keep showing the previous model's output.
  produced_latent = false;
  last_latent.clear();

  lastModelPath = inputs.model.file.filename;
  lastOutputIndex = inputs.output_index.value;
}

void ImageProcessor::operator()()
try
{
  ONNX_PROF_SCOPE(Total);
  if(!available)
    return;
  if(inputs.model.current_model_invalid)
    return;
  if(inputs.model.file.bytes.empty())
    return;

  // (Re)load / (re)classify on model or output-index change.
  if(!ctx || lastModelPath != inputs.model.file.filename)
  {
    if(!loadModel([this] { reloadModel(); }, inputs.model, name()))
      return;
  }
  else if(inputs.output_index.value != lastOutputIndex)
  {
    role = Onnx::classifyImage(
        toImageIO(spec, image_input_index), inputs.output_index.value);
    lastOutputIndex = inputs.output_index.value;
  }
  if(spec.inputs.empty() || spec.outputs.empty())
    return;
  if(!failures.ready())
    return; // the last frames failed the same way: backing off

  // A model with an image input is never run from a latent, whatever its
  // other inputs look like.
  if(!has_image_input
     && applyTaskOverride(role.kind, inputs.task.value)
            == ImageModelKind::LatentToImage)
    runLatent();
  else
    runImage();
}
catch(const std::exception& e)
{
  // A frame that fails is reported and skipped; the node keeps running.
  failures.failed(name(), inputs.model.file.filename, e.what());
}
catch(...)
{
  failures.failed(name(), inputs.model.file.filename, "unknown error");
}

// image -> image / mask / depth / data
void ImageProcessor::runImage()
{
  const ImageModelKind kind = applyTaskOverride(role.kind, inputs.task.value);
  auto& in_tex = inputs.image.texture;
  if(!in_tex.changed || !in_tex.bytes || in_tex.width <= 0 || in_tex.height <= 0)
    return;

  // Multi-input models (image-pair / parametric) take the synchronous binding
  // path; single-input models keep the original sync-or-async fast path below.
  if(multi_input)
  {
    runImageMulti();
    return;
  }

  // --- resolve model input size (fixed dims win; else snapped user res) ------
  int mw = role.in_w, mh = role.in_h;
  if(mw <= 0)
    mw = snapDim(inputs.resolution.value.x, role.in_stride);
  if(mh <= 0)
    mh = snapDim(inputs.resolution.value.y, role.in_stride);

  // --- preprocess (fused sample + normalize via ImageOps) --------------------
  float mean[3], invstd[3];
  normConstants(inputs.normalization.value, mean, invstd);
  const auto layout = toTensorLayout(role.in_layout, inputs.channel_order.value);
  const int channels = (layout == Onnx::TensorLayout::NchwGray) ? 1 : 3;
  storage.resize((std::size_t)channels * mw * mh, boost::container::default_init);

  const Onnx::ImageView src{in_tex.bytes, in_tex.width, in_tex.height, 4, 0};
  if(inputs.resize_mode.value == ResizeMode::Letterbox)
  {
    Onnx::letterboxToTensor(
        layout, src, mw, mh, /*center*/ true, /*pad*/ 0, mean, invstd,
        storage.data());
  }
  else
  {
    const float sw = (float)in_tex.width, sh = (float)in_tex.height;
    Onnx::Affine a;
    if(inputs.resize_mode.value == ResizeMode::Stretch)
    {
      a = Onnx::Affine{sw / mw, 0, 0, 0, sh / mh, 0};
    }
    else // Crop: keep aspect, fill, center-crop the overflow
    {
      const float step = std::min(sw / mw, sh / mh);
      a = Onnx::Affine{
          step, 0, (sw - step * mw) * 0.5f, 0, step, (sh - step * mh) * 0.5f};
    }
    Onnx::sampleAffineToTensor(
        layout, src, a, mw, mh, mean, invstd, storage.data(),
        Onnx::prof::WarpDet);
  }

  // --- input shape + dispatch (sync, or async for heavy models) --------------
  std::vector<int64_t> ishape;
  if(layout == Onnx::TensorLayout::NhwcRgb)
    ishape = {1, mh, mw, 3};
  else if(layout == Onnx::TensorLayout::NchwGray)
    ishape = {1, 1, mh, mw};
  else
    ishape = {1, 3, mh, mw};
  if(role.in_channels == 4)
    addZeroChannel(storage, ishape);

  const Onnx::WriteMode wm = resolveWriteMode(inputs.output_mode.value, kind);
  const bool heavy
      = isHeavyModel(inputs.model.file.bytes.size(), (int64_t)mw * mh);
  dispatchInfer(std::move(ishape), kind, wm, heavy);
}

// Multi-input image models (image-pair: inpaint/flow/stereo; parametric). Builds
// the full declared input array (primary image, optional Aux image, scalar
// Params, zero-filled leftovers) and runs synchronously. Single-input models
// never reach here. Backing buffers (members + the in-scope locals) outlive the
// synchronous infer() call.
void ImageProcessor::runImageMulti()
{
  const ImageModelKind kind = applyTaskOverride(role.kind, inputs.task.value);
  const Onnx::WriteMode wm = resolveWriteMode(inputs.output_mode.value, kind);
  const int nin = (int)spec.inputs.size();
  const int nout = (int)spec.output_names_char.size();
  if(nin <= 0 || nout <= 0)
    return;

  const int rx = inputs.resolution.value.x, ry = inputs.resolution.value.y;
  const int stride = role.in_stride;

  std::vector<Ort::Value> ins;
  ins.reserve(nin);
  for(int i = 0; i < nin; ++i)
    ins.emplace_back(nullptr);

  // Primary image.
  {
    auto& t = inputs.image.texture;
    const auto ish = preprocessTexture(
        t.bytes, t.width, t.height, spec.inputs[image_input_index].shape,
        inputs.normalization.value, inputs.channel_order.value,
        inputs.resize_mode.value, rx, ry, stride, storage);
    ins[image_input_index] = buildInputTensor(
        storage, ish, spec.inputs[image_input_index].elem_type, half_buf,
        u8_buf);
  }

  // Optional 2nd image (Aux). If unconnected, feed a zeroed frame of the right
  // size so the model still runs.
  if(aux_input_index >= 0 && aux_input_index < nin)
  {
    auto& t = inputs.aux.texture;
    std::vector<int64_t> ish;
    if(t.bytes && t.width > 0 && t.height > 0)
    {
      ish = preprocessTexture(
          t.bytes, t.width, t.height, spec.inputs[aux_input_index].shape,
          inputs.normalization.value, inputs.channel_order.value,
          inputs.resize_mode.value, rx, ry, stride, aux_storage);
    }
    else
    {
      const auto li
          = Onnx::detail::resolveLayout(spec.inputs[aux_input_index].shape);
      int mw = li.w > 0 ? li.w : snapDim(rx, stride);
      int mh = li.h > 0 ? li.h : snapDim(ry, stride);
      const int ch = (li.layout == Onnx::ImgLayout::NchwGray) ? 1 : 3;
      auto& pt = inputs.image.texture;
      if(ch == 3 && pt.bytes && pt.width > 0 && pt.height > 0)
      {
        // A 3-channel Aux is a SECOND IMAGE (frame interpolation: CAIN/RIFE).
        // With Aux unconnected, feeding zeros makes the model interpolate toward
        // a black frame -> dark/blurry garbage. Default it to the PRIMARY frame
        // (so an unwired interpolator gives ~identity, not charcoal).
        ish = preprocessTexture(
            pt.bytes, pt.width, pt.height, spec.inputs[aux_input_index].shape,
            inputs.normalization.value, inputs.channel_order.value,
            inputs.resize_mode.value, rx, ry, stride, aux_storage);
      }
      else
      {
        // A 1-channel Aux is a mask (inpainting): zeros = "no hole" = identity.
        aux_storage.assign((std::size_t)ch * mw * mh, 0.f);
        ish = (li.layout == Onnx::ImgLayout::NhwcRgb)
                  ? std::vector<int64_t>{1, mh, mw, 3}
                  : (ch == 1 ? std::vector<int64_t>{1, 1, mh, mw}
                             : std::vector<int64_t>{1, 3, mh, mw});
      }
    }
    ins[aux_input_index] = buildInputTensor(
        aux_storage, ish, spec.inputs[aux_input_index].elem_type, aux_half_buf,
        aux_u8_buf);
  }

  // Scalar params (single-element float tensors; their storage must outlive Run).
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  float pv[2] = {inputs.param1.value, inputs.param2.value};
  for(int k = 0; k < 2; ++k)
  {
    if(param_in[k] >= 0 && param_in[k] < nin)
    {
      auto sh = concreteShape(spec.inputs[param_in[k]].shape);
      ins[param_in[k]]
          = Ort::Value::CreateTensor<float>(mem, &pv[k], 1, sh.data(), sh.size());
    }
  }

  // Zero-fill any input we didn't bind (defensive; keeps the model runnable).
  std::vector<boost::container::vector<float>> zero_bufs(nin);
  for(int i = 0; i < nin; ++i)
  {
    if(ins[i])
      continue;
    auto sh = concreteShape(spec.inputs[i].shape);
    int64_t n = 1;
    for(auto d : sh)
      n *= d;
    zero_bufs[i].assign((std::size_t)n, 0.f);
    ins[i] = Onnx::vec_to_tensor<float>(zero_bufs[i], sh);
  }

  std::vector<Ort::Value> outs;
  outs.reserve(nout);
  for(int i = 0; i < nout; ++i)
    outs.emplace_back(nullptr);
  ctx->infer(spec, ins, outs);

  const int idx = std::clamp(inputs.output_index.value, 0, nout - 1);
  auto ds = decodeAll(outs, idx, kind, wm, out_kinds, out_scratch);
  applyDecoded(*this, ds);
}

// Run inference + decode synchronously, or hand a snapshot to the worker thread
// for heavy/generative models (latest-wins: drop frames while one is in flight).
void ImageProcessor::dispatchInfer(
    std::vector<int64_t> ishape, ImageModelKind kind, Onnx::WriteMode wm,
    bool force_async)
{
  const TensorElemType in_dt = spec.inputs[0].elem_type;

  if(force_async)
  {
    if(inferenceInProgress)
      return; // a job is already running; drop this frame
    inferenceInProgress = true;
    // Pooled job: lock-free acquire; recycled vectors keep their capacity so
    // the assignments below don't allocate in steady state.
    auto job = JobPool<InferJob>::instance().acquire();
    job->ctx = ctx;
    // Swap (not copy) the preprocessed input into the job: `storage` is fully
    // rewritten next frame, so it can take the job's recycled buffer back and we
    // avoid a multi-MB memcpy every async frame.
    std::swap(job->input, storage);
    job->ishape = std::move(ishape);
    job->in_dt = in_dt;
    job->output_index = inputs.output_index.value;
    job->kind = kind;
    job->wm = wm;
    job->out_kinds = out_kinds;
    worker.request(std::move(job));
    return;
  }

  Ort::Value in_val
      = buildInputTensor(storage, ishape, in_dt, half_buf, u8_buf);
  const int nout = (int)spec.output_names_char.size();
  std::vector<Ort::Value> outs;
  outs.reserve(nout);
  for(int i = 0; i < nout; ++i)
    outs.emplace_back(nullptr);
  Ort::Value ins[1] = {std::move(in_val)};
  ctx->infer(spec, ins, outs);

  const int idx = std::clamp(inputs.output_index.value, 0, nout - 1);
  auto ds = decodeAll(outs, idx, kind, wm, out_kinds, out_scratch);
  applyDecoded(*this, ds);
}

std::function<void(ImageProcessor&)>
ImageProcessor::worker::work(std::unique_ptr<InferJob> job)
{
  // RAII: whatever path we exit through, the job goes back to the lock-free
  // pool (with its buffer capacities intact) once the results are moved out.
  struct Recycle
  {
    std::unique_ptr<InferJob>& j;
    ~Recycle()
    {
      if(j)
        j->ctx.reset(); // don't keep the ORT session alive from the pool
      JobPool<InferJob>::instance().release(std::move(j));
    }
  } recycle{job};

  if(!job || !job->ctx)
    return [](ImageProcessor& self) { self.inferenceInProgress = false; };
  try
  {
    const auto& spec = job->ctx->readModelSpec();
    std::vector<uint16_t> half_buf;
    std::vector<uint8_t> u8_buf;
    Ort::Value in_val = buildInputTensor(
        job->input, job->ishape, job->in_dt, half_buf, u8_buf);

    const int nout = (int)spec.output_names_char.size();
    std::vector<Ort::Value> outs;
    outs.reserve(nout);
    for(int i = 0; i < nout; ++i)
      outs.emplace_back(nullptr);
    Ort::Value ins[1] = {std::move(in_val)};
    job->ctx->infer(spec, ins, outs);

    const int idx = std::clamp(job->output_index, 0, nout - 1);
    std::vector<float> scratch;
    auto ds = decodeAll(outs, idx, job->kind, job->wm, job->out_kinds, scratch);
    return [ds = std::move(ds)](ImageProcessor& self) mutable
    {
      self.inferenceInProgress = false;
      applyDecoded(self, ds);
    };
  }
  catch(const std::exception& e)
  {
    return [what = std::string(e.what())](ImageProcessor& self)
    {
      self.inferenceInProgress = false;
      self.failures.failed(ImageProcessor::name(), self.inputs.model.file.filename, what);
    };
  }
  catch(...)
  {
    return [](ImageProcessor& self)
    {
      self.inferenceInProgress = false;
      self.failures.failed(ImageProcessor::name(), self.inputs.model.file.filename, "unknown error");
    };
  }
}

// data -> image (latent vector input; single-network generative models).
// Always runs async (generative models are heavy and have no per-frame input).
void ImageProcessor::runLatent()
{
  const int dim = role.latent_dim;
  if(dim <= 0)
    return;
  if(inferenceInProgress)
    return; // a generation is in flight; don't queue another

  // Deterministic latent from the seed, overlaid with user-supplied dims, scaled.
  latent_vector.resize(dim);
  std::mt19937 gen((uint32_t)inputs.seed.value);
  // GAN z-latents are standard-normal N(0,1); the args are (mean, stddev), not
  // a [lo,hi] range, so (-1,1) would shift every latent a sigma off-manifold.
  std::normal_distribution<float> dist(0.f, 1.f);
  for(float& v : latent_vector)
    v = dist(gen);
  const int nuser = std::min<int>(dim, (int)inputs.latent.value.size());
  for(int i = 0; i < nuser; ++i)
    latent_vector[i] = inputs.latent.value[i];
  const float sc = inputs.latent_scale.value;
  for(float& v : latent_vector)
    v *= sc;

  // Skip re-generation when the latent is unchanged.
  if(produced_latent && latent_vector == last_latent)
    return;

  // Input tensor: the model's declared latent shape with every non-positive
  // (dynamic -1 / batch) dim resolved. The single dynamic axis carries the
  // latent length; any other dynamic dim becomes 1. Leaving a -1 in the shape
  // would make CreateTensor throw and permanently invalidate the model.
  std::vector<int64_t> ishape = spec.inputs[0].shape;
  if(ishape.empty())
  {
    ishape = {1, dim};
  }
  else
  {
    int dyn_axis = -1, dyn_count = 0;
    for(std::size_t i = 0; i < ishape.size(); ++i)
      if(ishape[i] <= 0)
      {
        ++dyn_count;
        dyn_axis = (int)i;
      }
    for(auto& d : ishape)
      if(d <= 0)
        d = 1;
    // Exactly one dynamic axis -> it holds the latent vector.
    if(dyn_count == 1)
      ishape[dyn_axis] = dim;
  }

  // Hand the latent to the shared dispatch as the input buffer (always async).
  storage.assign(latent_vector.begin(), latent_vector.end());
  const Onnx::WriteMode wm
      = resolveWriteMode(inputs.output_mode.value, ImageModelKind::LatentToImage);
  last_latent = latent_vector;
  produced_latent = true;
  dispatchInfer(
      std::move(ishape), ImageModelKind::LatentToImage, wm, /*force_async*/ true);
}

}
