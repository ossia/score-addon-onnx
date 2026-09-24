#include "VideoProcessor.hpp"

#include <OnnxModels/ImageDecode.hpp>
#include <OnnxModels/JobPool.hpp>

#include <Onnx/helpers/ImageOps.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>

namespace OnnxModels
{
using Onnx::ImageModelKind;
using Onnx::ImgLayout;
using Onnx::PortArchetype;
using Onnx::TensorElemType;

namespace
{
using namespace imgdec;

// float32 -> IEEE half. Only used for rare fp16-input models (image tolerates
// the simplified rounding). Copied verbatim from ImageProcessor.cpp.
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
      return (uint16_t)sign;
    mant |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t h = mant >> shift;
    if((mant >> (shift - 1)) & 1u)
      h += 1;
    return (uint16_t)(sign | h);
  }
  if(exp >= 0x1F)
    return (uint16_t)(sign | 0x7C00u);
  uint32_t h = ((uint32_t)exp << 10) | (mant >> 13);
  if((mant >> 12) & 1u)
    h += 1;
  return (uint16_t)(sign | h);
}

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

// Build an ImageModelRole-style ModelIO using ONLY the primary image input and
// the outputs, so classifyImage() ignores the recurrent-state / scalar ports.
Onnx::ModelIO toImageIO(const Onnx::ModelSpec& s, int image_in)
{
  Onnx::ModelIO io;
  io.inputs.push_back({s.inputs[image_in].name, s.inputs[image_in].shape});
  io.outputs.reserve(s.outputs.size());
  for(const auto& p : s.outputs)
    io.outputs.push_back({p.name, p.shape});
  return io;
}

// Build the dependency-free ArchIO (with dtypes) the archetype classifier wants.
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

// Map a TensorElemType to the ORT enum (for raw-bytes state tensors).
ONNXTensorElementDataType toOrtType(TensorElemType t)
{
  switch(t)
  {
    case TensorElemType::Float:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case TensorElemType::Float16:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case TensorElemType::BFloat16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    case TensorElemType::Double:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case TensorElemType::Uint8:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case TensorElemType::Int8:     return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case TensorElemType::Uint16:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case TensorElemType::Int16:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case TensorElemType::Uint32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case TensorElemType::Int32:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case TensorElemType::Uint64:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case TensorElemType::Int64:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case TensorElemType::Bool:     return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:                       return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
}

// A state tensor over raw bytes (any dtype). The buffer must outlive the Value.
Ort::Value stateTensor(StateSlot& s)
{
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  return Ort::Value::CreateTensor(
      mem, s.buffer.data(), s.buffer.size(), s.shape.data(), s.shape.size(),
      toOrtType(s.dtype));
}

bool isHeavyModel(std::size_t model_bytes, int64_t in_pixels)
{
  return model_bytes > 64u * 1024 * 1024 || in_pixels > (int64_t)512 * 512;
}

// Resolve a possibly-dynamic shape to a concrete one: dynamic dims (<=0) -> 1.
// Used for the FIRST-frame zero state (RVM accepts a minimal [1,C,1,1] init).
std::vector<int64_t> concreteShape(const std::vector<int64_t>& s)
{
  std::vector<int64_t> r = s;
  for(auto& d : r)
    if(d <= 0)
      d = 1;
  return r;
}

int64_t elemCount(const std::vector<int64_t>& s)
{
  int64_t n = 1;
  for(auto d : s)
    n *= (d > 0 ? d : 1);
  return n;
}

// Core inference + state threading, shared by sync and worker paths. Builds the
// FULL input array (image + scalars + held states) in declared order, runs, then
// (1) copies matching state outputs back into the slot buffers, resizing them to
// the now-concrete output shapes, and (2) decodes the primary visual output,
// then routes the other outputs to the outlets it left free (RVM: fgr to Image
// and pha to Mask in the same run).
DecodedOutputs runInferAndThread(
    Onnx::OnnxRunContext& ctx, const Onnx::ModelSpec& spec,
    const boost::container::vector<float>& image_input,
    const std::vector<int64_t>& ishape, int image_input_index,
    TensorElemType in_dt, int param0_in, float param0, int param1_in,
    float param1, std::vector<StateSlot>& states, int output_index,
    ImageModelKind kind, Onnx::WriteMode wm,
    const std::vector<ImageModelKind>& out_kinds, std::vector<uint16_t>& half_buf,
    std::vector<uint8_t>& u8_buf, std::vector<float>& scratch)
{
  const int nin = (int)spec.inputs.size();
  const int nout = (int)spec.outputs.size();

  // Scalar param scratch must outlive the Run() call.
  float pv0 = param0, pv1 = param1;

  std::vector<Ort::Value> ins;
  ins.reserve(nin);
  for(int i = 0; i < nin; ++i)
    ins.emplace_back(nullptr);

  // Primary image input.
  ins[image_input_index]
      = buildInputTensor(image_input, ishape, in_dt, half_buf, u8_buf);

  // Scalar params (e.g. downsample_ratio): a single-element tensor of the
  // declared shape/dtype. Only float is supported (RVM uses fp32).
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  if(param0_in >= 0 && param0_in < nin)
  {
    auto sh = concreteShape(spec.inputs[param0_in].shape);
    ins[param0_in] = Ort::Value::CreateTensor<float>(
        mem, &pv0, 1, sh.data(), sh.size());
  }
  if(param1_in >= 0 && param1_in < nin)
  {
    auto sh = concreteShape(spec.inputs[param1_in].shape);
    ins[param1_in] = Ort::Value::CreateTensor<float>(
        mem, &pv1, 1, sh.data(), sh.size());
  }

  // Held recurrent states.
  for(auto& s : states)
    if(s.input_index >= 0 && s.input_index < nin)
      ins[s.input_index] = stateTensor(s);

  std::vector<Ort::Value> outs;
  outs.reserve(nout);
  for(int i = 0; i < nout; ++i)
    outs.emplace_back(nullptr);

  ctx.infer(spec, ins, outs);

  // Copy matching state OUTPUTS back into the slot buffers (sized to the now
  // concrete output shapes — RVM's r#o grow from [1,C,1,1] to the real spatial
  // size on the first frame, then stay fixed).
  for(auto& s : states)
  {
    if(s.output_index < 0 || s.output_index >= nout)
      continue;
    Ort::Value& ov = outs[s.output_index];
    if(!ov)
      continue;
    const auto oinfo = ov.GetTensorTypeAndShapeInfo();
    s.shape = oinfo.GetShape();
    for(auto& d : s.shape)
      if(d <= 0)
        d = 1;
    s.count = elemCount(s.shape);
    s.dtype = Onnx::fromOrtElementType(oinfo.GetElementType());
    const size_t bytes = (size_t)s.count * Onnx::elemSize(s.dtype);
    s.buffer.resize(bytes);
    std::memcpy(s.buffer.data(), ov.GetTensorData<uint8_t>(), bytes);
  }

  // The state outputs only feed the next frame: drop them so they are not
  // routed to an outlet (RVM's 16-channel r1o would pass for an image), unless
  // the Output Index picks one.
  const int idx = std::clamp(output_index, 0, nout - 1);
  for(auto& s : states)
    if(s.output_index >= 0 && s.output_index < nout && s.output_index != idx)
      outs[s.output_index] = Ort::Value{nullptr};
  return decodeAll(outs, idx, kind, wm, out_kinds, scratch);
}
} // namespace

VideoProcessor::VideoProcessor() noexcept
{
  inputs.image.request_width = 512;
  inputs.image.request_height = 512;
  storage.reserve(512 * 512 * 3);
}

VideoProcessor::~VideoProcessor() = default;

// Identify the primary image input, the scalar param inputs, and the recurrent
// state slots (RecurrentState input -> matching output) using classifyModel.
void VideoProcessor::detectStates()
{
  arch = Onnx::classifyModel(toArchIO(spec));
  states.clear();
  imageInputIndex = 0;
  param0InputIndex = param1InputIndex = -1;

  const int nin = (int)spec.inputs.size();

  // Recurrent-state slots: the RecurrentState inputs, each fed back from the
  // output classifyModel paired it with (r#i/r#o and *_in/*_out names first,
  // then free same-shape outputs). classifyModel also recognises the name
  // pairs whose shapes are fully symbolic (RVM's r1i..r4i look like images).
  std::vector<char> is_state(nin, 0);
  for(int i = 0; i < nin; ++i)
  {
    if(arch.inputs[i].arch != PortArchetype::RecurrentState)
      continue;
    is_state[i] = 1;
    StateSlot s;
    s.input_index = i;
    s.output_index = arch.inputs[i].state_pair;
    s.dtype = spec.inputs[i].elem_type;
    s.shape = concreteShape(spec.inputs[i].shape);
    s.count = elemCount(s.shape);
    states.push_back(std::move(s));
  }

  // Primary image input: first non-state input classified Image (RVM: 'src').
  for(int i = 0; i < nin; ++i)
    if(!is_state[i] && arch.inputs[i].arch == PortArchetype::Image)
    {
      imageInputIndex = i;
      break;
    }

  // Scalar params (downsample_ratio etc.): non-state Scalar inputs, in order.
  for(int i = 0; i < nin; ++i)
  {
    if(!is_state[i] && arch.inputs[i].arch == PortArchetype::Scalar)
    {
      if(param0InputIndex < 0)
        param0InputIndex = i;
      else if(param1InputIndex < 0)
        param1InputIndex = i;
    }
  }
  zeroStates();
}

// Reset every recurrent state to its INITIAL declared size (dynamic dims -> 1)
// and zero it, so the model re-initializes the recurrence on the next frame.
// Resetting the SHAPE (not just zeroing the current buffer) is essential: once a
// frame has run, each slot's shape was adopted to the real spatial size (RVM:
// [1,C,1,1] -> [1,16,64,64]). After a downsample-ratio / resolution change the
// model computes a DIFFERENT internal size, and feeding the stale-shaped state
// makes an internal Expand fail to broadcast -> ORT throws every frame and the
// node stops. Shrinking back to [1,C,1,1] lets the model re-grow it cleanly.
void VideoProcessor::zeroStates()
{
  for(auto& s : states)
  {
    if(s.input_index >= 0 && s.input_index < (int)spec.inputs.size())
    {
      s.shape = concreteShape(spec.inputs[s.input_index].shape);
      s.count = elemCount(s.shape);
    }
    const size_t bytes = (size_t)s.count * Onnx::elemSize(s.dtype);
    s.buffer.assign(bytes, 0);
  }
}

void VideoProcessor::reloadModel()
{
  ctx = std::make_shared<Onnx::OnnxRunContext>(
      inputs.model.file.bytes, inputs.model.file.filename);
  spec = ctx->readModelSpec();
  detectStates();
  const auto io = toImageIO(spec, imageInputIndex);
  role = Onnx::classifyImage(io, inputs.output_index.value);
  out_kinds.clear();
  for(int j = 0; j < (int)spec.outputs.size(); ++j)
    out_kinds.push_back(Onnx::classifyImage(io, j).kind);
  lastModelPath = inputs.model.file.filename;
  lastOutputIndex = inputs.output_index.value;
  ++gen;
  resetPending = false;
  preferAsync = false;
}

void VideoProcessor::operator()()
try
{
  ONNX_PROF_SCOPE(Total);
  if(!available)
    return;
  if(inputs.model.current_model_invalid)
    return;
  if(inputs.model.file.bytes.empty())
    return;

  if(!ctx || lastModelPath != inputs.model.file.filename)
  {
    if(!loadModel([this] { reloadModel(); }, inputs.model, name()))
      return;
  }
  else if(inputs.output_index.value != lastOutputIndex)
  {
    role = Onnx::classifyImage(toImageIO(spec, imageInputIndex),
                               inputs.output_index.value);
    lastOutputIndex = inputs.output_index.value;
  }
  if(spec.inputs.empty() || spec.outputs.empty())
    return;

  // Reset impulse: re-zero all recurrent state. While a job is in flight the
  // worker owns the state, so the reset waits for it; its result is from
  // before the reset and does not bring the old states back (gen).
  if(inputs.reset)
  {
    ++gen;
    resetPending = true;
  }
  if(resetPending && !inferenceInProgress)
  {
    zeroStates();
    resetPending = false;
  }

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

void VideoProcessor::runImage()
{
  const ImageModelKind kind = applyTaskOverride(role.kind, inputs.task.value);
  auto& in_tex = inputs.image.texture;
  if(!in_tex.changed || !in_tex.bytes || in_tex.width <= 0 || in_tex.height <= 0)
    return;

  int mw = role.in_w, mh = role.in_h;
  if(mw <= 0)
    mw = snapDim(inputs.resolution.value.x, role.in_stride);
  if(mh <= 0)
    mh = snapDim(inputs.resolution.value.y, role.in_stride);

  // Re-zero the recurrent state when the geometry it was produced at changes
  // (input size or a downsample-ratio param). Carrying a stale-geometry state
  // into a recurrent model (RVM) makes an internal Expand fail to broadcast and
  // ORT throws every frame. Skip while a job is in flight (the worker owns the
  // state); the next frame re-zeros cleanly.
  if(!inferenceInProgress && !states.empty()
     && (mw != stateGeomW || mh != stateGeomH
         || inputs.param0.value != stateGeomP0
         || inputs.param1.value != stateGeomP1))
  {
    zeroStates();
    preferAsync = false; // the cost depends on the geometry: measure again
    stateGeomW = mw;
    stateGeomH = mh;
    stateGeomP0 = inputs.param0.value;
    stateGeomP1 = inputs.param1.value;
  }

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
    else
    {
      const float step = std::min(sw / mw, sh / mh);
      a = Onnx::Affine{
          step, 0, (sw - step * mw) * 0.5f, 0, step, (sh - step * mh) * 0.5f};
    }
    Onnx::sampleAffineToTensor(
        layout, src, a, mw, mh, mean, invstd, storage.data(),
        Onnx::prof::WarpDet);
  }

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
  const bool heavy = preferAsync
                     || isHeavyModel(inputs.model.file.bytes.size(), (int64_t)mw * mh);
  dispatchInfer(std::move(ishape), kind, wm, heavy);
}

void VideoProcessor::dispatchInfer(
    std::vector<int64_t> ishape, ImageModelKind kind, Onnx::WriteMode wm,
    bool force_async)
{
  const TensorElemType in_dt = spec.inputs[imageInputIndex].elem_type;
  const int out_idx = std::clamp(
      inputs.output_index.value, 0, (int)spec.outputs.size() - 1);

  if(force_async)
  {
    if(inferenceInProgress)
      return; // a job is already running; drop this frame (latest-wins)
    inferenceInProgress = true;
    // Pooled job: lock-free acquire; recycled vectors keep their capacity so
    // the assignments below don't allocate in steady state.
    auto job = JobPool<VideoInferJob>::instance().acquire();
    job->ctx = ctx;
    // Swap (not copy) the preprocessed input: `storage` is fully rewritten next
    // frame, so it can take the job's recycled buffer back and we avoid a
    // multi-MB memcpy each async frame. (States below MUST stay a copy — the
    // node keeps them as the next frame's recurrent input.)
    std::swap(job->input, storage);
    job->ishape = std::move(ishape);
    job->image_input_index = imageInputIndex;
    job->in_dt = in_dt;
    job->output_index = out_idx;
    job->kind = kind;
    job->wm = wm;
    job->param0 = inputs.param0.value;
    job->param1 = inputs.param1.value;
    job->param0_input_index = param0InputIndex;
    job->param1_input_index = param1InputIndex;
    job->states = states; // snapshot held states into the job (whole-vector
                          // assignment: replaces any recycled leftovers)
    job->out_kinds = out_kinds;
    job->gen = gen;
    worker.request(std::move(job));
    return;
  }

  // The size heuristic misses what dominates the cost (RVM's downsample
  // ratio, the execution provider): a run over the budget moves the model to
  // the worker, so the render thread blocks for one frame at most.
  const auto t0 = std::chrono::steady_clock::now();
  auto ds = runInferAndThread(
      *ctx, spec, storage, ishape, imageInputIndex, in_dt, param0InputIndex,
      inputs.param0.value, param1InputIndex, inputs.param1.value, states,
      out_idx, kind, wm, out_kinds, half_buf, u8_buf, out_scratch);
  if(std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(8))
    preferAsync = true;
  applyDecoded(*this, ds);
}

std::function<void(VideoProcessor&)>
VideoProcessor::worker::work(std::unique_ptr<VideoInferJob> job)
{
  // RAII: whatever path we exit through, the job goes back to the lock-free
  // pool (with its buffer capacities intact) once the results are moved out.
  struct Recycle
  {
    std::unique_ptr<VideoInferJob>& j;
    ~Recycle()
    {
      if(j)
        j->ctx.reset(); // don't keep the ORT session alive from the pool
      JobPool<VideoInferJob>::instance().release(std::move(j));
    }
  } recycle{job};

  if(!job || !job->ctx)
    return [](VideoProcessor& self) { self.inferenceInProgress = false; };
  try
  {
    const auto& spec = job->ctx->readModelSpec();
    std::vector<uint16_t> half_buf;
    std::vector<uint8_t> u8_buf;
    std::vector<float> scratch;

    auto ds = runInferAndThread(
        *job->ctx, spec, job->input, job->ishape, job->image_input_index,
        job->in_dt, job->param0_input_index, job->param0,
        job->param1_input_index, job->param1, job->states, job->output_index,
        job->kind, job->wm, job->out_kinds, half_buf, u8_buf, scratch);

    // Hand the UPDATED state buffers + decoded image back to the node. Adopting
    // job->states on the node keeps the recurrent chain coherent across frames.
    return [ds = std::move(ds), states = std::move(job->states), gen = job->gen](
               VideoProcessor& self) mutable
    {
      self.inferenceInProgress = false;
      // Only adopt states from the current model, and not across a Reset.
      if(gen == self.gen && self.states.size() == states.size())
        self.states = std::move(states);
      applyDecoded(self, ds);
    };
  }
  catch(const std::exception& e)
  {
    return [what = std::string(e.what())](VideoProcessor& self)
    {
      self.inferenceInProgress = false;
      self.failures.failed(VideoProcessor::name(), self.inputs.model.file.filename, what);
    };
  }
  catch(...)
  {
    return [](VideoProcessor& self)
    {
      self.inferenceInProgress = false;
      self.failures.failed(VideoProcessor::name(), self.inputs.model.file.filename, "unknown error");
    };
  }
}

}
