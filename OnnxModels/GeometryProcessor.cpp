#include "GeometryProcessor.hpp"

#include <OnnxModels/JobPool.hpp>

#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>

namespace OnnxModels
{
using Onnx::GeomOutputKind;
using Onnx::PointAxisLayout;
using Onnx::PointLayout;
using Onnx::TensorElemType;

namespace
{
// float32 -> IEEE half (binary16), round-to-nearest-even-ish. Only used for the
// rare fp16-input point model (same routine as ImageProcessor).
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

// Index of the first rank-3 point-set-like input in the spec, or -1.
int findCloudInput(const Onnx::ModelSpec& spec)
{
  for(int i = 0; i < (int)spec.inputs.size(); ++i)
    if(Onnx::detectPointLayout(spec.inputs[i].shape).valid())
      return i;
  return -1;
}

// In-place xyz normalization over an interleaved payload (stride floats/point).
// Only the first 3 components (coordinates) are touched; features pass through.
void normalizeCloud(
    std::vector<float>& cloud, int64_t npoints, int stride,
    PointNormalization mode)
{
  if(mode == PointNormalization::None || npoints <= 0)
    return;

  float cx = 0, cy = 0, cz = 0;
  if(mode == PointNormalization::UnitSphere)
  {
    for(int64_t p = 0; p < npoints; ++p)
    {
      const float* v = cloud.data() + p * stride;
      cx += v[0];
      cy += v[1];
      cz += v[2];
    }
    const float inv = 1.f / (float)npoints;
    cx *= inv;
    cy *= inv;
    cz *= inv;
    float maxr = 0.f;
    for(int64_t p = 0; p < npoints; ++p)
    {
      const float* v = cloud.data() + p * stride;
      const float dx = v[0] - cx, dy = v[1] - cy, dz = v[2] - cz;
      const float r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if(r > maxr)
        maxr = r;
    }
    const float s = (maxr > 1e-9f) ? 1.f / maxr : 1.f;
    for(int64_t p = 0; p < npoints; ++p)
    {
      float* v = cloud.data() + p * stride;
      v[0] = (v[0] - cx) * s;
      v[1] = (v[1] - cy) * s;
      v[2] = (v[2] - cz) * s;
    }
  }
  else // UnitCube
  {
    float mn[3] = {1e30f, 1e30f, 1e30f};
    float mx[3] = {-1e30f, -1e30f, -1e30f};
    for(int64_t p = 0; p < npoints; ++p)
    {
      const float* v = cloud.data() + p * stride;
      for(int c = 0; c < 3; ++c)
      {
        mn[c] = std::min(mn[c], v[c]);
        mx[c] = std::max(mx[c], v[c]);
      }
    }
    const float center[3]
        = {0.5f * (mn[0] + mx[0]), 0.5f * (mn[1] + mx[1]), 0.5f * (mn[2] + mx[2])};
    float edge = 0.f;
    for(int c = 0; c < 3; ++c)
      edge = std::max(edge, mx[c] - mn[c]);
    const float s = (edge > 1e-9f) ? 1.f / edge : 1.f;
    for(int64_t p = 0; p < npoints; ++p)
    {
      float* v = cloud.data() + p * stride;
      for(int c = 0; c < 3; ++c)
        v[c] = (v[c] - center[c]) * s;
    }
  }
}

// Build the ORT input tensor from the packed float buffer, converting to the
// model's declared element type. half_buf must outlive the returned Value.
Ort::Value buildInputTensor(
    const boost::container::vector<float>& input,
    const std::vector<int64_t>& ishape, TensorElemType in_dt,
    std::vector<uint16_t>& half_buf)
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
  return Onnx::vec_to_tensor<float>(
      const_cast<boost::container::vector<float>&>(input), ishape);
}

// Run with every declared input: the packed cloud at cloud_in, the others from
// their aux plan (scalars take Param 1 / 2, the rest zeros). Only the cloud
// used to be bound, so any second input made ORT refuse the run, and a cloud
// declared second went in under input 0's name.
std::vector<Ort::Value> runModel(
    Onnx::OnnxRunContext& ctx, const Onnx::ModelSpec& spec,
    const boost::container::vector<float>& packed, const std::vector<int64_t>& ishape,
    TensorElemType in_dt, int cloud_in, const std::vector<Onnx::AuxPlan>& aux,
    std::span<const float> params, std::vector<uint16_t>& half_buf)
{
  const int nin = (int)spec.inputs.size();
  std::vector<Ort::Value> ins;
  ins.reserve(nin);
  for(int i = 0; i < nin; ++i)
    ins.emplace_back(nullptr);
  ins[cloud_in] = buildInputTensor(packed, ishape, in_dt, half_buf);

  std::vector<std::vector<uint8_t>> aux_store(aux.size());
  std::vector<std::vector<int64_t>> aux_shape(aux.size());
  const Onnx::AuxHost host{.params = params, .primary_shape = ishape};
  for(std::size_t k = 0; k < aux.size(); ++k)
    if(aux[k].index >= 0 && aux[k].index < nin)
      ins[aux[k].index] = Onnx::fillAux(aux[k], host, aux_store[k], aux_shape[k]);

  const int nout = (int)spec.output_names_char.size();
  std::vector<Ort::Value> outs;
  outs.reserve(nout);
  for(int i = 0; i < nout; ++i)
    outs.emplace_back(nullptr);
  ctx.infer(spec, ins, outs);
  return outs;
}

// Decoded result staged in heap buffers (so the decode also runs off the worker
// thread). applyDecoded() then just moves them into the node's ports.
struct DecodedGeom
{
  GeomOutputKind kind = GeomOutputKind::Data;
  std::vector<float> cloud; // interleaved xyz(+f) when kind==PointCloud
  int out_points = 0;
  std::vector<float> data; // labels / seg / values
};

DecodedGeom decodeOutput(
    Ort::Value& res, GeomTaskMode task_override, int in_channels,
    std::vector<float>& scratch)
{
  DecodedGeom d;
  const auto info = res.GetTensorTypeAndShapeInfo();
  const auto oshape = info.GetShape();
  const int64_t ocount = (int64_t)info.GetElementCount();
  const TensorElemType odt = Onnx::fromOrtElementType(info.GetElementType());
  const void* raw = res.GetTensorData<uint8_t>();
  const float* f = Onnx::tensorToFloat(raw, ocount, odt, scratch);

  GeomOutputKind kind = Onnx::classifyGeomOutput(oshape, in_channels);
  if(task_override == GeomTaskMode::PointCloud)
    kind = GeomOutputKind::PointCloud;
  else if(task_override == GeomTaskMode::Data)
    kind = GeomOutputKind::Data;

  if(kind == GeomOutputKind::PointCloud && oshape.size() == 3)
  {
    const PointLayout pl = Onnx::detectPointLayout(oshape);
    const int64_t n = (pl.layout == PointAxisLayout::NPC) ? oshape[1] : oshape[2];
    // A forced PointCloud override on a non-cloud rank-3 output (detectPointLayout
    // returns Unknown) would read n*channels floats from an n-element buffer.
    // Only treat it as a cloud when the layout is valid and the payload covers
    // the read; otherwise fall through to the Data branch.
    if(pl.valid() && n > 0
       && n * static_cast<int64_t>(pl.channels) <= ocount)
    {
      d.kind = GeomOutputKind::PointCloud;
      d.out_points = (int)n;
      // Emit a plain xyz interleaved cloud (drop features on output for now).
      Onnx::unpackPoints(f, n, pl, /*out_stride*/ 3, d.cloud);
      return d;
    }
  }

  d.kind = GeomOutputKind::Data;
  d.data.assign(f, f + ocount);
  return d;
}

void applyDecoded(GeometryProcessor& self, DecodedGeom& d)
{
  switch(d.kind)
  {
    case GeomOutputKind::PointCloud:
      self.outputs.cloud.value = std::move(d.cloud);
      self.outputs.out_count.value = d.out_points;
      break;
    case GeomOutputKind::Data:
    default:
      self.outputs.data.value = std::move(d.data);
      break;
  }
}

// Heavy models run async so a slow inference can't freeze the render thread.
// PointNet++ grouping / completion get expensive with point count.
bool isHeavyModel(std::size_t model_bytes, int64_t npoints)
{
  return model_bytes > 32u * 1024 * 1024 || npoints > 8192;
}
} // namespace

GeometryProcessor::GeometryProcessor() noexcept
{
  inputs.point_stride.value = 3;
  normalized.reserve(2048 * 3);
  packed.reserve(2048 * 3);
}

GeometryProcessor::~GeometryProcessor() = default;

void GeometryProcessor::reloadModel()
{
  ctx = std::make_shared<Onnx::OnnxRunContext>(
      inputs.model.file.bytes, inputs.model.file.filename);
  spec = ctx->readModelSpec();
  arch = Onnx::classifyModel(toArchIO(spec));
  const int ci = findCloudInput(spec);
  in_layout = (ci >= 0) ? Onnx::detectPointLayout(spec.inputs[ci].shape)
                        : PointLayout{};
  cloudIn = std::max(ci, 0);
  const int owned[]{cloudIn};
  aux = Onnx::planAuxInputs(spec.inputs, owned);
  lastModelPath = inputs.model.file.filename;
  lastOutputIndex = inputs.output_index.value;
}

void GeometryProcessor::operator()()
try
{
  if(!available)
    return;
  if(inputs.model.current_model_invalid)
    return;
  if(inputs.model.file.bytes.empty())
    return;

  if(!ctx || lastModelPath != inputs.model.file.filename)
    reloadModel();
  else if(inputs.output_index.value != lastOutputIndex)
    lastOutputIndex = inputs.output_index.value;

  if(spec.inputs.empty() || spec.outputs.empty())
    return;
  if(!in_layout.valid())
    return; // not a point-set model

  runCloud();
}
catch(...)
{
  inputs.model.current_model_invalid = true;
}

void GeometryProcessor::runCloud()
{
  const std::vector<float>& src = inputs.cloud.value;
  if(src.empty())
    return;

  const int stride = std::max(3, inputs.point_stride.value);

  // Resolve the supplied point count: explicit control wins, else derive from
  // the payload length and stride.
  int64_t npoints = inputs.point_count.value;
  if(npoints <= 0)
    npoints = (int64_t)(src.size() / (size_t)stride);
  if(npoints <= 0)
    return;
  // Don't read past the payload.
  npoints = std::min<int64_t>(npoints, (int64_t)(src.size() / (size_t)stride));
  if(npoints <= 0)
    return;

  // Fixed-N model: clamp/pad to the model's declared point count.
  PointLayout pl = in_layout;
  if(!pl.dynamicCount())
    npoints = std::min<int64_t>(npoints, pl.count);

  // --- normalize (copy payload into a reusable buffer, then in-place) --------
  const int64_t flat = npoints * stride;
  normalized.assign(src.begin(), src.begin() + (size_t)flat);
  normalizeCloud(normalized, npoints, stride, inputs.normalization.value);

  // --- pack into the model's layout ------------------------------------------
  const int64_t pn = pl.dynamicCount() ? npoints : pl.count;
  // If the model wants more points than supplied (fixed N), zero-pad the tail.
  if(pn > npoints)
  {
    const int64_t need = pn * stride;
    if((int64_t)normalized.size() < need)
      normalized.resize(need, 0.f);
  }
  Onnx::packPoints(normalized.data(), pn, stride, pl, packed);

  const bool heavy = isHeavyModel(inputs.model.file.bytes.size(), pn);
  dispatchInfer(pn, heavy);
}

// Run inference + decode synchronously, or hand a snapshot to the worker thread
// for heavy models (latest-wins: drop frames while one is in flight).
void GeometryProcessor::dispatchInfer(int64_t npoints, bool force_async)
{
  const TensorElemType in_dt = spec.inputs[cloudIn].elem_type;
  std::vector<int64_t> ishape = Onnx::resolveShape(in_layout, npoints);

  if(force_async)
  {
    if(inferenceInProgress)
      return;
    inferenceInProgress = true;
    // Pooled job: lock-free acquire; recycled vectors keep their capacity so
    // the assignments below don't allocate in steady state.
    auto job = JobPool<GeomInferJob>::instance().acquire();
    job->ctx = ctx;
    job->input = packed;
    job->ishape = std::move(ishape);
    job->in_dt = in_dt;
    job->in_layout = in_layout;
    job->output_index = inputs.output_index.value;
    job->task = inputs.task.value; // honor the output-routing override async too
    job->cloud_in = cloudIn;
    job->aux = aux;
    job->params[0] = inputs.param1.value;
    job->params[1] = inputs.param2.value;
    worker.request(std::move(job));
    return;
  }

  const float params[2]{inputs.param1.value, inputs.param2.value};
  auto outs = runModel(*ctx, spec, packed, ishape, in_dt, cloudIn, aux, params, half_buf);
  const int nout = (int)outs.size();

  const int idx = std::clamp(inputs.output_index.value, 0, nout - 1);
  DecodedGeom d = decodeOutput(outs[idx], inputs.task.value, in_layout.channels, out_scratch);
  applyDecoded(*this, d);
}

std::function<void(GeometryProcessor&)>
GeometryProcessor::worker::work(std::unique_ptr<GeomInferJob> job)
{
  // RAII: whatever path we exit through, the job goes back to the lock-free
  // pool (with its buffer capacities intact) once the results are moved out.
  struct Recycle
  {
    std::unique_ptr<GeomInferJob>& j;
    ~Recycle()
    {
      if(j)
        j->ctx.reset(); // don't keep the ORT session alive from the pool
      JobPool<GeomInferJob>::instance().release(std::move(j));
    }
  } recycle{job};

  if(!job || !job->ctx)
    return [](GeometryProcessor& self) { self.inferenceInProgress = false; };
  try
  {
    const auto& spec = job->ctx->readModelSpec();
    std::vector<uint16_t> half_buf;
    auto outs = runModel(
        *job->ctx, spec, job->input, job->ishape, job->in_dt, job->cloud_in, job->aux,
        job->params, half_buf);
    const int nout = (int)outs.size();

    const int idx = std::clamp(job->output_index, 0, nout - 1);
    std::vector<float> scratch;
    DecodedGeom d = decodeOutput(outs[idx], job->task, job->in_layout.channels, scratch);
    return [d = std::move(d)](GeometryProcessor& self) mutable
    {
      self.inferenceInProgress = false;
      applyDecoded(self, d);
    };
  }
  catch(...)
  {
    return [](GeometryProcessor& self) { self.inferenceInProgress = false; };
  }
}

}
