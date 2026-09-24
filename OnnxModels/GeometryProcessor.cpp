#include "GeometryProcessor.hpp"

#include <OnnxModels/JobPool.hpp>

#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/TensorToTexture.hpp>
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

  if(mode == PointNormalization::UnitSphere)
  {
    // Summed in double: a million float additions would drift the centroid.
    double sx = 0, sy = 0, sz = 0;
    for(int64_t p = 0; p < npoints; ++p)
    {
      const float* v = cloud.data() + p * stride;
      sx += v[0];
      sy += v[1];
      sz += v[2];
    }
    const float cx = float(sx / npoints), cy = float(sy / npoints),
                cz = float(sz / npoints);
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

// Run with every declared input: the packed cloud at cloud_in, the others from
// their aux plan (scalars take Param 1 / 2, the rest zeros).
std::vector<Ort::Value> runModel(
    Onnx::OnnxRunContext& ctx, const Onnx::ModelSpec& spec,
    const boost::container::vector<float>& packed, const std::vector<int64_t>& ishape,
    TensorElemType in_dt, int cloud_in, const std::vector<Onnx::AuxPlan>& aux,
    std::span<const float> params, std::vector<uint8_t>& staging)
{
  const int nin = (int)spec.inputs.size();
  std::vector<Ort::Value> ins;
  ins.reserve(nin);
  for(int i = 0; i < nin; ++i)
    ins.emplace_back(nullptr);
  ins[cloud_in] = Onnx::typedTensor({packed.data(), packed.size()}, ishape, in_dt, staging);

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
  const float* f = Onnx::toFloat(raw, ocount, odt, scratch);

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
  self.failures.succeeded();
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
bool isHeavyCloud(std::size_t model_bytes, int64_t npoints)
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

namespace
{
// Worker side of a model change: the session, from the file (the port's
// mapping may be gone by now), and the routing of its inputs.
std::shared_ptr<const GeomModel> makeGeomModel(const std::string& path)
{
  auto m = std::make_shared<GeomModel>();
  m->path = path;
  m->ctx = Onnx::loadRunContext(path);
  m->spec = m->ctx->readModelSpec();
  const int ci = findCloudInput(m->spec);
  m->in_layout = (ci >= 0) ? Onnx::detectPointLayout(m->spec.inputs[ci].shape)
                           : PointLayout{};
  m->cloud_in = std::max(ci, 0);
  const int owned[]{m->cloud_in};
  m->aux = Onnx::planAuxInputs(m->spec.inputs, owned);
  return m;
}
}

void GeometryProcessor::requestBuild()
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

  // A new file: its model is built on the worker.
  if(!building && (!model || model->path != inputs.model.file.filename)
     && requested != inputs.model.file.filename)
    requestBuild();
  if(!model || model->spec.inputs.empty() || model->spec.outputs.empty())
    return;
  if(!failures.ready())
    return; // the last frames failed the same way: backing off
  if(!model->in_layout.valid())
    return; // not a point-set model

  runCloud();
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
  const PointLayout pl = model->in_layout;
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

  const bool heavy = isHeavyCloud(inputs.model.file.bytes.size(), pn);
  dispatchInfer(pn, heavy);
}

// Run inference + decode synchronously, or hand a snapshot to the worker thread
// for heavy models (latest-wins: drop frames while one is in flight).
void GeometryProcessor::dispatchInfer(int64_t npoints, bool force_async)
{
  const auto& m = *model;
  std::vector<int64_t> ishape = Onnx::resolveShape(m.in_layout, npoints);

  if(force_async)
  {
    if(inferenceInProgress)
      return;
    inferenceInProgress = true;
    // Pooled job: lock-free acquire; recycled vectors keep their capacity so
    // the assignments below don't allocate in steady state.
    auto job = JobPool<GeomInferJob>::instance().acquire();
    job->kind = GeomInferJob::Kind::Infer;
    job->model = model;
    // `packed` is rewritten next frame: it takes the job's buffer back.
    std::swap(job->input, packed);
    job->ishape = std::move(ishape);
    job->output_index = inputs.output_index.value;
    job->task = inputs.task.value; // honor the output-routing override async too
    job->params[0] = inputs.param1.value;
    job->params[1] = inputs.param2.value;
    worker.request(std::move(job));
    return;
  }

  const float params[2]{inputs.param1.value, inputs.param2.value};
  auto outs = runModel(
      *m.ctx, m.spec, packed, ishape, m.spec.inputs[m.cloud_in].elem_type, m.cloud_in,
      m.aux, params, staging);
  const int nout = (int)outs.size();

  const int idx = std::clamp(inputs.output_index.value, 0, nout - 1);
  DecodedGeom d
      = decodeOutput(outs[idx], inputs.task.value, m.in_layout.channels, out_scratch);
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
        j->model.reset(); // don't keep the ORT session alive from the pool
      JobPool<GeomInferJob>::instance().release(std::move(j));
    }
  } recycle{job};

  if(!job)
    return {};
  switch(job->kind)
  {
    case GeomInferJob::Kind::Dispose:
      job->model.reset(); // the session is freed here
      return {};
    case GeomInferJob::Kind::Build:
      try
      {
        auto m = makeGeomModel(job->build_path);
        return [m = std::move(m)](GeometryProcessor& self) mutable {
          self.building = false;
          if(m->path != self.inputs.model.file.filename)
          {
            self.requested.clear(); // another file was picked meanwhile
            self.dispose(std::move(m));
            return;
          }
          self.failures.succeeded();
          self.install(std::move(m));
        };
      }
      catch(const std::exception& e)
      {
        return [what = std::string(e.what()),
                path = job->build_path](GeometryProcessor& self) {
          self.building = false;
          if(path != self.inputs.model.file.filename)
            return;
          self.failures.failed(
              GeometryProcessor::name(), path, "cannot load the model: " + what);
          self.inputs.model.current_model_invalid = true;
          self.requested.clear(); // picking the file again retries
        };
      }
    case GeomInferJob::Kind::Infer:
      break;
  }

  if(!job->model)
    return [](GeometryProcessor& self) { self.inferenceInProgress = false; };
  try
  {
    const auto& m = *job->model;
    std::vector<uint8_t> staging;
    auto outs = runModel(
        *m.ctx, m.spec, job->input, job->ishape, m.spec.inputs[m.cloud_in].elem_type,
        m.cloud_in, m.aux, job->params, staging);
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
  catch(const std::exception& e)
  {
    return [what = std::string(e.what())](GeometryProcessor& self)
    {
      self.inferenceInProgress = false;
      self.failures.failed(GeometryProcessor::name(), self.inputs.model.file.filename, what);
    };
  }
  catch(...)
  {
    return [](GeometryProcessor& self)
    {
      self.inferenceInProgress = false;
      self.failures.failed(GeometryProcessor::name(), self.inputs.model.file.filename, "unknown error");
    };
  }
}

}
