#include "ImageGenerator.hpp"

#include <OnnxModels/JobPool.hpp>

#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>

namespace OnnxModels
{
using Onnx::TensorElemType;

namespace
{
// float32 -> IEEE half (binary16). Same rounding as ImageProcessor; only used for
// the rare fp16-input generator. (Mirrors ImageProcessor::floatToHalf so the two
// nodes stay consistent; kept local since it is a tiny leaf helper.)
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

Onnx::WriteMode resolveWriteMode(GenOutputMode m)
{
  switch(m)
  {
    case GenOutputMode::MinMaxNormalize: return Onnx::WriteMode::MinMaxNormalize;
    case GenOutputMode::Denormalize:     return Onnx::WriteMode::Denormalize;
    case GenOutputMode::Half255:         return Onnx::WriteMode::Half255;
    case GenOutputMode::DirectClamp:     return Onnx::WriteMode::DirectClamp;
    case GenOutputMode::Auto:
    default:
      // Resolved from the first output's range (pickAutoMode).
      return Onnx::WriteMode::MinMaxNormalize;
  }
}

// Auto pixel mapping from an output's value range: most generators end in a
// tanh ([-1,1]); a per-frame min/max stretch there shifted the brightness and
// pumped it as the latent moved.
Onnx::WriteMode pickAutoMode(float lo, float hi)
{
  // StyleGAN-type outputs overshoot [-1,1] a little (1.6 is common).
  if(lo < 0.f && lo >= -2.f && hi <= 2.f)
    return Onnx::WriteMode::Denormalize;
  if(lo >= 0.f && hi <= 1.05f)
    return Onnx::WriteMode::DirectClamp;
  if(lo >= 0.f && hi > 2.f && hi <= 300.f)
    return Onnx::WriteMode::Passthrough;
  return Onnx::WriteMode::MinMaxNormalize;
}

// Product of positive dims (treat dynamic <=0 as 1). Shared latent-size logic.
int64_t flatNonBatch(const std::vector<int64_t>& s)
{
  if(s.empty())
    return 0;
  int64_t p = 1;
  // Skip the leading batch dim when rank >= 2.
  const size_t start = (s.size() >= 2) ? 1u : 0u;
  for(size_t i = start; i < s.size(); ++i)
    if(s[i] > 0)
      p *= s[i];
  return p;
}

// Build an ORT input tensor from a float buffer, converting to the model's
// declared element type. The half staging buffer must outlive the returned Value.
Ort::Value buildInputTensor(
    boost::container::vector<float>& input, const std::vector<int64_t>& ishape,
    TensorElemType in_dt, std::vector<uint16_t>& half_buf)
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
  return Onnx::vec_to_tensor<float>(input, ishape);
}

// Resolve the [batch=1, ...] input shape for a model's first input, given the
// flat size we are feeding it. Uses the declared shape with batch forced to 1
// when concrete; falls back to a flat [1, dim].
std::vector<int64_t> resolveInShape(const Onnx::ModelSpec& spec, int64_t dim)
{
  std::vector<int64_t> ishape = spec.inputs[0].shape;
  if(ishape.empty())
    return {1, dim};
  if(ishape[0] <= 0)
    ishape[0] = 1;
  // If every non-batch dim is concrete and their product matches dim, keep the
  // declared shape (e.g. [1,512] or [1,16,512]); else flatten to [1, dim].
  if(flatNonBatch(ishape) == dim)
    return ishape;
  return {1, dim};
}

// Decoded result staged in a heap buffer so the per-pixel decode runs off the
// worker thread; applyDecoded() then only creates the texture + memcpy.
struct DecodedOutput
{
  enum Target
  {
    None,
    Image,
    Data
  } target = None;
  int w = 0, h = 0;
  Onnx::WriteMode mode{}; // the mapping used, for Auto to keep it
  std::vector<uint8_t> rgba; // Image: w*h*4
  std::vector<float> data;   // Data: flattened
};

DecodedOutput decodeOutput(
    Ort::Value& res, Onnx::WriteMode wm, bool pick_auto, std::vector<float>& scratch)
{
  DecodedOutput d;
  const auto info = res.GetTensorTypeAndShapeInfo();
  const auto oshape = info.GetShape();
  const int64_t ocount = (int64_t)info.GetElementCount();
  const TensorElemType odt = Onnx::fromOrtElementType(info.GetElementType());
  const void* raw = res.GetTensorData<uint8_t>();
  const Onnx::OutSpec os = Onnx::makeOutSpec(oshape);
  const float* f = Onnx::toFloat(raw, ocount, odt, scratch);

  if(!os.spatial)
  {
    d.target = DecodedOutput::Data;
    d.data.assign(f, f + ocount);
    return d;
  }
  d.target = DecodedOutput::Image;
  d.w = os.w;
  d.h = os.h;
  if(pick_auto && ocount > 0)
  {
    const auto [lo, hi] = std::minmax_element(f, f + ocount);
    wm = pickAutoMode(*lo, *hi);
  }
  d.mode = wm;
  d.rgba.resize((size_t)os.w * os.h * 4);
  Onnx::writeRgb(f, os, wm, d.rgba.data());
  return d;
}

void applyDecoded(ImageGenerator& self, DecodedOutput& d)
{
  switch(d.target)
  {
    case DecodedOutput::Image:
      self.outputs.image.create(d.w, d.h);
      std::memcpy(self.outputs.image.texture.bytes, d.rgba.data(), d.rgba.size());
      self.outputs.image.texture.changed = true;
      break;
    case DecodedOutput::Data:
      self.outputs.data.value = std::move(d.data);
      break;
    default:
      break;
  }
}

// Run one ORT session: build input 0 from `in` and the others from their aux
// plan (extra latents get seeded noise, scalars Param 1 / 2), infer, return
// output 0. Only input 0 used to be fed, so ORT refused multi-input generators
// such as EigenGAN (eps + six z_*). Heavy helper shared by the chain (mapping
// then synthesis) inside the worker thread.
Ort::Value runStage(
    Onnx::OnnxRunContext& ctx, const Onnx::ModelSpec& spec,
    boost::container::vector<float>& in, const std::vector<int64_t>& ishape,
    TensorElemType in_dt, const std::vector<Onnx::AuxPlan>& aux,
    const Onnx::AuxHost& host)
{
  std::vector<uint16_t> half_buf;
  const int nin = std::max<int>(1, (int)spec.input_names_char.size());
  std::vector<Ort::Value> ins;
  ins.reserve(nin);
  for(int i = 0; i < nin; ++i)
    ins.emplace_back(nullptr);
  ins[0] = buildInputTensor(in, ishape, in_dt, half_buf);

  std::vector<std::vector<uint8_t>> aux_store(aux.size());
  std::vector<std::vector<int64_t>> aux_shape(aux.size());
  for(std::size_t k = 0; k < aux.size(); ++k)
    if(aux[k].index > 0 && aux[k].index < nin)
      ins[aux[k].index] = Onnx::fillAux(aux[k], host, aux_store[k], aux_shape[k]);

  const int nout = (int)spec.output_names_char.size();
  std::vector<Ort::Value> outs;
  outs.reserve(nout);
  for(int i = 0; i < nout; ++i)
    outs.emplace_back(nullptr);
  ctx.infer(spec, ins, outs);
  return std::move(outs[0]);
}
} // namespace

ImageGenerator::ImageGenerator() noexcept = default;
ImageGenerator::~ImageGenerator() = default;

void ImageGenerator::buildLatent(
    std::vector<float>& z, int dim, uint32_t seed, const float* user, int user_n,
    float scale)
{
  if(dim <= 0)
  {
    z.clear();
    return;
  }
  z.resize(dim);
  std::mt19937 gen(seed);
  // GAN z-latents are trained against a standard normal N(0,1). The args are
  // (mean, stddev) — NOT a [lo,hi] range — so a (-1,1) here would bias every
  // latent a full sigma off-manifold.
  std::normal_distribution<float> dist(0.f, 1.f);
  for(float& v : z)
    v = dist(gen);
  const int nuser = std::min(dim, user_n);
  for(int i = 0; i < nuser; ++i)
    z[i] = user[i];
  for(float& v : z)
    v *= scale;
}

bool ImageGenerator::chainCompatible(
    const std::vector<int64_t>& map_out, const std::vector<int64_t>& synth_in)
{
  const int64_t w = flatNonBatch(map_out);
  if(w <= 0)
    return false;
  // Equal sizes, or a w+ synthesis [1,K,D] taking the mapping's w [1,D] K
  // times (StyleGAN / e4e: 18 x 512).
  return flatNonBatch(synth_in) == w
         || (synth_in.size() == 3 && synth_in[1] > 1 && synth_in[2] == w);
}

void ImageGenerator::reloadModel()
{
  ctx = std::make_shared<Onnx::OnnxRunContext>(
      inputs.model.file.bytes, inputs.model.file.filename);
  spec = ctx->readModelSpec();
  lastModelPath = inputs.model.file.filename;

  // Optional mapping net.
  if(!inputs.mapping_model.file.filename.empty()
     && !inputs.mapping_model.file.bytes.empty())
  {
    map_ctx
        = std::make_shared<Onnx::OnnxRunContext>(
            inputs.mapping_model.file.bytes, inputs.mapping_model.file.filename);
    map_spec = map_ctx->readModelSpec();
  }
  else
  {
    map_ctx.reset();
    map_spec = {};
  }
  lastMappingPath = inputs.mapping_model.file.filename;

  // The latent size we build z from is the FIRST stage's first-input flat size:
  // the mapping net when present, else the synthesis net.
  const Onnx::ModelSpec& first = map_ctx ? map_spec : spec;
  latent_dim
      = first.inputs.empty() ? 0 : (int)flatNonBatch(first.inputs[0].shape);

  const int owned[]{0};
  synth_aux = Onnx::planAuxInputs(spec.inputs, owned, {.seeded_noise = true});
  map_aux = map_ctx ? Onnx::planAuxInputs(map_spec.inputs, owned, {.seeded_noise = true})
                    : std::vector<Onnx::AuxPlan>{};

  // Invalidate the cached latent so the next frame regenerates.
  produced = false;
  autoMode.reset();
}

void ImageGenerator::operator()()
try
{
  if(!available)
    return;
  if(inputs.model.current_model_invalid)
    return;
  if(inputs.model.file.bytes.empty())
    return;

  // (Re)load on either model port changing.
  if(!ctx || lastModelPath != inputs.model.file.filename
     || lastMappingPath != inputs.mapping_model.file.filename)
    reloadModel();

  if(spec.inputs.empty() || spec.outputs.empty())
    return;
  if(latent_dim <= 0)
    return;

  runGenerate();
}
catch(...)
{
  inputs.model.current_model_invalid = true;
}

void ImageGenerator::runGenerate()
{
  if(inferenceInProgress)
    return; // a generation is in flight; latest-wins, drop this frame

  // Build z (seed + user overlay + scale). param1/param2 overlay the tail of the
  // latent generically (truncation psi / class-label style knobs) when there is
  // room; harmless otherwise.
  buildLatent(
      z_vector, latent_dim, (uint32_t)inputs.seed.value,
      inputs.latent.value.data(), (int)inputs.latent.value.size(),
      inputs.scale.value);
  if(latent_dim >= 2)
  {
    z_vector[latent_dim - 2] += inputs.param1.value;
    z_vector[latent_dim - 1] += inputs.param2.value;
  }
  else if(latent_dim >= 1)
  {
    z_vector[latent_dim - 1] += inputs.param1.value;
  }

  // Skip re-generation when nothing changed.
  if(produced && z_vector == last_z)
    return;

  // If a mapping net is loaded, verify the generic chain is dimension-compatible
  // BEFORE dispatching: mapping out 0 flat == synthesis in 0 flat. If not, we
  // cannot chain (no FBAnime-style reshape), so skip.
  if(map_ctx)
  {
    if(map_spec.outputs.empty() || spec.inputs.empty())
      return;
    if(!chainCompatible(map_spec.outputs[0].shape, spec.inputs[0].shape))
      return;
  }

  // First-stage input shape (mapping net if present, else synthesis net).
  const Onnx::ModelSpec& first = map_ctx ? map_spec : spec;
  std::vector<int64_t> z_shape = resolveInShape(first, latent_dim);

  last_z = z_vector;
  produced = true;
  inferenceInProgress = true;

  // Pooled job: lock-free acquire; recycled vectors keep their capacity so
  // the assignments below don't allocate in steady state.
  auto job = JobPool<GenJob>::instance().acquire();
  job->synth_ctx = ctx;
  job->map_ctx = map_ctx;
  job->z.assign(z_vector.begin(), z_vector.end());
  job->z_shape = std::move(z_shape);
  job->synth_in_dt = spec.inputs[0].elem_type;
  job->map_in_dt
      = map_ctx ? map_spec.inputs[0].elem_type : TensorElemType::Float;
  job->output_index = 0;
  job->wm = autoMode.value_or(resolveWriteMode(inputs.output_mode.value));
  job->pick_auto = inputs.output_mode.value == GenOutputMode::Auto && !autoMode;
  job->synth_aux = synth_aux;
  job->map_aux = map_aux;
  job->seed = (uint32_t)inputs.seed.value;
  job->scale = inputs.scale.value;
  job->params[0] = inputs.param1.value;
  job->params[1] = inputs.param2.value;
  worker.request(std::move(job));
}

std::function<void(ImageGenerator&)>
ImageGenerator::worker::work(std::unique_ptr<GenJob> job)
{
  // RAII: whatever path we exit through, the job goes back to the lock-free
  // pool (with its buffer capacities intact) once the results are moved out.
  struct Recycle
  {
    std::unique_ptr<GenJob>& j;
    ~Recycle()
    {
      if(j)
      {
        // don't keep the ORT sessions alive from the pool
        j->synth_ctx.reset();
        j->map_ctx.reset();
      }
      JobPool<GenJob>::instance().release(std::move(j));
    }
  } recycle{job};

  if(!job || !job->synth_ctx)
    return [](ImageGenerator& self) { self.inferenceInProgress = false; };
  try
  {
    const auto& synth_spec = job->synth_ctx->readModelSpec();
    const Onnx::AuxHost host{
        .params = job->params, .seed = job->seed, .noise_scale = job->scale};

    // Stage 1 (optional): z -> mapping -> w. Take output 0 as w and reshape it to
    // the synthesis input's declared shape (flat sizes were verified compatible).
    boost::container::vector<float> synth_in;
    std::vector<int64_t> synth_in_shape;
    if(job->map_ctx)
    {
      const auto& map_spec = job->map_ctx->readModelSpec();
      Ort::Value w = runStage(
          *job->map_ctx, map_spec, job->z, job->z_shape, job->map_in_dt,
          job->map_aux, host);

      const auto wi = w.GetTensorTypeAndShapeInfo();
      const int64_t wcount = (int64_t)wi.GetElementCount();
      std::vector<float> wscratch;
      const float* wf = Onnx::toFloat(
          w.GetTensorData<uint8_t>(), wcount,
          Onnx::fromOrtElementType(wi.GetElementType()), wscratch);
      synth_in.assign(wf, wf + wcount);

      // Feed w to the synthesis net using ITS declared input 0 shape (batch->1).
      // If the declared shape has dynamic dims whose concrete product no longer
      // matches the actual w element count, fall back to a flat [1, wcount].
      synth_in_shape = synth_spec.inputs.empty()
                           ? std::vector<int64_t>{1, wcount}
                           : synth_spec.inputs[0].shape;
      if(!synth_in_shape.empty() && synth_in_shape[0] <= 0)
        synth_in_shape[0] = 1;
      // w+ synthesis [1,K,D]: the same w for each of the K layers.
      if(synth_in_shape.size() == 3 && synth_in_shape[1] > 1
         && synth_in_shape[2] == wcount)
      {
        synth_in.resize((std::size_t)(synth_in_shape[1] * wcount));
        for(int64_t k = 1; k < synth_in_shape[1]; ++k)
          std::copy_n(synth_in.begin(), wcount, synth_in.begin() + k * wcount);
      }
      else if(flatNonBatch(synth_in_shape) != wcount
         && !(synth_in_shape.size() == 2 && synth_in_shape[1] == wcount))
        synth_in_shape = {1, wcount};
    }
    else
    {
      synth_in = job->z;
      synth_in_shape = job->z_shape;
    }

    // Stage 2: w/z -> synthesis -> image tensor.
    Ort::Value img = runStage(
        *job->synth_ctx, synth_spec, synth_in, synth_in_shape, job->synth_in_dt,
        job->synth_aux, host);

    std::vector<float> scratch;
    DecodedOutput d = decodeOutput(img, job->wm, job->pick_auto, scratch);
    return [d = std::move(d), picked = job->pick_auto](ImageGenerator& self) mutable
    {
      self.inferenceInProgress = false;
      if(picked)
        self.autoMode = d.mode; // keep it: no flicker between mappings
      applyDecoded(self, d);
    };
  }
  catch(...)
  {
    return [](ImageGenerator& self) { self.inferenceInProgress = false; };
  }
}

}
