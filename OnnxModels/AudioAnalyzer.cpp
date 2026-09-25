#include "AudioAnalyzer.hpp"

#include <OnnxModels/JobPool.hpp>

#include <Onnx/helpers/AudioRate.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/TensorToTexture.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace OnnxModels
{
using Onnx::PortArchetype;
using Onnx::TensorElemType;

namespace
{
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

int64_t flatPositive(const std::vector<int64_t>& s)
{
  int64_t p = 1;
  for(auto d : s)
    p *= (d > 0 ? d : 1);
  return p;
}

// CREPE: [B,1024] frames -> [B,360] pitch bins. It was trained on frames
// normalised to zero mean and unit variance: quiet input otherwise lands on one
// bin whatever its pitch.
bool isCrepe(const Onnx::ModelSpec& s)
{
  return s.inputs.size() == 1 && s.outputs.size() == 1 && s.inputs[0].shape.size() == 2
         && s.inputs[0].shape[1] == 1024 && s.outputs[0].shape.size() == 2
         && s.outputs[0].shape[1] == 360;
}

void normalizeFrames(std::vector<float>& x, int channels)
{
  const std::size_t n = channels > 0 ? x.size() / channels : x.size();
  for(int c = 0; c < std::max(channels, 1); ++c)
  {
    float* p = x.data() + c * n;
    double mean = 0, var = 0;
    for(std::size_t i = 0; i < n; ++i)
      mean += p[i];
    mean /= std::max<std::size_t>(n, 1);
    for(std::size_t i = 0; i < n; ++i)
      var += (p[i] - mean) * (p[i] - mean);
    const double sd = std::max(std::sqrt(var / std::max<std::size_t>(n, 1)), 1e-8);
    for(std::size_t i = 0; i < n; ++i)
      p[i] = (float)((p[i] - mean) / sd);
  }
}
} // namespace

AudioAnalyzer::AudioAnalyzer() noexcept = default;

AudioAnalyzer::~AudioAnalyzer() = default;

void AudioAnalyzer::prepare(halp::setup info)
{
  host_rate = info.rate > 0 ? info.rate : 48000.0;
  host_in_channels = info.input_channels;
  max_frames = info.frames > 0 ? (std::size_t)info.frames : 4096;
  // The next tick sees settings that differ from the pipeline's, and builds
  // one for the new host config.
}

static void buildPipeline(AnalyzerPipeline& P);

// Worker side of a build: the session (from the file, since the port's
// mapping may be gone by now; or the running one, for a settings change),
// then everything buildPipeline() derives from it.
static std::shared_ptr<AnalyzerPipeline>
makePipeline(const AnalyzerBuildParams& bp, std::shared_ptr<Onnx::OnnxRunContext> ctx)
{
  auto P = std::make_shared<AnalyzerPipeline>();
  P->params = bp;
  if(!ctx)
  {
    std::ifstream f(bp.path, std::ios::binary);
    if(!f)
      throw std::runtime_error("cannot read the file");
    const std::string bytes{std::istreambuf_iterator<char>(f), {}};
    ctx = std::make_shared<Onnx::OnnxRunContext>(
        bytes, bp.path, Onnx::Options::precise());
  }
  P->ctx = std::move(ctx);
  P->spec = P->ctx->readModelSpec();
  P->arch = Onnx::classifyModel(toArchIO(P->spec));
  if(!P->spec.inputs.empty() && !P->spec.outputs.empty())
    buildPipeline(*P);
  return P;
}

static void buildPipeline(AnalyzerPipeline& P)
{
  const AnalyzerBuildParams& bp = P.params;
  P.wave_in_index = -1;
  P.states.clear();

  for(int i = 0; i < (int)P.arch.inputs.size(); ++i)
  {
    const auto a = P.arch.inputs[i].arch;
    if(a == PortArchetype::Waveform || a == PortArchetype::Spectrogram)
    {
      P.wave_in_index = i;
      break;
    }
  }
  if(P.wave_in_index < 0)
  {
    for(int i = 0; i < (int)P.arch.inputs.size(); ++i)
    {
      const auto a = P.arch.inputs[i].arch;
      if(a != PortArchetype::RecurrentState && a != PortArchetype::Scalar
         && a != PortArchetype::TokenSeq)
      {
        P.wave_in_index = i;
        break;
      }
    }
  }
  if(P.wave_in_index < 0)
    P.wave_in_index = 0;

  for(int i = 0; i < (int)P.arch.inputs.size(); ++i)
  {
    if(P.arch.inputs[i].arch != PortArchetype::RecurrentState)
      continue;
    AnalyzerStatePort sp;
    sp.in_index = i;
    sp.shape = P.spec.inputs[i].shape;
    for(auto& d : sp.shape)
      if(d < 0)
        d = 1;
    // Paired once by classifyModel (names first, then free same-shape outputs).
    sp.out_index = P.arch.inputs[i].state_pair;
    sp.data.assign((std::size_t)flatPositive(sp.shape), 0.f);
    P.states.push_back(std::move(sp));
  }

  std::vector<int> owned{P.wave_in_index};
  for(const auto& s : P.states)
    owned.push_back(s.in_index);
  P.aux = Onnx::planAuxInputs(P.spec.inputs, owned, {.nparams = 1});
  P.aux_store.resize(P.aux.size());
  P.aux_shapes.resize(P.aux.size());

  // WaveformShape has no rank-4 layout: CLAP's mel_fusion [1,4,1001,64] would
  // be fed as [1,1,64] and fail on every block.
  if(P.spec.inputs[P.wave_in_index].shape.size() >= 4)
  {
    P.refusal = "takes a rank-4 input (" + P.spec.inputs[P.wave_in_index].name
                + "), which needs a spectrogram frontend; not supported";
    return;
  }

  P.in_shape
      = Onnx::WaveformShape::fromInputShape(P.spec.inputs[P.wave_in_index].shape);
  P.model_rate = bp.rate_override > 0
                   ? (double)bp.rate_override
                   : Onnx::audioModelRate(*P.ctx, bp.path, 16000.);
  P.normalize_frames = isCrepe(P.spec);

  const int hc = bp.host_channels > 0 ? bp.host_channels : 1;
  const int64_t block = P.in_shape.block > 0 ? P.in_shape.block : 1024;
  P.audio_in.prepare(P.in_shape, bp.host_rate, P.model_rate, block, block, hc,
                     bp.max_frames);
  P.staged.reserve((std::size_t)P.in_shape.channels * block + 16);
}

void AudioAnalyzer::zeroStates()
{
  if(!pipe)
    return;
  auto& P = *pipe;
  for(auto& s : P.states)
    std::fill(s.data.begin(), s.data.end(), 0.f);
  P.audio_in.reset();
}

AnalyzerBuildParams AudioAnalyzer::currentParams() const
{
  return {
      .path = std::string(inputs.model.file.filename),
      .host_rate = host_rate,
      .host_channels = host_in_channels,
      .max_frames = max_frames,
      .rate_override = inputs.model_rate.value};
}

void AudioAnalyzer::requestBuild(const AnalyzerBuildParams& p, bool reuse_session)
{
  building = true;
  requested = p;
  auto job = JobPool<AnalyzerJob>::instance().acquire();
  job->kind = AnalyzerJob::Kind::Build;
  job->build = p;
  job->ctx = reuse_session && pipe ? pipe->ctx : nullptr;
  worker.request(std::move(job));
}

void AudioAnalyzer::install(std::shared_ptr<AnalyzerPipeline> p)
{
  std::swap(pipe, p);
  dispose(std::move(p));
}

void AudioAnalyzer::dispose(std::shared_ptr<AnalyzerPipeline> p)
{
  if(!p)
    return;
  auto job = JobPool<AnalyzerJob>::instance().acquire();
  job->kind = AnalyzerJob::Kind::Dispose;
  job->pipeline = std::move(p);
  worker.request(std::move(job));
}

void AudioAnalyzer::operator()(int frames)
try
{
  if(!available || inputs.model.current_model_invalid
     || inputs.model.file.bytes.empty())
    return;

  // A new file, or new settings: a pipeline is built on the worker; the
  // running one (if any) keeps analysing until it arrives.
  if(!building)
  {
    if(!pipe || pipe->params.path != inputs.model.file.filename)
    {
      if(!pipe || requested.path != inputs.model.file.filename)
        requestBuild(currentParams(), false);
    }
    else if(!pipe->params.sameSettings(currentParams()))
      requestBuild(currentParams(), true);
  }
  if(!pipe || !pipe->refusal.empty() || pipe->spec.inputs.empty()
     || pipe->spec.outputs.empty())
    return;
  auto& P = *pipe;

  if(inputs.reset.value)
  {
    zeroStates();
    inputs.reset.value.reset();
  }

  const int hc = inputs.audio.channels;
  if(hc > 0 && frames > 0)
    P.audio_in.push(inputs.audio.samples, hc, (std::size_t)frames);

  // A block that fails is reported and skipped; while the same failure
  // repeats, the blocks are dropped instead of run (see FailureLog).
  int guard = 0;
  if(!failures.ready())
  {
    while(P.audio_in.ready() && guard++ < 32)
      P.audio_in.fill(P.staged);
  }
  while(P.audio_in.ready() && guard++ < 32)
  {
    try
    {
      runBlock();
    }
    catch(const std::exception& e)
    {
      failures.failed(name(), inputs.model.file.filename, e.what());
      break;
    }
  }
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

std::function<void(AudioAnalyzer&)>
AudioAnalyzer::worker::work(std::unique_ptr<AnalyzerJob> job)
{
  // Back to the lock-free pool, emptied, whatever path we exit through.
  struct Recycle
  {
    std::unique_ptr<AnalyzerJob>& j;
    ~Recycle()
    {
      if(j)
      {
        j->ctx.reset();
        j->pipeline.reset();
      }
      JobPool<AnalyzerJob>::instance().release(std::move(j));
    }
  } recycle{job};
  if(!job)
    return {};
  if(job->kind == AnalyzerJob::Kind::Dispose)
  {
    job->pipeline.reset(); // the session and the buffers are freed here
    return {};
  }
  try
  {
    auto p = makePipeline(job->build, std::move(job->ctx));
    return [p = std::move(p)](AudioAnalyzer& self) mutable
    {
      self.building = false;
      if(p->params.path != self.inputs.model.file.filename)
      {
        self.dispose(std::move(p)); // another file was picked meanwhile
        return;
      }
      if(!p->refusal.empty())
      {
        self.failures.failed(AudioAnalyzer::name(), p->params.path, p->refusal);
        self.inputs.model.current_model_invalid = true;
        self.dispose(std::move(p));
        return;
      }
      self.failures.succeeded();
      self.install(std::move(p));
    };
  }
  catch(const std::exception& e)
  {
    return [what = std::string(e.what()), path = job->build.path](AudioAnalyzer& self)
    {
      self.building = false;
      if(path != self.inputs.model.file.filename)
        return;
      self.failures.failed(AudioAnalyzer::name(), path, "cannot load the model: " + what);
      self.inputs.model.current_model_invalid = true;
    };
  }
}

void AudioAnalyzer::runBlock()
{
  auto& P = *pipe;
  const int64_t n = P.audio_in.fill(P.staged);
  if(n <= 0)
    return;
  if(P.normalize_frames)
    normalizeFrames(P.staged, P.in_shape.channels);

  // On the audio thread: the pipeline's scratch (ins, outs, ishape, flat) is
  // reused block after block.
  const int nin = (int)P.spec.inputs.size();
  auto& ins = P.ins;
  ins.clear();
  P.in_shape.tensorShapeInto(n, P.ishape);
  const auto& ishape = P.ishape;
  const float param = inputs.param.value;
  const Onnx::AuxHost host{
      .params = {&param, 1}, .sample_rate = P.model_rate, .primary_shape = ishape};
  for(int i = 0; i < nin; ++i)
  {
    if(i == P.wave_in_index)
    {
      ins.emplace_back(Onnx::vec_to_tensor<float>(P.staged, ishape));
      continue;
    }
    AnalyzerStatePort* sp = nullptr;
    for(auto& s : P.states)
      if(s.in_index == i)
      {
        sp = &s;
        break;
      }
    if(sp)
    {
      ins.emplace_back(Onnx::vec_to_tensor<float>(sp->data, sp->shape));
    }
    else
    {
      // The other inputs, by their aux plan: Silero's sr gets the model rate
      // (as its int64), a bool flag false, a scalar the Param.
      std::size_t k = 0;
      while(k < P.aux.size() && P.aux[k].index != i)
        ++k;
      if(k < P.aux.size())
        ins.emplace_back(Onnx::fillAux(P.aux[k], host, P.aux_store[k], P.aux_shapes[k]));
      else
        ins.emplace_back(nullptr);
    }
  }

  const int nout = (int)P.spec.output_names_char.size();
  auto& outs = P.outs;
  outs.clear();
  for(int i = 0; i < nout; ++i)
    outs.emplace_back(nullptr);
  P.ctx->infer(P.spec, ins, outs);
  failures.succeeded();

  // Choose the primary (non-state) output: the first output that is not paired
  // with a recurrent state input.
  int primary = 0;
  for(int o = 0; o < nout; ++o)
  {
    bool is_state = false;
    for(auto& s : P.states)
      if(s.out_index == o)
      {
        is_state = true;
        break;
      }
    if(!is_state)
    {
      primary = o;
      break;
    }
  }

  // Flatten the primary output to Data.
  auto& res = outs[primary];
  const auto info = res.GetTensorTypeAndShapeInfo();
  const int64_t cnt = (int64_t)info.GetElementCount();
  const TensorElemType odt = Onnx::fromOrtElementType(info.GetElementType());
  const float* f
      = Onnx::toFloat(res.GetTensorData<uint8_t>(), cnt, odt, P.out_scratch);
  P.flat.assign(f, f + cnt);

  // Feed recurrent state back.
  for(auto& s : P.states)
  {
    if(s.out_index < 0 || s.out_index >= nout)
      continue;
    auto& sres = outs[s.out_index];
    const auto sinfo = sres.GetTensorTypeAndShapeInfo();
    const int64_t scnt = (int64_t)sinfo.GetElementCount();
    const TensorElemType sdt
        = Onnx::fromOrtElementType(sinfo.GetElementType());
    const float* sf
        = Onnx::toFloat(sres.GetTensorData<uint8_t>(), scnt, sdt, P.out_scratch);
    s.data.assign(sf, sf + scnt);
  }

  reduceOutputs(P.flat, primary);
}

void AudioAnalyzer::reduceOutputs(const std::vector<float>& flat, int primary)
{
  outputs.data.value.assign(flat.begin(), flat.end());
  if(flat.empty())
    return;

  AnalyzerReduce mode = inputs.reduce.value;
  if(mode == AnalyzerReduce::Auto)
  {
    // A long vector (>= 64) looks like a pitch/class distribution -> argmax;
    // a tiny vector (1..2) is a probability/scalar pair -> first values.
    mode = (flat.size() >= 64) ? AnalyzerReduce::Argmax
                               : AnalyzerReduce::FirstTwo;
  }

  switch(mode)
  {
    case AnalyzerReduce::Argmax:
    {
      std::size_t am = 0;
      float mv = flat[0];
      for(std::size_t i = 1; i < flat.size(); ++i)
        if(flat[i] > mv)
        {
          mv = flat[i];
          am = i;
        }
      // value1 = normalized bin position (0..1), value2 = peak confidence.
      outputs.value1.value
          = (flat.size() > 1) ? (float)am / (float)(flat.size() - 1) : 0.f;
      outputs.value2.value = mv;
      break;
    }
    case AnalyzerReduce::MeanPeak:
    {
      double sum = 0.0;
      float pk = flat[0];
      for(float v : flat)
      {
        sum += v;
        pk = std::max(pk, v);
      }
      outputs.value1.value = (float)(sum / (double)flat.size());
      outputs.value2.value = pk;
      break;
    }
    case AnalyzerReduce::FirstTwo:
    case AnalyzerReduce::Auto:
    default:
      outputs.value1.value = flat[0];
      outputs.value2.value = flat.size() > 1 ? flat[1] : flat[0];
      break;
  }
  (void)primary;
}

}
