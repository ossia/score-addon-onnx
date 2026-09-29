#include "SequenceProcessor.hpp"

#include <OnnxModels/JobPool.hpp>

#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/TensorToTexture.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>
#include <stdexcept>

namespace OnnxModels
{
using Onnx::ArchIO;
using Onnx::NodeKind;
using Onnx::PortArchetype;
using Onnx::TensorElemType;

namespace
{
void applyResult(SequenceProcessor& self, SeqResult& r);

// Normalises each frame (the last dim) of the input in place.
void normalizeFrames(
    std::span<float> v, const std::vector<int64_t>& shape, SeqNormalize mode)
{
  if(mode == SeqNormalize::None || v.empty())
    return;
  const std::size_t f
      = (!shape.empty() && shape.back() > 0) ? (std::size_t)shape.back() : v.size();
  for(std::size_t start = 0; start + f <= v.size(); start += f)
  {
    auto frame = v.subspan(start, f);
    if(mode == SeqNormalize::L2)
    {
      double sq = 0.;
      for(float x : frame)
        sq += (double)x * x;
      if(sq > 0.)
      {
        const float inv = (float)(1. / std::sqrt(sq));
        for(float& x : frame)
          x *= inv;
      }
    }
    else
    {
      double sum = 0., sq = 0.;
      for(float x : frame)
      {
        sum += x;
        sq += (double)x * x;
      }
      const double mean = sum / (double)f;
      const double var = std::max(0., sq / (double)f - mean * mean);
      const double sd = std::sqrt(var);
      const float inv = sd > 1e-12 ? (float)(1. / sd) : 1.f;
      for(float& x : frame)
        x = (float)((x - mean) * inv);
    }
  }
}
} // namespace

SequenceProcessor::SequenceProcessor() noexcept = default;

SequenceProcessor::~SequenceProcessor() = default;

namespace
{
// Run a full multi-IO inference: place the primary input + all recurrent-state
// inputs at their model-declared indices, run every declared output, decode the
// primary/data outputs and capture the new state values into `r`. `sc` and
// `r` keep their buffers from run to run.
void runInference(
    Onnx::OnnxRunContext& ctx, const Onnx::ModelSpec& spec,
    std::vector<float>& input, const std::vector<int64_t>& ishape,
    TensorElemType in_dt, int primary_in, int primary_out, int data_out,
    std::vector<SeqState>& states, const std::vector<Onnx::AuxPlan>& aux,
    std::span<const float> params, int64_t batch, SeqScratch& sc, SeqResult& r)
{
  const int nin = (int)spec.input_names_char.size();
  const int nout = (int)spec.output_names_char.size();

  auto& ins = sc.ins;
  ins.clear();
  for(int i = 0; i < nin; ++i)
    ins.emplace_back(nullptr);

  ins[primary_in] = Onnx::typedTensor(input, ishape, in_dt, sc.primary);

  sc.state_store.resize(states.size());
  for(size_t k = 0; k < states.size(); ++k)
  {
    auto& st = states[k];
    if(st.in_index < 0 || st.in_index >= nin)
      continue;
    ins[st.in_index] = Onnx::typedTensor(st.values, st.shape, st.dt, sc.state_store[k]);
  }

  // The other inputs, by the aux plan: sr gets the rate, scalars Param 1 / 2,
  // the rest zeros. Each keeps its own storage while the tensors are alive.
  sc.aux_store.resize(aux.size());
  sc.aux_shape.resize(aux.size());
  Onnx::AuxHost host{.params = params, .primary_shape = ishape, .batch = batch};
  for(std::size_t k = 0; k < aux.size(); ++k)
    if(aux[k].index >= 0 && aux[k].index < nin && !ins[aux[k].index])
      ins[aux[k].index] = Onnx::fillAux(aux[k], host, sc.aux_store[k], sc.aux_shape[k]);

  // Anything still unfilled (a state the pairing released) gets zeros so ORT
  // has a value for every declared name.
  sc.filler.resize(nin);
  sc.filler_store.resize(nin);
  sc.filler_shape.resize(nin);
  for(int i = 0; i < nin; ++i)
  {
    if(ins[i])
      continue;
    auto& shp = sc.filler_shape[i];
    shp.assign(spec.inputs[i].shape.begin(), spec.inputs[i].shape.end());
    for(auto& d : shp)
      if(d <= 0)
        d = 1;
    sc.filler[i].assign((size_t)Onnx::flatSize(shp), 0.f);
    ins[i] = Onnx::typedTensor(
        sc.filler[i], shp, spec.inputs[i].elem_type, sc.filler_store[i]);
  }

  auto& outs = sc.outs;
  outs.clear();
  for(int i = 0; i < nout; ++i)
    outs.emplace_back(nullptr);

  ctx.infer(spec, ins, outs);

  auto decode = [&](int idx, std::vector<float>& dst)
  {
    const auto info = outs[idx].GetTensorTypeAndShapeInfo();
    const int64_t cnt = (int64_t)info.GetElementCount();
    const TensorElemType odt
        = Onnx::fromOrtElementType(info.GetElementType());
    const void* raw = outs[idx].GetTensorData<uint8_t>();
    const float* f = Onnx::toFloat(raw, cnt, odt, sc.decode);
    // A replicated batch: every slice is the same, keep the first.
    sc.oshape.resize(info.GetDimensionsCount());
    Ort::ThrowOnError(
        Ort::GetApi().GetDimensions(info, sc.oshape.data(), sc.oshape.size()));
    const int64_t n
        = (batch > 1 && sc.oshape.size() >= 2 && sc.oshape[0] == batch) ? cnt / batch
                                                                          : cnt;
    dst.assign(f, f + std::max<int64_t>(n, 0));
  };

  const int po = std::clamp(primary_out, 0, nout - 1);
  decode(po, r.out);
  r.has_data = data_out >= 0 && data_out < nout;
  if(r.has_data)
    decode(data_out, r.data);

  // Capture new state values for feedback.
  std::size_t n = 0;
  for(auto& st : states)
  {
    if(st.out_index < 0 || st.out_index >= nout)
      continue;
    if(r.states.size() <= n)
      r.states.emplace_back();
    r.states[n].in_index = st.in_index;
    decode(st.out_index, r.states[n].values);
    ++n;
  }
  r.states.resize(n);
}

// Swaps the result in: the outputs and states take its buffers, and it takes
// theirs back, so the next run decodes into buffers with the capacity.
void applyResult(SequenceProcessor& self, SeqResult& r)
{
  self.failures.succeeded();
  std::swap(self.outputs.out.value, r.out);
  if(r.has_data)
    std::swap(self.outputs.data.value, r.data);
  else
    self.outputs.data.value.assign(
        self.outputs.out.value.begin(), self.outputs.out.value.end());

  if(!self.pipe)
    return;
  for(auto& ns : r.states)
    for(auto& sb : self.pipe->states)
      if(sb.in_index == ns.in_index)
      {
        if(ns.values.size() == sb.values.size())
          std::swap(sb.values, ns.values);
        break;
      }
}

static void routeSeqIO(SeqPipeline& P)
{
  // --- route I/O from the archetype --------------------------------------
  // Primary input: the first Sequence/Vector/Latent/Scalar input that is not a
  // recurrent state. Primary output: the first Sequence/Vector output that is
  // not a recurrent state; the second non-state output (if any) feeds Data.
  P.primaryIn = 0;
  P.primaryOut = 0;
  P.dataOut = -1;
  P.stateful = P.arch.stateful;


  // --- recurrent state: each state input with the output it is fed from ---
  // Paired once by classifyModel (names first, then free same-shape outputs,
  // dynamic dims as wildcards); claim those outputs BEFORE routing the data
  // outputs. An output is claimed by at most one state input.
  P.states.clear();
  std::vector<bool> out_is_state(P.arch.outputs.size(), false);
  if(P.stateful)
  {
    for(size_t i = 0; i < P.arch.inputs.size(); ++i)
    {
      if(P.arch.inputs[i].arch != PortArchetype::RecurrentState)
        continue;
      const int matched = P.arch.inputs[i].state_pair;
      if(matched < 0 || out_is_state[matched])
        continue; // no feedback output -> can't thread this state; skip it
      out_is_state[matched] = true;

      SeqState sb;
      sb.in_index = (int)i;
      sb.out_index = matched;
      sb.shape = P.spec.inputs[i].shape;
      for(auto& d : sb.shape)
        if(d <= 0)
          d = 1; // concretize dynamic state dims (batch etc.) to 1
      sb.dt = P.spec.inputs[i].elem_type;
      sb.values.assign((size_t)Onnx::flatSize(sb.shape), 0.f);
      P.states.push_back(std::move(sb));
    }
  }
  // A state pairing that consumes EVERY output is spurious (e.g. a stateless
  // same-shape autoencoder whose input==output shape got over-tagged): there
  // must remain at least one output to drive `Out`. Release state pairs (newest
  // first) until a data output survives.
  auto anyDataOut = [&]
  {
    for(size_t i = 0; i < P.arch.outputs.size(); ++i)
      if(!out_is_state[i])
        return true;
    return false;
  };
  while(!P.states.empty() && !anyDataOut())
  {
    out_is_state[P.states.back().out_index] = false;
    P.states.pop_back();
  }
  P.stateful = !P.states.empty(); // tagged-but-unpaired/released -> effectively stateless

  // The data input, once the threaded states are known: a sequence / vector
  // / latent first; else anything that is not a control (scalar), a bool
  // mask or token ids; else the first input that is not a threaded state. A
  // scalar declared first (t, then x) would otherwise take the payload and
  // fail on every frame. An input tagged as a state whose pairing was
  // released above (x [1,4] -> y [1,4]) is data again.
  {
    auto threaded = [&](size_t i) {
      return std::any_of(P.states.begin(), P.states.end(), [&](const SeqState& sb) {
        return sb.in_index == (int)i;
      });
    };
    auto pick = [&](auto pred) {
      for(size_t i = 0; i < P.arch.inputs.size(); ++i)
        if(!threaded(i) && P.spec.inputs[i].elem_type != TensorElemType::Bool
           && pred(P.arch.inputs[i].arch))
          return (int)i;
      return -1;
    };
    int p = pick([](PortArchetype a) {
      return a == PortArchetype::Sequence || a == PortArchetype::Vector
             || a == PortArchetype::Latent;
    });
    if(p < 0)
      p = pick([](PortArchetype a) {
        return a != PortArchetype::Scalar && a != PortArchetype::Unknown
               && a != PortArchetype::TokenSeq;
      });
    if(p < 0)
      for(size_t i = 0; i < P.arch.inputs.size(); ++i)
        if(!threaded(i))
        {
          p = (int)i;
          break;
        }
    P.primaryIn = std::max(p, 0);
  }

  // Route primary + secondary (Data) outputs from the non-state remainder.
  bool first_out = true;
  for(size_t i = 0; i < P.arch.outputs.size(); ++i)
  {
    if(out_is_state[i])
      continue;
    if(first_out)
    {
      P.primaryOut = (int)i;
      first_out = false;
    }
    else if(P.dataOut < 0)
    {
      P.dataOut = (int)i;
      break;
    }
  }

  std::vector<int> owned{P.primaryIn};
  for(const auto& sb : P.states)
    owned.push_back(sb.in_index);
  P.aux = Onnx::planAuxInputs(P.spec.inputs, owned);
}

// One zero inference at load, when the model's input shape does not depend
// on the payload (every dim but the batch is concrete): a model that cannot
// run is refused with its error instead of failing on every tick. When it
// fails at batch 1 on a dynamic batch, the graph may have one baked in
// (Informer: 2), which is tried next. Runs in the build job, on the worker.
static void probeSeq(SeqPipeline& P)
{
  const auto& pin = P.spec.inputs[P.primaryIn].shape;
  if(pin.empty())
    return;
  for(std::size_t k = 1; k < pin.size(); ++k)
    if(pin[k] <= 0)
      return;

  auto run = [&](int64_t b)
  {
    auto shape = pin;
    shape[0] = b;
    std::vector<float> input((size_t)Onnx::flatSize(shape), 0.f);
    auto st = P.states;
    const float params[2]{};
    SeqScratch sc;
    SeqResult r;
    runInference(
        *P.ctx, P.spec, input, shape, P.spec.inputs[P.primaryIn].elem_type,
        P.primaryIn, P.primaryOut, P.dataOut, st, P.aux, params, b, sc, r);
  };

  try
  {
    run(P.batch);
  }
  catch(const std::exception& e)
  {
    const std::string first = e.what();
    if(pin.size() >= 2 && pin[0] <= 0 && P.batch == 1 && P.states.empty())
    {
      try
      {
        run(2);
        P.batch = 2;
        std::fprintf(
            stderr,
            "Sequence Processor: %s: the graph needs a batch of 2, the input "
            "is fed twice\n",
            P.path.c_str());
        return;
      }
      catch(...)
      {
      }
    }
    P.refusal = "the model does not run: " + first;
  }
}

// Worker side of a model change: the session (from the file, since the
// port's mapping may be gone by now), the routing, the probe, the window.
static std::shared_ptr<SeqPipeline>
makeSeqPipeline(const std::string& path, std::size_t model_bytes)
{
  auto pp = std::make_shared<SeqPipeline>();
  auto& P = *pp;
  P.path = path;
  P.model_bytes = model_bytes;
  P.ctx = Onnx::loadRunContext(path);
  P.spec = P.ctx->readModelSpec();
  P.arch = Onnx::classifyModel(Onnx::toArchIO(P.spec));
  if(P.spec.inputs.empty() || P.spec.outputs.empty())
    return pp;
  routeSeqIO(P);

  // A concrete declared batch is honoured; a dynamic one is 1 unless the
  // probe finds that the graph only runs at another.
  const auto& pin = P.spec.inputs[P.primaryIn].shape;
  P.batch = (pin.size() >= 2 && pin[0] > 1 && !P.stateful) ? pin[0] : 1;
  probeSeq(P);

  // A fixed [1,T,F] input: its window is sized now rather than on the
  // processing thread when the first frames arrive.
  if(pin.size() >= 3 && pin[1] > 1 && pin.back() > 0)
    P.window.configure(pin[1], pin.back());
  P.in_scratch.reserve(4096);
  return pp;
}
} // namespace

void SequenceProcessor::resetState()
{
  ++gen;
  if(!pipe)
    return;
  for(auto& s : pipe->states)
    std::fill(s.values.begin(), s.values.end(), 0.f);
  pipe->window.reset();
}

void SequenceProcessor::reportError(std::string_view what)
{
  failures.failed(name(), inputs.model.file.filename, what);
}

// On the processing thread: the new model replaces the running one. A job of
// the old model still running brings nothing back (gen) but keeps
// inferenceInProgress until it returns, so that jobs never overlap.
void SequenceProcessor::modelInstalled()
{
  pipe = models.model();
  if(!pipe)
    return;
  failures.succeeded();
  resetState();
  resolveWindow();
}

// The model needs a fixed T when the primary input rank>=3 with a concrete time
// dim: Sliding lets us stream single frames into it. Re-run whenever the
// Window setting changes, not only on reload.
void SequenceProcessor::resolveWindow()
{
  lastWindowMode = inputs.window_mode.value;
  resolvedWindow = Onnx::WindowMode::Passthrough;
  if(!pipe)
    return;
  const auto& pin = pipe->spec.inputs[pipe->primaryIn].shape;
  const bool fixedTime = pin.size() >= 3 && pin[1] > 1;
  switch(inputs.window_mode.value)
  {
    case SeqWindowMode::Sliding:
      resolvedWindow = Onnx::WindowMode::Sliding;
      break;
    case SeqWindowMode::Passthrough:
      resolvedWindow = Onnx::WindowMode::Passthrough;
      break;
    case SeqWindowMode::Auto:
    default:
      resolvedWindow
          = fixedTime ? Onnx::WindowMode::Sliding : Onnx::WindowMode::Passthrough;
      break;
  }
  // Start the history over, in the buffers it has (no reallocation).
  pipe->window.reset();
}

void SequenceProcessor::operator()()
try
{
  ONNX_PROF_SCOPE(Total);
  if(!available || inputs.model.current_model_invalid
     || inputs.model.file.bytes.empty())
    return;

  // A new file: its pipeline is built (and probed) on the worker.
  if(!models.requested().is(inputs.model))
    models.request(worker, ModelFile::of(inputs.model));
  if(!pipe || pipe->spec.inputs.empty() || pipe->spec.outputs.empty())
    return;
  auto& P = *pipe;
  if(inputs.window_mode.value != lastWindowMode)
    resolveWindow();

  if(inputs.reset)
    resetState();

  if(inputs.in.value.empty())
    return;
  if(!failures.ready())
    return; // the last frames failed the same way: backing off

  // Build the primary input ([1,T,F] / [1,D]) from the payload, windowing as
  // resolved. feat_hint comes from a concrete last declared dim if present.
  const auto& declared = P.spec.inputs[P.primaryIn].shape;
  const int64_t feat_hint
      = (!declared.empty() && declared.back() > 0) ? declared.back() : 0;

  Onnx::InputBuild b = Onnx::buildInput(
      declared, inputs.in.value, resolvedWindow, P.window, P.in_scratch, feat_hint,
      /*require_full_window*/ true);
  if(!b.ready || !b.data || b.count <= 0)
    return; // window not yet filled, or nothing to feed

  // Into the pipeline's buffer (the job hands its own back, see dispatchInfer).
  P.input.resize((std::size_t)(b.count * std::max<int64_t>(P.batch, 1)));
  std::copy_n(b.data, b.count, P.input.begin());
  normalizeFrames(std::span<float>(P.input.data(), (std::size_t)b.count), b.shape,
                  inputs.normalize.value);
  if(P.batch > 1 && !b.shape.empty())
  {
    for(int64_t k = 1; k < P.batch; ++k)
      std::copy_n(P.input.begin(), b.count, P.input.begin() + k * b.count);
    b.shape[0] = P.batch;
  }
  P.ishape.assign(b.shape.begin(), b.shape.end());

  // Heavy if the model is large or the flattened input is big.
  const bool heavy = P.model_bytes > 64u * 1024 * 1024 || b.count > 1 << 20;
  dispatchInfer(heavy);
}
catch(const std::exception& e)
{
  // A frame that fails to run (a payload the model cannot take) is skipped;
  // the next one may work. Models that cannot run at all fail the probe.
  reportError(e.what());
}
catch(...)
{
  reportError("unknown error");
}

void SequenceProcessor::dispatchInfer(bool force_async)
{
  auto& P = *pipe;
  const TensorElemType in_dt = P.spec.inputs[P.primaryIn].elem_type;

  if(force_async)
  {
    if(inferenceInProgress)
      return; // drop this frame; a job is in flight
    inferenceInProgress = true;

    // Pooled job: lock-free acquire; recycled vectors keep their capacity so
    // the assignments below don't allocate in steady state.
    auto job = JobPool<SeqInferJob>::instance().acquire();
    job->ctx = P.ctx;
    std::swap(job->input, P.input); // the job's recycled buffer comes back
    job->ishape.assign(P.ishape.begin(), P.ishape.end());
    job->in_dt = in_dt;
    job->primary_in_index = P.primaryIn;
    job->primary_out_index = P.primaryOut;
    job->aux = P.aux;
    job->params[0] = inputs.param1.value;
    job->params[1] = inputs.param2.value;
    job->data_out_index = P.dataOut;
    job->batch = P.batch;
    job->gen = gen;
    // The job is recycled: resize + indexed assignment (not push_back) so
    // stale states from a previous use never accumulate.
    job->states.resize(P.states.size());
    for(std::size_t k = 0; k < P.states.size(); ++k)
    {
      auto& js = job->states[k];
      js.in_index = P.states[k].in_index;
      js.out_index = P.states[k].out_index;
      js.shape.assign(P.states[k].shape.begin(), P.states[k].shape.end());
      js.dt = P.states[k].dt;
      js.values.assign(P.states[k].values.begin(), P.states[k].values.end());
    }
    worker.request(std::move(job));
    return;
  }

  // Synchronous: the states are fed as they are, and the pipeline's scratch
  // and result are reused.
  const float params[2]{inputs.param1.value, inputs.param2.value};
  runInference(
      *P.ctx, P.spec, P.input, P.ishape, in_dt, P.primaryIn, P.primaryOut, P.dataOut,
      P.states, P.aux, params, P.batch, P.scratch, P.result);
  applyResult(*this, P.result);
}

std::function<void(SequenceProcessor&)>
SequenceProcessor::worker::work(std::unique_ptr<SeqInferJob> job)
{
  // RAII: whatever path we exit through, the job goes back to the lock-free
  // pool (with its buffer capacities intact) once the results are moved out.
  struct Recycle
  {
    std::unique_ptr<SeqInferJob>& j;
    ~Recycle()
    {
      if(j)
      {
        j->ctx.reset(); // don't keep the ORT session alive from the pool
      }
      JobPool<SeqInferJob>::instance().release(std::move(j));
    }
  } recycle{job};

  if(job && job->load.active())
    return ModelLoader<SeqPipeline, SeqInferJob>::work<&SequenceProcessor::models, SequenceProcessor>(
        job->load,
        [](const ModelFile& f) {
      std::shared_ptr<SeqPipeline> p;
      try
      {
        p = makeSeqPipeline(f.path, f.size);
      }
      catch(const std::exception& e)
      {
        throw std::runtime_error(std::string("cannot load the model: ") + e.what());
      }
      if(!p->refusal.empty())
        throw std::runtime_error(p->refusal);
      return p;
    },
        [](SequenceProcessor& self) { self.modelInstalled(); },
        [](SequenceProcessor& self, std::string_view what) {
      self.failures.failed(SequenceProcessor::name(), self.models.requested().path, what);
      // Not when another file was picked since: that one is still to load.
      if(self.models.requested().is(self.inputs.model))
        self.inputs.model.current_model_invalid = true;
    });

  if(!job || !job->ctx)
    return [](SequenceProcessor& self) { self.inferenceInProgress = false; };
  try
  {
    const auto& spec = job->ctx->readModelSpec();
    runInference(
        *job->ctx, spec, job->input, job->ishape, job->in_dt,
        job->primary_in_index, job->primary_out_index, job->data_out_index,
        job->states, job->aux, job->params, job->batch, job->scratch, job->result);
    // The tensors and the session are released here, not on the processing
    // thread.
    job->scratch.ins.clear();
    job->scratch.outs.clear();
    job->ctx.reset();
    std::shared_ptr<SeqInferJob> done{job.release(), [](SeqInferJob* j) {
      JobPool<SeqInferJob>::instance().release(std::unique_ptr<SeqInferJob>(j));
    }};
    return [done = std::move(done)](SequenceProcessor& self)
    {
      self.inferenceInProgress = false;
      if(done->gen != self.gen)
        return; // Reset or a new model while it ran: its state is stale
      applyResult(self, done->result);
    };
  }
  catch(const std::exception& e)
  {
    return [what = std::string(e.what())](SequenceProcessor& self)
    {
      self.inferenceInProgress = false;
      self.reportError(what);
    };
  }
  catch(...)
  {
    return [](SequenceProcessor& self)
    {
      self.inferenceInProgress = false;
      self.reportError("unknown error");
    };
  }
}

}
