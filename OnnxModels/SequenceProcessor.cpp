#include "SequenceProcessor.hpp"

#include <OnnxModels/JobPool.hpp>

#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace OnnxModels
{
using Onnx::ArchIO;
using Onnx::NodeKind;
using Onnx::PortArchetype;
using Onnx::TensorElemType;

namespace
{
// Build the dependency-free ArchIO view (shape + name + dtype) the classifier
// consumes, from the ORT-derived ModelSpec.
ArchIO toArchIO(const Onnx::ModelSpec& s)
{
  ArchIO io;
  io.inputs.reserve(s.inputs.size());
  io.outputs.reserve(s.outputs.size());
  for(const auto& p : s.inputs)
    io.inputs.push_back({p.name, p.shape, p.elem_type});
  for(const auto& p : s.outputs)
    io.outputs.push_back({p.name, p.shape, p.elem_type});
  return io;
}

int64_t flatPos(const std::vector<int64_t>& s)
{
  int64_t p = 1;
  for(auto d : s)
    p *= (d > 0 ? d : 1);
  return p;
}

// Decode any supported numeric output tensor to float into `scratch`, returning
// a pointer to a contiguous float view (scratch is only touched for non-fp32).
const float* toFloatView(
    const void* raw, int64_t count, TensorElemType dt, std::vector<float>& scratch)
{
  if(dt == TensorElemType::Float)
    return reinterpret_cast<const float*>(raw);

  scratch.resize((size_t)std::max<int64_t>(count, 0));
  switch(dt)
  {
    case TensorElemType::Double:
    {
      auto* p = reinterpret_cast<const double*>(raw);
      for(int64_t i = 0; i < count; ++i)
        scratch[i] = (float)p[i];
      break;
    }
    case TensorElemType::Int64:
    {
      auto* p = reinterpret_cast<const int64_t*>(raw);
      for(int64_t i = 0; i < count; ++i)
        scratch[i] = (float)p[i];
      break;
    }
    case TensorElemType::Int32:
    {
      auto* p = reinterpret_cast<const int32_t*>(raw);
      for(int64_t i = 0; i < count; ++i)
        scratch[i] = (float)p[i];
      break;
    }
    case TensorElemType::Uint8:
    {
      auto* p = reinterpret_cast<const uint8_t*>(raw);
      for(int64_t i = 0; i < count; ++i)
        scratch[i] = (float)p[i];
      break;
    }
    case TensorElemType::Int8:
    {
      auto* p = reinterpret_cast<const int8_t*>(raw);
      for(int64_t i = 0; i < count; ++i)
        scratch[i] = (float)p[i];
      break;
    }
    default:
      std::fill(scratch.begin(), scratch.end(), 0.f);
      break;
  }
  return scratch.data();
}

// Wrap a float buffer as an ORT tensor of the model's declared input dtype.
// fp32 is wrapped zero-copy; int/double inputs are converted into the supplied
// staging buffer (which must outlive the returned Value).
Ort::Value buildTensor(
    std::vector<float>& f, const std::vector<int64_t>& shape, TensorElemType dt,
    std::vector<int64_t>& i64_buf, std::vector<int32_t>& i32_buf,
    std::vector<double>& f64_buf)
{
  const int64_t n = (int64_t)f.size();
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  switch(dt)
  {
    case TensorElemType::Int64:
      i64_buf.assign(f.begin(), f.end());
      return Ort::Value::CreateTensor<int64_t>(
          mem, i64_buf.data(), (size_t)n, shape.data(), shape.size());
    case TensorElemType::Int32:
      i32_buf.assign(f.begin(), f.end());
      return Ort::Value::CreateTensor<int32_t>(
          mem, i32_buf.data(), (size_t)n, shape.data(), shape.size());
    case TensorElemType::Double:
      f64_buf.assign(f.begin(), f.end());
      return Ort::Value::CreateTensor<double>(
          mem, f64_buf.data(), (size_t)n, shape.data(), shape.size());
    case TensorElemType::Float:
    default:
      return Onnx::vec_to_tensor<float>(f, shape);
  }
}

// Decoded result staged in heap (so readback also runs off the worker thread):
// the primary output (Out), an optional secondary Data output, and the new
// recurrent-state values to swap into the node's persistent buffers.
struct SeqResult
{
  std::vector<float> out;
  std::vector<float> data;
  bool has_data = false;
  struct NewState
  {
    int in_index = 0;
    std::vector<float> values;
  };
  std::vector<NewState> states;
};

void applyResult(SequenceProcessor& self, SeqResult& r);
} // namespace

SequenceProcessor::SequenceProcessor() noexcept
{
  in_scratch.reserve(4096);
}

SequenceProcessor::~SequenceProcessor() = default;

void SequenceProcessor::resetState()
{
  for(auto& s : states)
    std::fill(s.values.begin(), s.values.end(), 0.f);
  window.reset();
}

void SequenceProcessor::reloadModel()
{
  ctx = std::make_shared<Onnx::OnnxRunContext>(
      inputs.model.file.bytes, inputs.model.file.filename);
  spec = ctx->readModelSpec();
  arch = Onnx::classifyModel(toArchIO(spec));
  lastModelPath = inputs.model.file.filename;

  // --- route I/O from the archetype --------------------------------------
  // Primary input: the first Sequence/Vector/Latent/Scalar input that is not a
  // recurrent state. Primary output: the first Sequence/Vector output that is
  // not a recurrent state; the second non-state output (if any) feeds Data.
  primaryIn = 0;
  primaryOut = 0;
  dataOut = -1;
  stateful = arch.stateful;

  for(size_t i = 0; i < arch.inputs.size(); ++i)
  {
    if(arch.inputs[i].arch != PortArchetype::RecurrentState)
    {
      primaryIn = (int)i;
      break;
    }
  }

  // --- recurrent state: each state input with the output it is fed from ---
  // Paired once by classifyModel (names first, then free same-shape outputs,
  // dynamic dims as wildcards); claim those outputs BEFORE routing the data
  // outputs. An output is claimed by at most one state input.
  states.clear();
  std::vector<bool> out_is_state(arch.outputs.size(), false);
  if(stateful)
  {
    for(size_t i = 0; i < arch.inputs.size(); ++i)
    {
      if(arch.inputs[i].arch != PortArchetype::RecurrentState)
        continue;
      const int matched = arch.inputs[i].state_pair;
      if(matched < 0 || out_is_state[matched])
        continue; // no feedback output -> can't thread this state; skip it
      out_is_state[matched] = true;

      StateBuf sb;
      sb.in_index = (int)i;
      sb.out_index = matched;
      sb.shape = spec.inputs[i].shape;
      for(auto& d : sb.shape)
        if(d <= 0)
          d = 1; // concretize dynamic state dims (batch etc.) to 1
      sb.dt = spec.inputs[i].elem_type;
      sb.values.assign((size_t)flatPos(sb.shape), 0.f);
      states.push_back(std::move(sb));
    }
  }
  // A state pairing that consumes EVERY output is spurious (e.g. a stateless
  // same-shape autoencoder whose input==output shape got over-tagged): there
  // must remain at least one output to drive `Out`. Release state pairs (newest
  // first) until a data output survives.
  auto anyDataOut = [&]
  {
    for(size_t i = 0; i < arch.outputs.size(); ++i)
      if(!out_is_state[i])
        return true;
    return false;
  };
  while(!states.empty() && !anyDataOut())
  {
    out_is_state[states.back().out_index] = false;
    states.pop_back();
  }
  stateful = !states.empty(); // tagged-but-unpaired/released -> effectively stateless

  // Route primary + secondary (Data) outputs from the non-state remainder.
  bool first_out = true;
  for(size_t i = 0; i < arch.outputs.size(); ++i)
  {
    if(out_is_state[i])
      continue;
    if(first_out)
    {
      primaryOut = (int)i;
      first_out = false;
    }
    else if(dataOut < 0)
    {
      dataOut = (int)i;
      break;
    }
  }

  std::vector<int> owned{primaryIn};
  for(const auto& sb : states)
    owned.push_back(sb.in_index);
  aux = Onnx::planAuxInputs(spec.inputs, owned);

  resolveWindow();
  resetState();

  // A concrete declared batch is honoured; a dynamic one is 1 unless the
  // probe finds that the graph only runs at another.
  const auto& pin = spec.inputs[primaryIn].shape;
  batch = (pin.size() >= 2 && pin[0] > 1 && !stateful) ? pin[0] : 1;
  lastError.clear();
  if(!probe())
    inputs.model.current_model_invalid = true;
}

void SequenceProcessor::reportError(std::string_view what)
{
  if(what == lastError)
    return;
  lastError = what;
  std::fprintf(
      stderr, "Sequence Processor: %s: %s\n", lastModelPath.c_str(),
      lastError.c_str());
}

// The model needs a fixed T when the primary input rank>=3 with a concrete time
// dim: Sliding lets us stream single frames into it. Re-run whenever the
// Window setting changes, not only on reload, or a change did nothing until
// the model was reloaded.
void SequenceProcessor::resolveWindow()
{
  lastWindowMode = inputs.window_mode.value;
  resolvedWindow = Onnx::WindowMode::Passthrough;
  const auto& pin = spec.inputs[primaryIn].shape;
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
  window.configure(0, 0); // forces reconfigure on first buildInput
}

void SequenceProcessor::operator()()
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
    try
    {
      reloadModel();
    }
    catch(const std::exception& e)
    {
      lastModelPath = inputs.model.file.filename;
      reportError(e.what());
      ctx.reset();
      inputs.model.current_model_invalid = true;
      return;
    }
    if(inputs.model.current_model_invalid)
      return; // the probe failed
  }
  if(spec.inputs.empty() || spec.outputs.empty())
    return;
  if(inputs.window_mode.value != lastWindowMode)
    resolveWindow();

  if(inputs.reset)
    resetState();

  if(inputs.in.value.empty())
    return;

  // Build the primary input ([1,T,F] / [1,D]) from the payload, windowing as
  // resolved. feat_hint comes from a concrete last declared dim if present.
  const auto& declared = spec.inputs[primaryIn].shape;
  const int64_t feat_hint
      = (!declared.empty() && declared.back() > 0) ? declared.back() : 0;

  Onnx::InputBuild b = Onnx::buildInput(
      declared, inputs.in.value, resolvedWindow, window, in_scratch, feat_hint,
      /*require_full_window*/ true);
  if(!b.ready || !b.data || b.count <= 0)
    return; // window not yet filled, or nothing to feed

  std::vector<float> input(b.data, b.data + b.count);
  if(batch > 1 && !b.shape.empty())
  {
    input.reserve(input.size() * batch);
    for(int64_t k = 1; k < batch; ++k)
      input.insert(input.end(), b.data, b.data + b.count);
    b.shape[0] = batch;
  }

  // Heavy if the model is large or the flattened input is big.
  const bool heavy = inputs.model.file.bytes.size() > 64u * 1024 * 1024
                     || b.count > 1 << 20;
  dispatchInfer(std::move(input), std::move(b.shape), heavy);
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

namespace
{
// Run a full multi-IO inference: place the primary input + all recurrent-state
// inputs at their model-declared indices, run every declared output, decode the
// primary/data outputs and capture the new state values.
SeqResult runInference(
    Onnx::OnnxRunContext& ctx, const Onnx::ModelSpec& spec,
    std::vector<float>& input, const std::vector<int64_t>& ishape,
    TensorElemType in_dt, int primary_in, int primary_out, int data_out,
    std::vector<SeqInferJob::State>& states, const std::vector<Onnx::AuxPlan>& aux,
    std::span<const float> params, int64_t batch)
{
  const int nin = (int)spec.input_names_char.size();
  const int nout = (int)spec.output_names_char.size();

  // Staging buffers must outlive the inference (tensors are non-owning views).
  std::vector<int64_t> i64_buf;
  std::vector<int32_t> i32_buf;
  std::vector<double> f64_buf;
  // Per-state staging (own vectors so views stay valid).
  std::vector<std::vector<int64_t>> s_i64(states.size());
  std::vector<std::vector<int32_t>> s_i32(states.size());
  std::vector<std::vector<double>> s_f64(states.size());

  std::vector<Ort::Value> ins;
  ins.reserve(nin);
  for(int i = 0; i < nin; ++i)
    ins.emplace_back(nullptr);

  ins[primary_in]
      = buildTensor(input, ishape, in_dt, i64_buf, i32_buf, f64_buf);

  for(size_t k = 0; k < states.size(); ++k)
  {
    auto& st = states[k];
    if(st.in_index < 0 || st.in_index >= nin)
      continue;
    ins[st.in_index] = buildTensor(
        st.values, st.shape, st.dt, s_i64[k], s_i32[k], s_f64[k]);
  }

  // The other inputs, by the aux plan: sr gets the rate, scalars Param 1 / 2,
  // the rest zeros. Each keeps its own storage while the tensors are alive.
  std::vector<std::vector<uint8_t>> aux_store(aux.size());
  std::vector<std::vector<int64_t>> aux_shape(aux.size());
  Onnx::AuxHost host{.params = params, .primary_shape = ishape, .batch = batch};
  for(std::size_t k = 0; k < aux.size(); ++k)
    if(aux[k].index >= 0 && aux[k].index < nin && !ins[aux[k].index])
      ins[aux[k].index] = Onnx::fillAux(aux[k], host, aux_store[k], aux_shape[k]);

  // Anything still unfilled (a state the pairing released) gets zeros so ORT
  // has a value for every declared name.
  std::vector<std::vector<float>> filler(nin);
  std::vector<std::vector<int64_t>> f_i64(nin);
  std::vector<std::vector<int32_t>> f_i32(nin);
  std::vector<std::vector<double>> f_f64(nin);
  for(int i = 0; i < nin; ++i)
  {
    if(ins[i])
      continue;
    auto shp = spec.inputs[i].shape;
    for(auto& d : shp)
      if(d <= 0)
        d = 1;
    filler[i].assign((size_t)flatPos(shp), 0.f);
    ins[i] = buildTensor(
        filler[i], shp, spec.inputs[i].elem_type, f_i64[i], f_i32[i], f_f64[i]);
  }

  std::vector<Ort::Value> outs;
  outs.reserve(nout);
  for(int i = 0; i < nout; ++i)
    outs.emplace_back(nullptr);

  ctx.infer(spec, ins, outs);

  SeqResult r;
  std::vector<float> scratch;
  auto decode = [&](int idx, std::vector<float>& dst)
  {
    const auto info = outs[idx].GetTensorTypeAndShapeInfo();
    const int64_t cnt = (int64_t)info.GetElementCount();
    const TensorElemType odt
        = Onnx::fromOrtElementType(info.GetElementType());
    const void* raw = outs[idx].GetTensorData<uint8_t>();
    const float* f = toFloatView(raw, cnt, odt, scratch);
    // A replicated batch: every slice is the same, keep the first.
    const auto oshape = info.GetShape();
    const int64_t n = (batch > 1 && oshape.size() >= 2 && oshape[0] == batch)
                          ? cnt / batch
                          : cnt;
    dst.assign(f, f + std::max<int64_t>(n, 0));
  };

  const int po = std::clamp(primary_out, 0, nout - 1);
  decode(po, r.out);
  if(data_out >= 0 && data_out < nout)
  {
    decode(data_out, r.data);
    r.has_data = true;
  }

  // Capture new state values for feedback.
  for(auto& st : states)
  {
    if(st.out_index < 0 || st.out_index >= nout)
      continue;
    SeqResult::NewState ns;
    ns.in_index = st.in_index;
    decode(st.out_index, ns.values);
    r.states.push_back(std::move(ns));
  }
  return r;
}

void applyResult(SequenceProcessor& self, SeqResult& r)
{
  self.outputs.out.value = std::move(r.out);
  if(r.has_data)
    self.outputs.data.value = std::move(r.data);
  else
    self.outputs.data.value = self.outputs.out.value;

  // Swap new state values back into the persistent buffers (size-checked).
  for(auto& ns : r.states)
    for(auto& sb : self.states)
      if(sb.in_index == ns.in_index)
      {
        if(ns.values.size() == sb.values.size())
          sb.values = std::move(ns.values);
        break;
      }
}
} // namespace

void SequenceProcessor::dispatchInfer(
    std::vector<float> input, std::vector<int64_t> ishape, bool force_async)
{
  const TensorElemType in_dt = spec.inputs[primaryIn].elem_type;

  if(force_async)
  {
    if(inferenceInProgress)
      return; // drop this frame; a job is in flight
    inferenceInProgress = true;

    // Pooled job: lock-free acquire; recycled vectors keep their capacity so
    // the assignments below don't allocate in steady state.
    auto job = JobPool<SeqInferJob>::instance().acquire();
    job->ctx = ctx;
    job->input = std::move(input);
    job->ishape = std::move(ishape);
    job->in_dt = in_dt;
    job->primary_in_index = primaryIn;
    job->primary_out_index = primaryOut;
    job->aux = aux;
    job->params[0] = inputs.param1.value;
    job->params[1] = inputs.param2.value;
    job->data_out_index = dataOut;
    job->batch = batch;
    // The job is recycled: resize + indexed assignment (not push_back) so
    // stale states from a previous use never accumulate.
    job->states.resize(states.size());
    for(std::size_t k = 0; k < states.size(); ++k)
    {
      auto& js = job->states[k];
      js.in_index = states[k].in_index;
      js.out_index = states[k].out_index;
      js.shape = states[k].shape;
      js.dt = states[k].dt;
      js.values = states[k].values;
    }
    worker.request(std::move(job));
    return;
  }

  std::vector<SeqInferJob::State> st;
  for(auto& sb : states)
    st.push_back({sb.in_index, sb.out_index, sb.shape, sb.dt, sb.values});

  const float params[2]{inputs.param1.value, inputs.param2.value};
  SeqResult r = runInference(
      *ctx, spec, input, ishape, in_dt, primaryIn, primaryOut, dataOut, st, aux,
      params, batch);
  applyResult(*this, r);
}

// One zero inference at load, when the model's input shape does not depend
// on the payload (every dim but the batch is concrete): a model that cannot
// run is reported with its error instead of failing on every tick. When it
// fails at batch 1 on a dynamic batch, the graph may have one baked in
// (Informer: 2), which is tried next.
bool SequenceProcessor::probe()
{
  const auto& pin = spec.inputs[primaryIn].shape;
  if(pin.empty())
    return true;
  for(std::size_t k = 1; k < pin.size(); ++k)
    if(pin[k] <= 0)
      return true;
  if(inputs.model.file.bytes.size() > 64u * 1024 * 1024)
    return true; // the probe would stall the render thread

  auto run = [&](int64_t b)
  {
    auto shape = pin;
    shape[0] = b;
    std::vector<float> input((size_t)flatPos(shape), 0.f);
    std::vector<SeqInferJob::State> st;
    for(auto& sb : states)
      st.push_back({sb.in_index, sb.out_index, sb.shape, sb.dt, sb.values});
    const float params[2]{};
    runInference(
        *ctx, spec, input, shape, spec.inputs[primaryIn].elem_type, primaryIn,
        primaryOut, dataOut, st, aux, params, b);
  };

  try
  {
    run(batch);
    return true;
  }
  catch(const std::exception& e)
  {
    const std::string first = e.what();
    if(pin.size() >= 2 && pin[0] <= 0 && batch == 1 && states.empty())
    {
      try
      {
        run(2);
        batch = 2;
        std::fprintf(
            stderr,
            "Sequence Processor: %s: the graph needs a batch of 2, the input "
            "is fed twice\n",
            lastModelPath.c_str());
        return true;
      }
      catch(...)
      {
      }
    }
    reportError("the model does not run: " + first);
    return false;
  }
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
        j->ctx.reset(); // don't keep the ORT session alive from the pool
      JobPool<SeqInferJob>::instance().release(std::move(j));
    }
  } recycle{job};

  if(!job || !job->ctx)
    return [](SequenceProcessor& self) { self.inferenceInProgress = false; };
  try
  {
    const auto& spec = job->ctx->readModelSpec();
    SeqResult r = runInference(
        *job->ctx, spec, job->input, job->ishape, job->in_dt,
        job->primary_in_index, job->primary_out_index, job->data_out_index,
        job->states, job->aux, job->params, job->batch);
    return [r = std::move(r)](SequenceProcessor& self) mutable
    {
      self.inferenceInProgress = false;
      applyResult(self, r);
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
