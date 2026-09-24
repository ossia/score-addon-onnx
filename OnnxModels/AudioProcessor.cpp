#include "AudioProcessor.hpp"

#include <OnnxModels/JobPool.hpp>

#include <Onnx/helpers/AudioRate.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/TensorToTexture.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <cstring>
#include <utility>

namespace OnnxModels
{
using Onnx::PortArchetype;
using Onnx::TensorElemType;

namespace
{
// Build the dependency-free ArchIO view classifyModel wants from a ModelSpec.
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

// Best guess at the model's native sample-rate from its identity. The node's
// kind is fixed (audio->audio), so we can't read SR from the graph; we use the
// well-known rates for the common model families, falling back to host rate.
// The frames of the selected stem. A [B,S,C,N] output (Demucs) holds S stems
// of C channels, which were pushed as one scrambled mono stream; Param 1 picks
// the stem. Any other output is a single stem.
struct StemView
{
  const float* data;
  int channels;
  int64_t frames;
};
StemView selectStem(
    const float* f, int64_t cnt, const std::vector<int64_t>& osh, int fallback_channels,
    float select)
{
  const auto os = Onnx::WaveformShape::fromOutputShape(osh, fallback_channels);
  const int oc = os.channels > 0 ? os.channels : std::max(fallback_channels, 1);
  const int S = std::max(os.stems, 1);
  const int64_t per_stem = cnt / S;
  const int s = std::clamp((int)(select * (float)S), 0, S - 1);
  return {f + s * per_stem, oc, oc > 0 ? per_stem / oc : per_stem};
}
} // namespace

AudioProcessor::AudioProcessor() noexcept
{
  staged.reserve(2 * 48000);
  out_planar.reserve(2 * 48000);
}

AudioProcessor::~AudioProcessor() = default;

void AudioProcessor::prepare(halp::setup info)
{
  host_rate = info.rate > 0 ? info.rate : 48000.0;
  host_in_channels = info.input_channels;
  max_frames = info.frames > 0 ? (std::size_t)info.frames : 4096;
  // Force re-resolution of the audio pipeline against the new host config.
  lastModelPath.clear();
  ctx.reset();
}

void AudioProcessor::reloadModel()
{
  ctx = std::make_shared<Onnx::OnnxRunContext>(
      inputs.model.file.bytes, inputs.model.file.filename);
  spec = ctx->readModelSpec();
  arch = Onnx::classifyModel(toArchIO(spec));
  lastModelPath = inputs.model.file.filename;
  resolveIO();
}

// Find the waveform in/out ports, the model sample-rate, and wire up recurrent
// state ports (input shape matches an output, or *state* names). This node's
// kind is FIXED to audio->audio, so the *primary* audio input is treated as a
// waveform even if classifyModel mistags it (Silero-VAD's bare [1,512] 'input').
void AudioProcessor::resolveIO()
{
  wave_in_index = -1;
  wave_out_index = -1;
  states.clear();

  // Primary waveform input: prefer a Waveform/Spectrogram archetype, else the
  // first non-state, non-scalar input.
  for(int i = 0; i < (int)arch.inputs.size(); ++i)
  {
    const auto a = arch.inputs[i].arch;
    if(a == PortArchetype::Waveform || a == PortArchetype::Spectrogram)
    {
      wave_in_index = i;
      break;
    }
  }
  if(wave_in_index < 0)
  {
    for(int i = 0; i < (int)arch.inputs.size(); ++i)
    {
      const auto a = arch.inputs[i].arch;
      if(a != PortArchetype::RecurrentState && a != PortArchetype::Scalar
         && a != PortArchetype::TokenSeq)
      {
        wave_in_index = i;
        break;
      }
    }
  }
  if(wave_in_index < 0)
    wave_in_index = 0;

  // Waveform outputs: each separation stem / the single enhanced or vocoded
  // signal. Param 1 selects the stem when there is more than one.
  wave_out_indices.clear();
  for(int i = 0; i < (int)arch.outputs.size(); ++i)
    if(arch.outputs[i].arch == PortArchetype::Waveform)
      wave_out_indices.push_back(i);
  if(wave_out_indices.empty())
    wave_out_indices.push_back(0);
  wave_out_index = wave_out_indices.front();

  // Recurrent state: inputs tagged RecurrentState, each fed back from the
  // output classifyModel paired it with (names first, then free same-shape
  // outputs) -- never from the audio output itself.
  for(int i = 0; i < (int)arch.inputs.size(); ++i)
  {
    if(arch.inputs[i].arch != PortArchetype::RecurrentState)
      continue;
    StatePort sp;
    sp.in_index = i;
    sp.shape = spec.inputs[i].shape;
    for(auto& d : sp.shape)
      if(d < 0)
        d = 1; // resolve dynamic state dims to 1
    sp.out_index = arch.inputs[i].state_pair;
    if(sp.out_index == wave_out_index)
      sp.out_index = -1;
    sp.data.assign((std::size_t)flatPositive(sp.shape), 0.f);
    states.push_back(std::move(sp));
  }

  in_shape = Onnx::WaveformShape::fromInputShape(spec.inputs[wave_in_index].shape);
  // [1,F,T] with F > 8: a spectrogram, not F channels of audio. A vocoder
  // (mel bins in; a waveform out, or Vocos' magnitude / cos / sin spectrum)
  // gets the log-mel of the input. STFT bins (F = 2^k + 1: a magnitude/phase
  // model) or another output need a pipeline this node does not have:
  // refused once, instead of ORT throwing on every block.
  mel_input = false;
  spec_out[0] = spec_out[1] = spec_out[2] = -1;
  {
    const auto& s = spec.inputs[wave_in_index].shape;
    if(s.size() == 3 && s[1] > 8)
    {
      const int64_t f = s[1];
      const bool stftBins = ((f - 1) & (f - 2)) == 0;
      const auto& o = spec.outputs[wave_out_index].shape;
      const bool waveOut = o.size() <= 2 || (o.size() == 3 && o[1] <= 2);
      for(int i = 0; i < (int)spec.outputs.size(); ++i)
      {
        const auto& n = spec.outputs[i].name;
        const int k = (n == "mag" || n == "magnitude") ? 0
                      : (n == "x" || n == "cos")       ? 1
                      : (n == "y" || n == "sin")       ? 2
                                                       : -1;
        if(k >= 0 && spec.outputs[i].shape.size() == 3)
          spec_out[k] = i;
      }
      const bool spectrumOut = spec_out[0] >= 0 && spec_out[1] >= 0 && spec_out[2] >= 0;
      if(!spectrumOut)
        spec_out[0] = spec_out[1] = spec_out[2] = -1;
      if(stftBins || !(waveOut || spectrumOut))
      {
        std::fprintf(
            stderr,
            "Audio Processor: %s: the input [1,%lld,T] is a spectrogram and "
            "the model is not a mel vocoder; not supported\n",
            lastModelPath.c_str(), (long long)f);
        inputs.model.current_model_invalid = true;
        return;
      }
      mel_input = true;
    }
  }
  out_shape = Onnx::WaveformShape::fromOutputShape(
      spec.outputs[wave_out_index].shape, in_shape.channels);
  if(out_shape.channels < 1)
    out_shape.channels = in_shape.channels;

  // The other inputs: scalars take Params 2..4 (Param 1 picks the stem).
  {
    std::vector<int> owned{wave_in_index};
    for(const auto& s : states)
      owned.push_back(s.in_index);
    aux = Onnx::planAuxInputs(spec.inputs, owned, {.nparams = 3});
    aux_store.resize(aux.size());
    aux_shapes.resize(aux.size());
  }

  // Vocoder features: the style from the Mel input (Auto: Vocos for 100 bins,
  // the vocos-mel-24khz layout, else HiFi-GAN), then the model's metadata
  // (sherpa-onnx exports carry n_fft / hop_length) over the style's values.
  lastMelStyle = inputs.mel_style.value;
  Onnx::MelConfig melcfg;
  if(mel_input)
  {
    const int n_mels = (int)spec.inputs[wave_in_index].shape[1];
    const bool vocos = lastMelStyle == AudioMelStyle::Vocos
                       || (lastMelStyle == AudioMelStyle::Auto && n_mels == 100);
    melcfg = vocos ? Onnx::MelConfig::vocos(n_mels, 24000.)
                   : Onnx::MelConfig::hifigan(n_mels, 22050.);
    try
    {
      Ort::AllocatorWithDefaultOptions alloc;
      auto meta = ctx->session.GetModelMetadata();
      auto number = [&](const char* key, auto& dst) {
        if(auto v = meta.LookupCustomMetadataMapAllocated(key, alloc))
          if(const double d = std::atof(v.get()); d > 0)
            dst = (std::remove_reference_t<decltype(dst)>)d;
      };
      number("n_fft", melcfg.n_fft);
      number("hop_length", melcfg.hop);
      number("fmin", melcfg.fmin);
      number("fmax", melcfg.fmax);
    }
    catch(...)
    {
    }
  }

  lastRateOverride = inputs.model_rate.value;
  model_rate = lastRateOverride > 0 ? (double)lastRateOverride
               : Onnx::audioModelRate(
                   *ctx, inputs.model.file.filename,
                   mel_input ? melcfg.rate : host_rate);

  const int hc = host_in_channels > 0 ? host_in_channels : 1;
  if(mel_input)
  {
    melcfg.rate = model_rate;
    if(melcfg.fmax > model_rate / 2.)
      melcfg.fmax = model_rate / 2.;
    auto m = std::make_shared<Onnx::MelFrontend>();
    m->prepare(melcfg);
    mel = std::move(m);
    in_shape.channels = 1;
    in_shape.block = mel->blockSize();
    mel_shape = {1, melcfg.n_mels, mel->blockFrames()};
    if(spec_out[0] >= 0)
    {
      auto sy = std::make_shared<Onnx::SpectrumSynth>();
      sy->prepare(melcfg.n_fft, melcfg.hop);
      synth = std::move(sy);
      synth_acc.assign(melcfg.n_fft, 0.f);
      out_shape.channels = 1; // the spectrum is one channel of audio
    }
    else
      synth.reset();
  }
  else
  {
    mel.reset();
    synth.reset();
  }
  const int64_t block = in_shape.block > 0 ? in_shape.block : 1024;
  // Heavy models (separation/vocoder, >32MB or big block) run async; light
  // streaming models (recurrent denoise) run inline for low latency. A
  // vocoder always does: its mel analysis alone is too long for the audio
  // thread.
  async_model = inputs.model.file.bytes.size() > 32u * 1024 * 1024 || block > 48000
                || mel_input;
  const int backlog = async_model ? 4 : 0;
  lastOverlap = inputs.overlap.value;
  const int64_t hop = mel_input ? mel->hopSize()
                      : lastOverlap == AudioOverlap::Half          ? block / 2
                      : lastOverlap == AudioOverlap::ThreeQuarters ? block / 4
                                                                   : block;
  audio_in.prepare(in_shape, host_rate, model_rate, block, hop, hc,
                   max_frames, backlog);
  audio_out.prepare(out_shape.channels, model_rate, host_rate, block,
                    max_frames, backlog);
  audio_out.prepareOverlap(block, mel_input ? block : hop);
  staged.reserve((std::size_t)out_shape.channels * block + 16);
  out_planar.reserve((std::size_t)out_shape.channels * block + 16);
}

void AudioProcessor::zeroStates()
{
  for(auto& s : states)
    std::fill(s.data.begin(), s.data.end(), 0.f);
  std::fill(synth_acc.begin(), synth_acc.end(), 0.f);
  audio_in.reset();
  audio_out.reset();
}

void AudioProcessor::operator()(int frames)
try
{
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
  if(spec.inputs.empty() || spec.outputs.empty())
    return;
  if((inputs.model_rate.value != lastRateOverride
      || inputs.overlap.value != lastOverlap
      || inputs.mel_style.value != lastMelStyle)
     && !inferenceInProgress)
    resolveIO(); // re-prepares the resamplers and the block hop

  if(inputs.reset.value)
  {
    zeroStates();
    ++gen;
    inputs.reset.value.reset();
  }

  // Accumulate host audio into the model-rate rings.
  const int hc = inputs.audio.channels;
  if(hc > 0 && frames > 0)
    audio_in.push(inputs.audio.samples, hc, (std::size_t)frames);

  // Fire the model for every full block currently buffered (bounded loop to
  // avoid runaway if a huge host block arrived).
  // A block that fails is reported and skipped, here rather than in the
  // outer catch so the output below is still pulled. While the same failure
  // repeats, the blocks are dropped instead of run (see FailureLog).
  int guard = 0;
  if(!failures.ready())
  {
    while(audio_in.ready() && guard++ < 32)
      audio_in.fill(staged);
  }
  while(audio_in.ready() && guard++ < 32)
  {
    if(async_model && inferenceInProgress)
      break; // keep the block in the ring until the worker is free
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

  // Drain processed audio back to the host outputs.
  const int oc = outputs.audio.channels;
  if(oc > 0 && frames > 0)
    audio_out.pull(outputs.audio.samples, oc, (std::size_t)frames);
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

void AudioProcessor::runBlock()
{
  const int64_t n = audio_in.fill(staged);
  if(n <= 0)
    return;

  // Param 1 selects which separation stem to route to the output (0..1 mapped
  // across the available waveform outputs).
  if(wave_out_indices.size() > 1)
  {
    const int sel = std::clamp(
        (int)(inputs.param1.value * (float)wave_out_indices.size()), 0,
        (int)wave_out_indices.size() - 1);
    wave_out_index = wave_out_indices[sel];
  }

  dispatchInfer(n, async_model);
}

void AudioProcessor::dispatchInfer(int64_t n, bool force_async)
{
  if(force_async)
  {
    if(inferenceInProgress)
      return;
    inferenceInProgress = true;
    // Pooled job: lock-free acquire; recycled vectors keep their capacity so
    // the assignments below don't allocate in steady state.
    auto job = JobPool<AudioInferJob>::instance().acquire();
    job->ctx = ctx;
    job->input = staged;
    job->ishape = mel_input ? mel_shape : in_shape.tensorShape(n);
    job->mel = mel;
    job->synth = synth;
    job->synth_acc = synth_acc;
    for(int k = 0; k < 3; ++k)
      job->spec_out[k] = spec_out[k];
    if(mel)
    {
      // Only the new frames: the context frames' output is dropped.
      const int64_t h = synth ? 1 : mel->cfg.hop;
      job->keep_from = mel->context * h;
      job->keep_count = mel->frames * h;
    }
    else
      job->keep_count = -1;
    job->wave_in_index = wave_in_index;
    job->wave_out_index = wave_out_index;
    job->out_shape = out_shape;
    job->gen = gen;
    job->stem = inputs.param1.value;
    job->aux = aux;
    job->params[0] = inputs.param2.value;
    job->params[1] = inputs.param3.value;
    job->params[2] = inputs.param4.value;
    job->model_rate = model_rate;
    job->state_in.resize(states.size());
    job->state_in_shape.resize(states.size());
    job->state_in_index.resize(states.size());
    job->state_out_index.resize(states.size());
    for(std::size_t k = 0; k < states.size(); ++k)
    {
      job->state_in[k] = states[k].data;
      job->state_in_shape[k] = states[k].shape;
      job->state_in_index[k] = states[k].in_index;
      job->state_out_index[k] = states[k].out_index;
    }
    worker.request(std::move(job));
    return;
  }

  // --- synchronous inference (build all model inputs incl. state) ----------
  const int nin = (int)spec.inputs.size();
  std::vector<Ort::Value> ins;
  ins.reserve(nin);
  auto ishape = in_shape.tensorShape(n); // vocoders always run async
  const float params[3]{inputs.param2.value, inputs.param3.value, inputs.param4.value};
  const Onnx::AuxHost host{
      .params = params, .sample_rate = model_rate, .primary_shape = ishape};
  for(int i = 0; i < nin; ++i)
  {
    if(i == wave_in_index)
    {
      ins.emplace_back(Onnx::vec_to_tensor<float>(staged, ishape));
      continue;
    }
    StatePort* sp = nullptr;
    for(auto& s : states)
      if(s.in_index == i)
      {
        sp = &s;
        break;
      }
    if(sp)
    {
      ins.emplace_back(Onnx::vec_to_tensor<float>(sp->data, sp->shape));
      continue;
    }
    // The other inputs, by their aux plan, typed as declared.
    std::size_t k = 0;
    while(k < aux.size() && aux[k].index != i)
      ++k;
    if(k < aux.size())
      ins.emplace_back(Onnx::fillAux(aux[k], host, aux_store[k], aux_shapes[k]));
    else
      ins.emplace_back(nullptr);
  }

  const int nout = (int)spec.output_names_char.size();
  std::vector<Ort::Value> outs;
  outs.reserve(nout);
  for(int i = 0; i < nout; ++i)
    outs.emplace_back(nullptr);
  ctx->infer(spec, ins, outs);
  failures.succeeded();

  // Waveform output -> resample -> host ring.
  {
    auto& res = outs[std::clamp(wave_out_index, 0, nout - 1)];
    const auto info = res.GetTensorTypeAndShapeInfo();
    const auto osh = info.GetShape();
    const int64_t cnt = (int64_t)info.GetElementCount();
    const TensorElemType odt = Onnx::fromOrtElementType(info.GetElementType());
    const float* f = Onnx::toFloat(res.GetTensorData<uint8_t>(), cnt, odt,
                                   out_scratch);
    const auto st = selectStem(f, cnt, osh, out_shape.channels, inputs.param1.value);
    out_planar.assign(st.data, st.data + st.channels * st.frames);
    audio_out.pushOverlap(out_planar.data(), st.channels, st.frames);
  }

  // Feed recurrent outputs back into state inputs.
  for(auto& s : states)
  {
    if(s.out_index < 0 || s.out_index >= nout)
      continue;
    auto& res = outs[s.out_index];
    const auto info = res.GetTensorTypeAndShapeInfo();
    const int64_t cnt = (int64_t)info.GetElementCount();
    const TensorElemType odt = Onnx::fromOrtElementType(info.GetElementType());
    const float* f
        = Onnx::toFloat(res.GetTensorData<uint8_t>(), cnt, odt, out_scratch);
    s.data.assign(f, f + cnt);
  }

  // Surface a small per-channel RMS summary of the produced block as Data.
  if(!out_planar.empty())
  {
    const int oc = out_shape.channels > 0 ? out_shape.channels : 1;
    const std::size_t per = out_planar.size() / (std::size_t)oc;
    outputs.data.value.assign((std::size_t)oc, 0.f);
    for(int c = 0; c < oc; ++c)
    {
      double acc = 0.0;
      const float* p = out_planar.data() + (std::size_t)c * per;
      for(std::size_t i = 0; i < per; ++i)
        acc += (double)p[i] * p[i];
      outputs.data.value[c] = per ? (float)std::sqrt(acc / (double)per) : 0.f;
    }
  }
}

std::function<void(AudioProcessor&)>
AudioProcessor::worker::work(std::unique_ptr<AudioInferJob> job)
{
  // RAII: whatever path we exit through, the job goes back to the lock-free
  // pool (with its buffer capacities intact) once the results are moved out.
  struct Recycle
  {
    std::unique_ptr<AudioInferJob>& j;
    ~Recycle()
    {
      if(j)
        j->ctx.reset(); // don't keep the ORT session alive from the pool
      JobPool<AudioInferJob>::instance().release(std::move(j));
    }
  } recycle{job};

  if(!job || !job->ctx)
    return [](AudioProcessor& self) { self.inferenceInProgress = false; };
  try
  {
    const auto& spec = job->ctx->readModelSpec();
    const int nin = (int)spec.inputs.size();
    std::vector<Ort::Value> ins;
    ins.reserve(nin);
    // The job (owned here for the whole call) backs the input/state tensors;
    // the other inputs get their aux plan, each in its own storage.
    std::vector<std::vector<uint8_t>> aux_store(job->aux.size());
    std::vector<std::vector<int64_t>> aux_sh(job->aux.size());
    const Onnx::AuxHost host{
        .params = job->params, .sample_rate = job->model_rate,
        .primary_shape = job->ishape};
    // A vocoder's input: the log-mel of the audio block.
    std::vector<float> features;
    if(job->mel)
    {
      std::vector<float> magnitude;
      features.resize((std::size_t)job->mel->cfg.n_mels * job->mel->blockFrames());
      job->mel->compute(job->input.data(), features.data(), magnitude);
    }
    for(int i = 0; i < nin; ++i)
    {
      if(i == job->wave_in_index)
      {
        ins.emplace_back(Onnx::vec_to_tensor<float>(
            job->mel ? features : job->input, job->ishape));
        continue;
      }
      int si = -1;
      for(int k = 0; k < (int)job->state_in_index.size(); ++k)
        if(job->state_in_index[k] == i)
        {
          si = k;
          break;
        }
      if(si >= 0)
        ins.emplace_back(Onnx::vec_to_tensor<float>(
            job->state_in[si], job->state_in_shape[si]));
      else
      {
        std::size_t k = 0;
        while(k < job->aux.size() && job->aux[k].index != i)
          ++k;
        if(k < job->aux.size())
          ins.emplace_back(Onnx::fillAux(job->aux[k], host, aux_store[k], aux_sh[k]));
        else
          ins.emplace_back(nullptr);
      }
    }

    const int nout = (int)spec.output_names_char.size();
    std::vector<Ort::Value> outs;
    outs.reserve(nout);
    for(int i = 0; i < nout; ++i)
      outs.emplace_back(nullptr);
    job->ctx->infer(spec, ins, outs);

    // Decode waveform output to a planar float buffer on this thread.
    std::vector<float> scratch;
    std::vector<float> planar;
    int oc = 1;
    int64_t on = 0;
    if(job->synth && job->spec_out[0] >= 0)
    {
      // Vocos: magnitude * (cos + i sin), inverse STFT of the new frames.
      std::vector<float> parts[3];
      int64_t T = 0;
      for(int k = 0; k < 3; ++k)
      {
        auto& o = outs[job->spec_out[k]];
        const auto oi = o.GetTensorTypeAndShapeInfo();
        const auto sh = oi.GetShape();
        const int64_t c = (int64_t)oi.GetElementCount();
        const float* p = Onnx::toFloat(
            o.GetTensorData<uint8_t>(), c,
            Onnx::fromOrtElementType(oi.GetElementType()), scratch);
        parts[k].assign(p, p + c);
        if(sh.size() == 3)
          T = sh[2];
      }
      const int F = job->synth->n_fft / 2 + 1;
      if(T > 0 && (int64_t)parts[0].size() == F * T)
      {
        const int64_t t0 = std::min<int64_t>(job->keep_from, T);
        const int64_t t1 = std::min<int64_t>(t0 + job->keep_count, T);
        job->synth->push(
            parts[0].data(), parts[1].data(), parts[2].data(), (int)T, (int)t0,
            (int)t1, job->synth_acc, planar);
      }
      on = (int64_t)planar.size();
    }
    else
    {
      auto& res = outs[std::clamp(job->wave_out_index, 0, nout - 1)];
      const auto info = res.GetTensorTypeAndShapeInfo();
      const auto osh = info.GetShape();
      const int64_t cnt = (int64_t)info.GetElementCount();
      const Onnx::TensorElemType odt
          = Onnx::fromOrtElementType(info.GetElementType());
      const float* f
          = Onnx::toFloat(res.GetTensorData<uint8_t>(), cnt, odt, scratch);
      const auto st = selectStem(f, cnt, osh, job->out_shape.channels, job->stem);
      oc = st.channels;
      on = st.frames;
      if(job->keep_count >= 0 && oc == 1)
      {
        // A vocoder's waveform: the samples of the new frames only.
        const int64_t from = std::min<int64_t>(job->keep_from, on);
        on = std::min<int64_t>(job->keep_count, on - from);
        planar.assign(st.data + from, st.data + from + on);
      }
      else
        planar.assign(st.data, st.data + st.channels * st.frames);
    }

    // Capture new state values.
    std::vector<std::pair<int, std::vector<float>>> new_states;
    for(int k = 0; k < (int)job->state_out_index.size(); ++k)
    {
      const int o = job->state_out_index[k];
      if(o < 0 || o >= nout)
        continue;
      const auto si = outs[o].GetTensorTypeAndShapeInfo();
      const int64_t sc = (int64_t)si.GetElementCount();
      const Onnx::TensorElemType sdt
          = Onnx::fromOrtElementType(si.GetElementType());
      std::vector<float> st;
      const float* sf
          = Onnx::toFloat(outs[o].GetTensorData<uint8_t>(), sc, sdt, scratch);
      st.assign(sf, sf + sc);
      new_states.emplace_back(job->state_in_index[k], std::move(st));
    }

    return [planar = std::move(planar), oc, on, gen = job->gen,
            new_states = std::move(new_states),
            acc = std::move(job->synth_acc)](AudioProcessor& self) mutable
    {
      self.inferenceInProgress = false;
      self.failures.succeeded();
      if(gen != self.gen)
        return; // Reset while it ran: keep the zeroed states, drop the block
      if(acc.size() == self.synth_acc.size())
        self.synth_acc = std::move(acc);
      if(on > 0)
        self.audio_out.pushOverlap(planar.data(), oc, on);
      for(auto& ns : new_states)
        for(auto& s : self.states)
          // Size guard: if the model was swapped while this job was in flight,
          // the result carries the OLD model's state shapes. Adopting them would
          // feed the new model a wrong-sized tensor and brick it. Same check as
          // SequenceProcessor's state feedback.
          if(s.in_index == ns.first && s.data.size() == ns.second.size())
            s.data = std::move(ns.second);
    };
  }
  catch(const std::exception& e)
  {
    return [what = std::string(e.what())](AudioProcessor& self)
    {
      self.inferenceInProgress = false;
      self.failures.failed(AudioProcessor::name(), self.inputs.model.file.filename, what);
    };
  }
  catch(...)
  {
    return [](AudioProcessor& self)
    {
      self.inferenceInProgress = false;
      self.failures.failed(AudioProcessor::name(), self.inputs.model.file.filename, "unknown error");
    };
  }
}

}
