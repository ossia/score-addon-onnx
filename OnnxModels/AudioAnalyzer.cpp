#include "AudioAnalyzer.hpp"

#include <Onnx/helpers/AudioRate.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/TensorToTexture.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdio>
#include <cmath>

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

AudioAnalyzer::AudioAnalyzer() noexcept
{
  staged.reserve(48000);
}

AudioAnalyzer::~AudioAnalyzer() = default;

void AudioAnalyzer::prepare(halp::setup info)
{
  host_rate = info.rate > 0 ? info.rate : 48000.0;
  host_in_channels = info.input_channels;
  max_frames = info.frames > 0 ? (std::size_t)info.frames : 4096;
  lastModelPath.clear();
  ctx.reset();
}

void AudioAnalyzer::reloadModel()
{
  ctx = std::make_shared<Onnx::OnnxRunContext>(
      inputs.model.file.bytes, inputs.model.file.filename);
  spec = ctx->readModelSpec();
  arch = Onnx::classifyModel(toArchIO(spec));
  lastModelPath = inputs.model.file.filename;
  resolveIO();
}

void AudioAnalyzer::resolveIO()
{
  wave_in_index = -1;
  states.clear();

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

  for(int i = 0; i < (int)arch.inputs.size(); ++i)
  {
    if(arch.inputs[i].arch != PortArchetype::RecurrentState)
      continue;
    StatePort sp;
    sp.in_index = i;
    sp.shape = spec.inputs[i].shape;
    for(auto& d : sp.shape)
      if(d < 0)
        d = 1;
    // Paired once by classifyModel (names first, then free same-shape outputs).
    sp.out_index = arch.inputs[i].state_pair;
    sp.data.assign((std::size_t)flatPositive(sp.shape), 0.f);
    states.push_back(std::move(sp));
  }

  std::vector<int> owned{wave_in_index};
  for(const auto& s : states)
    owned.push_back(s.in_index);
  aux = Onnx::planAuxInputs(spec.inputs, owned, {.nparams = 1});
  aux_store.resize(aux.size());
  aux_shapes.resize(aux.size());

  // WaveformShape has no rank-4 layout: CLAP's mel_fusion [1,4,1001,64] would
  // be fed as [1,1,64] and fail on every block.
  unsupported_input = spec.inputs[wave_in_index].shape.size() >= 4;
  if(unsupported_input)
    std::fprintf(
        stderr,
        "Audio Analyzer: %s takes a rank-4 input (%s), which needs a spectrogram "
        "frontend; not supported\n",
        std::string(inputs.model.file.filename).c_str(),
        spec.inputs[wave_in_index].name.c_str());

  in_shape
      = Onnx::WaveformShape::fromInputShape(spec.inputs[wave_in_index].shape);
  lastRateOverride = inputs.model_rate.value;
  model_rate = lastRateOverride > 0
                   ? (double)lastRateOverride
                   : Onnx::audioModelRate(*ctx, inputs.model.file.filename, 16000.);
  normalize_frames = isCrepe(spec);

  const int hc = host_in_channels > 0 ? host_in_channels : 1;
  const int64_t block = in_shape.block > 0 ? in_shape.block : 1024;
  audio_in.prepare(in_shape, host_rate, model_rate, block, block, hc,
                   max_frames);
  staged.reserve((std::size_t)in_shape.channels * block + 16);
}

void AudioAnalyzer::zeroStates()
{
  for(auto& s : states)
    std::fill(s.data.begin(), s.data.end(), 0.f);
  audio_in.reset();
}

void AudioAnalyzer::operator()(int frames)
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
  if(spec.inputs.empty() || spec.outputs.empty() || unsupported_input)
    return;
  if(inputs.model_rate.value != lastRateOverride)
    resolveIO(); // re-prepares the resampler for the new rate

  if(inputs.reset.value)
  {
    zeroStates();
    inputs.reset.value.reset();
  }

  const int hc = inputs.audio.channels;
  if(hc > 0 && frames > 0)
    audio_in.push(inputs.audio.samples, hc, (std::size_t)frames);

  int guard = 0;
  while(audio_in.ready() && guard++ < 32)
    runBlock();
}
catch(...)
{
  inputs.model.current_model_invalid = true;
}

void AudioAnalyzer::runBlock()
{
  const int64_t n = audio_in.fill(staged);
  if(n <= 0)
    return;
  if(normalize_frames)
    normalizeFrames(staged, in_shape.channels);

  const int nin = (int)spec.inputs.size();
  std::vector<Ort::Value> ins;
  ins.reserve(nin);
  auto ishape = in_shape.tensorShape(n);
  const float param = inputs.param.value;
  const Onnx::AuxHost host{
      .params = {&param, 1}, .sample_rate = model_rate, .primary_shape = ishape};
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
    }
    else
    {
      // The other inputs, by their aux plan: Silero's sr gets the model rate
      // (as its int64), a bool flag false, a scalar the Param.
      std::size_t k = 0;
      while(k < aux.size() && aux[k].index != i)
        ++k;
      if(k < aux.size())
        ins.emplace_back(Onnx::fillAux(aux[k], host, aux_store[k], aux_shapes[k]));
      else
        ins.emplace_back(nullptr);
    }
  }

  const int nout = (int)spec.output_names_char.size();
  std::vector<Ort::Value> outs;
  outs.reserve(nout);
  for(int i = 0; i < nout; ++i)
    outs.emplace_back(nullptr);
  ctx->infer(spec, ins, outs);

  // Choose the primary (non-state) output: the first output that is not paired
  // with a recurrent state input.
  int primary = 0;
  for(int o = 0; o < nout; ++o)
  {
    bool is_state = false;
    for(auto& s : states)
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
      = Onnx::toFloat(res.GetTensorData<uint8_t>(), cnt, odt, out_scratch);
  std::vector<float> flat(f, f + cnt);

  // Feed recurrent state back.
  for(auto& s : states)
  {
    if(s.out_index < 0 || s.out_index >= nout)
      continue;
    auto& sres = outs[s.out_index];
    const auto sinfo = sres.GetTensorTypeAndShapeInfo();
    const int64_t scnt = (int64_t)sinfo.GetElementCount();
    const TensorElemType sdt
        = Onnx::fromOrtElementType(sinfo.GetElementType());
    const float* sf
        = Onnx::toFloat(sres.GetTensorData<uint8_t>(), scnt, sdt, out_scratch);
    s.data.assign(sf, sf + scnt);
  }

  reduceOutputs(flat, primary);
}

void AudioAnalyzer::reduceOutputs(const std::vector<float>& flat, int primary)
{
  outputs.data.value = flat;
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
