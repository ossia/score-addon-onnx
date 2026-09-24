#include "TextToken.hpp"

#include <OnnxModels/JobPool.hpp>

#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/TensorToTexture.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <span>
#include <utility>

namespace OnnxModels
{
using Onnx::AuxRole;
using Onnx::PortArchetype;
using Onnx::TensorElemType;
using Onnx::TokenOutputRole;

namespace
{
// Build the dependency-free ArchIO view classifyModel / TokenIO want.
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

// Resolve a model input's declared shape to concrete dims (dynamic -> 1).
std::vector<int64_t> resolveShape(std::vector<int64_t> s)
{
  for(auto& d : s)
    if(d < 0)
      d = 1;
  if(s.empty())
    s = {1};
  return s;
}

// Native sample-rate guess from the port names, for models that say nothing
// else. VITS/Piper are typically 22050; some multi-speaker / high-quality
// exports are 16k / 44.1k / 48k.
double guessTtsRate(const Onnx::ModelArchetype& a)
{
  auto named = [&](std::initializer_list<const char*> ks)
  {
    for(const auto& p : a.inputs)
      for(auto k : ks)
        if(Onnx::detail::nameContains(p.name, k))
          return true;
    for(const auto& p : a.outputs)
      for(auto k : ks)
        if(Onnx::detail::nameContains(p.name, k))
          return true;
    return false;
  };
  if(named({"16khz", "16000"}))
    return 16000.0;
  if(named({"44100", "44k"}))
    return 44100.0;
  if(named({"48000", "48k"}))
    return 48000.0;
  return 22050.0; // VITS / Piper default
}

double parseRate(const std::string& s)
{
  try
  {
    const double r = std::stod(s);
    return (r >= 1000. && r <= 384000.) ? r : 0.;
  }
  catch(...)
  {
    return 0.;
  }
}

// The model's own rate: the "sample_rate" metadata property (sherpa-onnx
// Piper, Kitten, Kokoro, Matcha exports carry it), else audio.sample_rate in
// the <model>.onnx.json sidecar of the upstream Piper voices, else the name
// guess. The port names of these models never carry the rate.
double ttsRate(Ort::Session& session, std::string_view model_path,
               const Onnx::ModelArchetype& a)
{
  try
  {
    Ort::AllocatorWithDefaultOptions alloc;
    auto meta = session.GetModelMetadata();
    if(auto v = meta.LookupCustomMetadataMapAllocated("sample_rate", alloc))
      if(const double r = parseRate(v.get()); r > 0.)
        return r;
  }
  catch(...)
  {
  }

  if(!model_path.empty())
  {
    QFile f(QString::fromUtf8(model_path.data(), (qsizetype)model_path.size()) + ".json");
    if(f.open(QIODevice::ReadOnly))
    {
      const auto doc = QJsonDocument::fromJson(f.readAll());
      const double r
          = doc.object().value("audio").toObject().value("sample_rate").toDouble();
      if(r >= 1000. && r <= 384000.)
        return r;
    }
  }

  return guessTtsRate(a);
}

bool isIntDtype(TensorElemType t) noexcept
{
  return Onnx::detail::isIntType(t);
}

// Aux inputs with one value per token: fed at the token tensor's shape.
bool tokenShaped(AuxRole r) noexcept
{
  return r == AuxRole::Mask || r == AuxRole::TokenTypes;
}

// Resample a planar [channel][n] utterance to the host rate in one go (each
// utterance is independent, so no resampler state carries over).
TtsUtterance toHostRate(
    const float* planar, int channels, int64_t n, double model_rate, double host_rate)
{
  TtsUtterance u;
  u.channels = std::max(channels, 1);
  std::vector<std::vector<float>> chans(u.channels);
  for(int c = 0; c < u.channels && n > 0; ++c)
  {
    Onnx::ResamplerLin rs;
    rs.prepare(model_rate, host_rate);
    chans[c].reserve((std::size_t)(n * rs.ratio) + 4);
    rs.process(planar + (std::size_t)c * n, (std::size_t)n, chans[c]);
  }
  u.frames = chans[0].size();
  for(auto& ch : chans)
    u.frames = std::min(u.frames, ch.size());
  u.samples.resize(u.frames * u.channels);
  for(int c = 0; c < u.channels; ++c)
    std::copy_n(chans[c].begin(), u.frames, u.samples.begin() + c * u.frames);
  return u;
}

// Token ids and int aux values are staged as int64; the tensor must carry the
// model's declared element type, as ORT rejects any other ("Unexpected input
// data type"): sherpa punctuation and ct-transformer take int32, Tacotron2's
// decoder a bool mask. `narrow` backs a converted copy and must outlive the
// returned tensor; int64 inputs use `src` directly.
Ort::Value intTensor(
    TensorElemType dt, std::vector<int64_t>& src, const std::vector<int64_t>& shape,
    std::vector<uint8_t>& narrow)
{
  auto convert = [&]<typename T>(ONNXTensorElementDataType ort, bool as_bool) {
    narrow.resize(src.size() * sizeof(T));
    auto* d = reinterpret_cast<T*>(narrow.data());
    for(std::size_t i = 0; i < src.size(); ++i)
      d[i] = as_bool ? T(src[i] != 0) : T(src[i]);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor(
        mem, narrow.data(), narrow.size(), shape.data(), shape.size(), ort);
  };
  switch(dt)
  {
    case TensorElemType::Int32:
      return convert.operator()<int32_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, false);
    case TensorElemType::Uint32:
      return convert.operator()<uint32_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, false);
    case TensorElemType::Int16:
      return convert.operator()<int16_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, false);
    case TensorElemType::Uint16:
      return convert.operator()<uint16_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, false);
    case TensorElemType::Int8:
      return convert.operator()<int8_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, false);
    case TensorElemType::Uint8:
      return convert.operator()<uint8_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, false);
    case TensorElemType::Bool:
      return convert.operator()<uint8_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, true);
    case TensorElemType::Uint64:
      return convert.operator()<uint64_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, false);
    default:
      return Onnx::vec_to_tensor<int64_t>(src, shape);
  }
}
} // namespace

TextToken::TextToken() noexcept
{
  token_buf.reserve(512);
}

TextToken::~TextToken() = default;

void TextToken::prepare(halp::setup info)
{
  host_rate = info.rate > 0 ? info.rate : 48000.0;
  max_frames = info.frames > 0 ? (std::size_t)info.frames : 4096;
  // Force re-resolution of the audio pipeline against the new host config.
  lastModelPath.clear();
  ctx.reset();
}

void TextToken::reloadModel()
{
  ctx = std::make_shared<Onnx::OnnxRunContext>(
      inputs.model.file.bytes, inputs.model.file.filename);
  spec = ctx->readModelSpec();
  const auto io = toArchIO(spec);
  arch = Onnx::classifyModel(io);

  // REFUSE autoregressive decode loops (KV-cache / past_* inputs, or a state
  // carried from an output back to an input: Tacotron2's decoder_iter,
  // pocket-tts): a single forward cannot drive their per-token decode.
  refused = Onnx::isAutoregressive(io) || arch.stateful;

  lastModelPath = inputs.model.file.filename;
  last_tokens.clear();
  pending = false;
  ++gen; // a job still running on the previous model is dropped
  utterance.pos = utterance.frames;
  if(!refused)
    resolveIO();

  // A style-conditioned TTS reads its voices; which shape they have is in the
  // metadata (style_dim "1,256" or "511,1,256").
  style_rows = 1;
  style_dim = 0;
  sibling_voices.clear();
  for(const auto& p : aux)
    if(p.role == AuxRole::Style)
      style_dim = (int)flatPositive(p.shape);
  if(style_dim > 0)
  {
    try
    {
      Ort::AllocatorWithDefaultOptions alloc;
      auto meta = ctx->session.GetModelMetadata();
      if(auto v = meta.LookupCustomMetadataMapAllocated("style_dim", alloc))
      {
        std::vector<int> dims;
        for(const char* c = v.get(); *c;)
        {
          char* end{};
          dims.push_back((int)std::strtol(c, &end, 10));
          c = (*end == ',') ? end + 1 : end;
          if(end == c && *c)
            break;
        }
        if(dims.size() == 3 && dims[0] > 0)
          style_rows = dims[0];
      }
    }
    catch(...)
    {
    }
    const std::string path{inputs.model.file.filename};
    const auto slash = path.find_last_of("/\\");
    std::ifstream f(
        (slash == std::string::npos ? std::string{} : path.substr(0, slash + 1))
            + "voices.bin",
        std::ios::binary);
    if(f)
    {
      const std::string raw{std::istreambuf_iterator<char>(f), {}};
      sibling_voices.resize(raw.size() / sizeof(float));
      std::memcpy(sibling_voices.data(), raw.data(), sibling_voices.size() * sizeof(float));
    }
  }

  // Nor a model with an input the node cannot synthesise: it only makes
  // scalars, small vectors and token-shaped masks, not a [.,.,288] encoder
  // output (moonshine) or Tacotron's memory.
  if(!refused)
    for(int i = 0; i < (int)spec.inputs.size(); ++i)
      if(i != token_index && spec.inputs[i].shape.size() >= 3)
      {
        refused = true;
        break;
      }
  if(refused)
    std::fprintf(
        stderr,
        "Text Token Processor: %s is an autoregressive decoder or needs inputs "
        "this node cannot build; not run\n",
        std::string(inputs.model.file.filename).c_str());
}

// Find the token input, the output to route (waveform -> audio, else -> data),
// classify every non-token input into an aux plan, and map the 4 Params onto
// the scale / speaker / length aux roles.
void TextToken::resolveIO()
{
  token_index = -1;
  wave_out_index = -1;
  aux.clear();

  // Primary token input: prefer a TokenSeq archetype, else the first int input,
  // else input 0.
  for(int i = 0; i < (int)arch.inputs.size(); ++i)
  {
    if(arch.inputs[i].arch == PortArchetype::TokenSeq)
    {
      token_index = i;
      break;
    }
  }
  // Fallback: the first int input with rank<=2 that is NOT a known scalar aux
  // (lengths / sid). This catches dynamic-length token inputs ([1,-1]) that
  // classifyPort tags Latent because their flat size collapses to 1 — a real
  // and common VITS/Piper export (see docs/texttoken-PLAN.md, classifier gap).
  if(token_index < 0)
  {
    for(int i = 0; i < (int)spec.inputs.size(); ++i)
    {
      // A bool tensor is a mask, never the ids.
      if(!isIntDtype(spec.inputs[i].elem_type)
         || spec.inputs[i].elem_type == TensorElemType::Bool)
        continue;
      if(spec.inputs[i].shape.size() > 2)
        continue;
      const auto r = Onnx::classifyAux(
          spec.inputs[i].name, spec.inputs[i].shape, spec.inputs[i].elem_type);
      if(r == AuxRole::InputLength || r == AuxRole::SpeakerId
         || r == AuxRole::Mask || r == AuxRole::TokenTypes)
        continue; // these are aux inputs, not the token sequence
      token_index = i;
      break;
    }
  }
  if(token_index < 0)
    token_index = 0;

  token_in = Onnx::TokenInput::fromInputShape(spec.inputs[token_index].shape);

  // Aux plans for every other input.
  int next_param = 0; // Param 1..4 round-robin for unmapped scale-like roles
  auto takeParam = [&]() -> int
  { return (next_param < 4) ? next_param++ : -1; };

  for(int i = 0; i < (int)spec.inputs.size(); ++i)
  {
    if(i == token_index)
      continue;
    TokenAuxPlan p;
    p.model_index = i;
    p.dtype = spec.inputs[i].elem_type;
    p.shape = resolveShape(spec.inputs[i].shape);
    p.role = Onnx::classifyAux(
        spec.inputs[i].name, spec.inputs[i].shape, p.dtype);

    switch(p.role)
    {
      case AuxRole::InputLength:
        p.param_index = -1; // filled from the token length, not a Param
        break;
      case AuxRole::NoiseScale:
        p.param_index = 0;
        p.default_value = 0.667f;
        break;
      case AuxRole::LengthScale:
        p.param_index = 1;
        p.default_value = 1.0f;
        break;
      case AuxRole::NoiseScaleW:
        p.param_index = 2;
        p.default_value = 0.8f;
        break;
      case AuxRole::Scales:
        // Packed [noise, length, noise_w] vector: filled directly from Params
        // 1..3 in run(); no single param_index.
        p.param_index = -1;
        break;
      case AuxRole::SpeakerId:
        p.param_index = 3;
        p.default_value = 0.f;
        break;
      case AuxRole::GenericFloat:
        p.param_index = takeParam();
        break;
      case AuxRole::GenericInt:
        p.param_index = takeParam();
        break;
      default:
        p.param_index = -1;
        break;
    }
    aux.push_back(std::move(p));
  }

  // Output routing: honour an explicit task override; else classify output 0.
  produces_audio = false;
  wave_out_index = 0;
  for(int o = 0; o < (int)spec.outputs.size(); ++o)
  {
    const auto r = Onnx::classifyTokenOutput(
        spec.outputs[o].name, spec.outputs[o].shape, spec.outputs[o].elem_type);
    if(r == TokenOutputRole::Waveform)
    {
      produces_audio = true;
      wave_out_index = o;
      break;
    }
  }

  if(produces_audio)
  {
    out_shape
        = Onnx::WaveformShape::fromInputShape(spec.outputs[wave_out_index].shape);
    if(out_shape.channels < 1)
      out_shape.channels = 1;
    model_rate = ttsRate(ctx->session, inputs.model.file.filename, arch);
  }
}

float TextToken::paramValue(int idx) const
{
  switch(idx)
  {
    case 0: return inputs.param1.value;
    case 1: return inputs.param2.value;
    case 2: return inputs.param3.value;
    case 3: return inputs.param4.value;
    default: return 0.f;
  }
}

void TextToken::operator()(int frames)
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
  if(refused) // autoregressive: documented no-op
    return;
  if(spec.inputs.empty() || spec.outputs.empty())
    return;

  // Reset stops the utterance and runs the model again on the current ids.
  const auto& ids = inputs.tokens.value;
  if(inputs.reset.value)
  {
    utterance.pos = utterance.frames;
    pending = true;
    ++gen;
    inputs.reset.value.reset();
  }
  else if(!ids.empty() && ids != last_tokens)
  {
    pending = true;
    ++gen;
  }

  if(pending && !ids.empty() && !inferenceInProgress)
  {
    const int64_t explicit_len
        = (inputs.length.value >= 0) ? (int64_t)inputs.length.value : -1;
    const int64_t L = Onnx::buildTokenTensor(
        ids.data(), ids.size(), token_in, token_buf, explicit_len);
    last_tokens = ids;
    pending = false;

    // A style-conditioned TTS without voices would only make noise.
    if(style_dim > 0 && !styleFor((int64_t)ids.size()))
    {
      std::fprintf(
          stderr, "Text Token Processor: %s needs a voices file (Voices port)\n",
          std::string(inputs.model.file.filename).c_str());
      return;
    }

    // Heavy TTS runs async; lightweight text encoders run inline.
    const bool heavy = produces_audio
                       || inputs.model.file.bytes.size() > 32u * 1024 * 1024;
    dispatchInfer(L, heavy);
  }

  if(produces_audio)
    playUtterance(frames);
}
catch(...)
{
  inputs.model.current_model_invalid = true;
}

// The style vector of the voice Param 4 picks, from the Voices port or a
// voices.bin next to the model: `style_rows` rows of `style_dim` floats per
// voice; with several rows (Kokoro) the row follows the token count.
bool TextToken::styleFor(int64_t token_count)
{
  std::span<const float> voices;
  const auto& file = inputs.voices.file.bytes;
  if(file.size() >= sizeof(float))
    voices = {reinterpret_cast<const float*>(file.data()), file.size() / sizeof(float)};
  else
    voices = sibling_voices;
  const std::size_t per_voice = (std::size_t)style_rows * style_dim;
  if(style_dim <= 0 || voices.size() < per_voice)
    return false;
  const int nvoices = (int)(voices.size() / per_voice);
  const int v = std::clamp((int)std::lround(inputs.param4.value), 0, nvoices - 1);
  const int r = style_rows > 1 ? (int)std::min<int64_t>(token_count, style_rows - 1) : 0;
  const float* row = voices.data() + (std::size_t)v * per_voice + (std::size_t)r * style_dim;
  style_buf.assign(row, row + style_dim);
  return true;
}

// Copy the next frames of the utterance to the audio outlet, silence after it.
void TextToken::playUtterance(int frames)
{
  const int oc = outputs.audio.channels;
  if(oc <= 0 || frames <= 0)
    return;
  const std::size_t n = std::min<std::size_t>(
      (std::size_t)frames, utterance.frames - std::min(utterance.pos, utterance.frames));
  for(int c = 0; c < oc; ++c)
  {
    float* out = outputs.audio.samples[c];
    if(n > 0)
    {
      const int src = std::min(c, utterance.channels - 1);
      std::copy_n(
          utterance.samples.data() + src * utterance.frames + utterance.pos, n, out);
    }
    std::fill(out + n, out + frames, 0.f);
  }
  utterance.pos += n;
}

void TextToken::dispatchInfer(int64_t token_len, bool force_async)
{
  if(force_async)
  {
    if(inferenceInProgress)
      return;
    inferenceInProgress = true;
    // Pooled job: lock-free acquire; recycled vectors keep their capacity so
    // the assignments below don't allocate in steady state.
    auto job = JobPool<TokenInferJob>::instance().acquire();
    job->ctx = ctx;
    job->tokens = token_buf;
    job->token_shape = token_in.tensorShape((int64_t)token_buf.size());
    job->token_index = token_index;
    job->token_len = token_len;
    job->aux = aux;
    // The job is recycled: size (not push_back onto) the stale vector.
    job->aux_values.resize(aux.size());
    for(std::size_t k = 0; k < aux.size(); ++k)
      job->aux_values[k] = aux[k].param_index >= 0
                               ? paramValue(aux[k].param_index)
                               : aux[k].default_value;
    job->wave_out_index = wave_out_index;
    job->data_out_index = 0;
    job->produces_audio = produces_audio;
    // Snapshot all three VITS scales so the async Scales path doesn't drop
    // Params 2/3 (the sync path fills them; TTS is always async, so without
    // this the packed-scales models would run with length/noise_w hardcoded).
    job->scales[0] = inputs.param1.value;
    job->scales[1] = inputs.param2.value;
    job->scales[2] = inputs.param3.value;
    job->out_shape = out_shape;
    job->model_rate = model_rate;
    job->host_rate = host_rate;
    job->gen = gen;
    job->style.assign(style_buf.begin(), style_buf.end());
    worker.request(std::move(job));
    return;
  }

  // --- synchronous inference (build all model inputs incl. aux scalars) -----
  const int nin = (int)spec.inputs.size();
  std::vector<Ort::Value> ins;
  ins.reserve(nin);
  aux_int_bufs.assign(nin, {});
  aux_flt_bufs.assign(nin, {});
  int_narrow_bufs.resize(nin);

  auto token_shape = token_in.tensorShape((int64_t)token_buf.size());

  for(int i = 0; i < nin; ++i)
  {
    if(i == token_index)
    {
      ins.emplace_back(intTensor(
          spec.inputs[i].elem_type, token_buf, token_shape, int_narrow_bufs[i]));
      continue;
    }
    // Find the aux plan for this input.
    const TokenAuxPlan* plan = nullptr;
    for(auto& p : aux)
      if(p.model_index == i)
      {
        plan = &p;
        break;
      }
    std::vector<int64_t> sh = plan ? plan->shape : resolveShape(spec.inputs[i].shape);
    if(plan && tokenShaped(plan->role) && sh.size() == token_shape.size())
      sh = token_shape;
    const TensorElemType dt = spec.inputs[i].elem_type;
    const int64_t cnt = flatPositive(sh);

    if(isIntDtype(dt))
    {
      auto& buf = aux_int_bufs[i];
      buf.assign((std::size_t)cnt, 0);
      if(plan && plan->role == AuxRole::Mask)
        std::fill(buf.begin(), buf.end(), 1);
      else if(plan && plan->role == AuxRole::TokenTypes)
        ; // zeros
      else if(plan && plan->role == AuxRole::InputLength)
        std::fill(buf.begin(), buf.end(), token_len);
      else if(plan && plan->param_index >= 0)
        buf[0] = (int64_t)std::lround(paramValue(plan->param_index));
      else if(plan)
        buf[0] = (int64_t)std::lround(plan->default_value);
      ins.emplace_back(intTensor(dt, buf, sh, int_narrow_bufs[i]));
    }
    else
    {
      auto& buf = aux_flt_bufs[i];
      buf.assign((std::size_t)cnt, 0.f);
      if(plan && plan->role == AuxRole::Scales)
      {
        // Packed [noise, length, noise_w] from Params 1..3.
        if(cnt >= 1) buf[0] = inputs.param1.value;
        if(cnt >= 2) buf[1] = inputs.param2.value;
        if(cnt >= 3) buf[2] = inputs.param3.value;
      }
      else if(plan && plan->role == AuxRole::Style)
      {
        std::copy_n(style_buf.begin(), std::min(style_buf.size(), buf.size()), buf.begin());
      }
      else if(plan && plan->param_index >= 0)
      {
        std::fill(buf.begin(), buf.end(), paramValue(plan->param_index));
      }
      else if(plan)
      {
        std::fill(buf.begin(), buf.end(), plan->default_value);
      }
      ins.emplace_back(Onnx::vec_to_tensor<float>(buf, sh));
    }
  }

  const int nout = (int)spec.output_names_char.size();
  std::vector<Ort::Value> outs;
  outs.reserve(nout);
  for(int i = 0; i < nout; ++i)
    outs.emplace_back(nullptr);
  ctx->infer(spec, ins, outs);

  // Route the chosen output.
  const int idx = std::clamp(produces_audio ? wave_out_index : 0, 0, nout - 1);
  auto& res = outs[idx];
  const auto info = res.GetTensorTypeAndShapeInfo();
  const auto osh = info.GetShape();
  const int64_t ocnt = (int64_t)info.GetElementCount();
  const TensorElemType odt = Onnx::fromOrtElementType(info.GetElementType());
  const float* f
      = Onnx::toFloat(res.GetTensorData<uint8_t>(), ocnt, odt, out_scratch);

  if(produces_audio)
  {
    const Onnx::WaveformShape os = Onnx::WaveformShape::fromInputShape(osh);
    const int oc = os.channels > 0 ? os.channels : out_shape.channels;
    const int64_t on = (oc > 0) ? ocnt / oc : ocnt;
    utterance = toHostRate(f, oc, on, model_rate, host_rate);
  }
  else
  {
    outputs.data.value.assign(f, f + ocnt);
  }
}

std::function<void(TextToken&)>
TextToken::worker::work(std::unique_ptr<TokenInferJob> job)
{
  // RAII: whatever path we exit through, the job goes back to the lock-free
  // pool (with its buffer capacities intact) once the results are moved out.
  struct Recycle
  {
    std::unique_ptr<TokenInferJob>& j;
    ~Recycle()
    {
      if(j)
        j->ctx.reset(); // don't keep the ORT session alive from the pool
      JobPool<TokenInferJob>::instance().release(std::move(j));
    }
  } recycle{job};

  if(!job || !job->ctx)
    return [](TextToken& self) { self.inferenceInProgress = false; };
  try
  {
    const auto& spec = job->ctx->readModelSpec();
    const int nin = (int)spec.inputs.size();
    std::vector<Ort::Value> ins;
    ins.reserve(nin);

    // The job (owned here for the whole call) backs the token tensor; the
    // per-input bufs keep every other backing store alive through infer().
    std::vector<std::vector<int64_t>> int_bufs(nin);
    std::vector<std::vector<float>> flt_bufs(nin);
    std::vector<std::vector<uint8_t>> narrow_bufs(nin);

    auto flatPos = [](const std::vector<int64_t>& s)
    {
      int64_t p = 1;
      for(auto d : s)
        p *= (d > 0 ? d : 1);
      return p;
    };

    for(int i = 0; i < nin; ++i)
    {
      if(i == job->token_index)
      {
        ins.emplace_back(intTensor(
            spec.inputs[i].elem_type, job->tokens, job->token_shape, narrow_bufs[i]));
        continue;
      }
      const TokenAuxPlan* plan = nullptr;
      float pval = 0.f;
      for(std::size_t k = 0; k < job->aux.size(); ++k)
        if(job->aux[k].model_index == i)
        {
          plan = &job->aux[k];
          pval = job->aux_values[k];
          break;
        }
      std::vector<int64_t> sh
          = plan ? plan->shape : std::vector<int64_t>{1};
      for(auto& d : sh)
        if(d < 0)
          d = 1;
      if(sh.empty())
        sh = {1};
      if(plan && tokenShaped(plan->role) && sh.size() == job->token_shape.size())
        sh = job->token_shape;
      const Onnx::TensorElemType dt = spec.inputs[i].elem_type;
      const int64_t cnt = flatPos(sh);

      if(Onnx::detail::isIntType(dt))
      {
        auto& buf = int_bufs[i];
        buf.assign((std::size_t)cnt, 0);
        if(plan && plan->role == Onnx::AuxRole::Mask)
          std::fill(buf.begin(), buf.end(), 1);
        else if(plan && plan->role == Onnx::AuxRole::TokenTypes)
          ; // zeros
        else if(plan && plan->role == Onnx::AuxRole::InputLength)
          std::fill(buf.begin(), buf.end(), job->token_len);
        else if(plan)
          buf[0] = (int64_t)std::lround(pval);
        ins.emplace_back(intTensor(dt, buf, sh, narrow_bufs[i]));
      }
      else
      {
        auto& buf = flt_bufs[i];
        buf.assign((std::size_t)cnt, 0.f);
        if(plan && plan->role == Onnx::AuxRole::Scales)
        {
          // Packed [noise, length, noise_w] from the snapshot of Params 1..3
          // (matches the sync path; no longer drops Params 2/3).
          if(cnt >= 1) buf[0] = job->scales[0];
          if(cnt >= 2) buf[1] = job->scales[1];
          if(cnt >= 3) buf[2] = job->scales[2];
        }
        else if(plan && plan->role == Onnx::AuxRole::Style)
        {
          std::copy_n(
              job->style.begin(), std::min(job->style.size(), buf.size()), buf.begin());
        }
        else if(plan)
        {
          std::fill(buf.begin(), buf.end(), pval);
        }
        ins.emplace_back(Onnx::vec_to_tensor<float>(buf, sh));
      }
    }

    const int nout = (int)spec.output_names_char.size();
    std::vector<Ort::Value> outs;
    outs.reserve(nout);
    for(int i = 0; i < nout; ++i)
      outs.emplace_back(nullptr);
    job->ctx->infer(spec, ins, outs);

    std::vector<float> scratch;
    const int out_idx = job->produces_audio ? job->wave_out_index
                                            : job->data_out_index;
    auto& res = outs[std::clamp(out_idx, 0, nout - 1)];
    const auto info = res.GetTensorTypeAndShapeInfo();
    const auto osh = info.GetShape();
    const int64_t cnt = (int64_t)info.GetElementCount();
    const Onnx::TensorElemType odt
        = Onnx::fromOrtElementType(info.GetElementType());
    const float* f
        = Onnx::toFloat(res.GetTensorData<uint8_t>(), cnt, odt, scratch);
    std::vector<float> planar(f, f + cnt);

    // A heavy text encoder (>32 MB CLIP/SigLIP) is async but has no audio out;
    // its result goes to the Data outlet.
    if(!job->produces_audio)
    {
      return [planar = std::move(planar), gen = job->gen](TextToken& self) mutable
      {
        self.inferenceInProgress = false;
        if(gen == self.gen) // else superseded while running
          self.outputs.data.value.assign(planar.begin(), planar.end());
      };
    }

    const Onnx::WaveformShape os = Onnx::WaveformShape::fromInputShape(osh);
    const int oc = os.channels > 0 ? os.channels : 1;
    const int64_t on = (oc > 0) ? cnt / oc : cnt;
    auto u = toHostRate(planar.data(), oc, on, job->model_rate, job->host_rate);

    // The new utterance replaces the playing one; the old buffer leaves with
    // this lambda, so the processing thread only swaps.
    return [u = std::move(u), gen = job->gen](TextToken& self) mutable
    {
      self.inferenceInProgress = false;
      if(gen == self.gen)
        std::swap(self.utterance, u);
    };
  }
  catch(...)
  {
    return [](TextToken& self) { self.inferenceInProgress = false; };
  }
}

}
