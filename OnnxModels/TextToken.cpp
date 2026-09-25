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
#include <stdexcept>
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
  // The utterances are resampled to host_rate in the jobs: nothing to rebuild.
}

static void resolveTokenIO(TokenPipeline& P);

// Worker side of a model change: the session (from the file, since the
// port's mapping may be gone by now), the routing, the voices.
static std::shared_ptr<TokenPipeline>
makeTokenPipeline(const std::string& path, double host_rate)
{
  auto pp = std::make_shared<TokenPipeline>();
  auto& P = *pp;
  P.path = path;
  P.host_rate = host_rate;
  {
    std::ifstream f(path, std::ios::binary);
    if(!f)
      throw std::runtime_error("cannot read the file");
    const std::string bytes{std::istreambuf_iterator<char>(f), {}};
    P.ctx = std::make_shared<Onnx::OnnxRunContext>(
        bytes, path, Onnx::Options::precise());
  }
  P.spec = P.ctx->readModelSpec();
  const auto io = toArchIO(P.spec);
  P.arch = Onnx::classifyModel(io);

  // REFUSE autoregressive decode loops (KV-cache / past_* inputs, or a state
  // carried from an output back to an input: Tacotron2's decoder_iter,
  // pocket-tts): a single forward cannot drive their per-token decode.
  P.refused = Onnx::isAutoregressive(io) || P.arch.stateful;

  if(!P.refused)
    resolveTokenIO(P);

  // A style-conditioned TTS reads its voices; which shape they have is in the
  // metadata (style_dim "1,256" or "511,1,256").
  P.style_rows = 1;
  P.style_dim = 0;
  P.sibling_voices.clear();
  for(const auto& p : P.aux)
    if(p.role == AuxRole::Style)
      P.style_dim = (int)flatPositive(p.shape);
  if(P.style_dim > 0)
  {
    try
    {
      Ort::AllocatorWithDefaultOptions alloc;
      auto meta = P.ctx->session.GetModelMetadata();
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
          P.style_rows = dims[0];
      }
    }
    catch(...)
    {
    }
    const auto slash = path.find_last_of("/\\");
    std::ifstream f(
        (slash == std::string::npos ? std::string{} : path.substr(0, slash + 1))
            + "voices.bin",
        std::ios::binary);
    if(f)
    {
      const std::string raw{std::istreambuf_iterator<char>(f), {}};
      P.sibling_voices.resize(raw.size() / sizeof(float));
      std::memcpy(P.sibling_voices.data(), raw.data(), P.sibling_voices.size() * sizeof(float));
    }
  }

  // Nor a model with an input the node cannot synthesise: it only makes
  // scalars, small vectors and token-shaped masks, not a [.,.,288] encoder
  // output (moonshine) or Tacotron's memory.
  if(!P.refused)
    for(int i = 0; i < (int)P.spec.inputs.size(); ++i)
      if(i != P.token_index && P.spec.inputs[i].shape.size() >= 3)
      {
        P.refused = true;
        break;
      }
  if(P.refused)
    std::fprintf(
        stderr,
        "Text Token Processor: %s is an autoregressive decoder or needs inputs "
        "this node cannot build; not run\n",
        path.c_str());
  return pp;
}

// Find the token input, the output to route (waveform -> audio, else -> data),
// classify every non-token input into an aux plan, and map the 4 Params onto
// the scale / speaker / length aux roles.
static void resolveTokenIO(TokenPipeline& P)
{
  P.token_index = -1;
  P.wave_out_index = -1;
  P.aux.clear();

  // Primary token input: prefer a TokenSeq archetype, else the first int input,
  // else input 0.
  for(int i = 0; i < (int)P.arch.inputs.size(); ++i)
  {
    if(P.arch.inputs[i].arch == PortArchetype::TokenSeq)
    {
      P.token_index = i;
      break;
    }
  }
  // Fallback: the first int input with rank<=2 that is NOT a known scalar aux
  // (lengths / sid). This catches dynamic-length token inputs ([1,-1]) that
  // classifyPort tags Latent because their flat size collapses to 1 — a real
  // and common VITS/Piper export (see docs/texttoken-PLAN.md, classifier gap).
  if(P.token_index < 0)
  {
    for(int i = 0; i < (int)P.spec.inputs.size(); ++i)
    {
      // A bool tensor is a mask, never the ids.
      if(!isIntDtype(P.spec.inputs[i].elem_type)
         || P.spec.inputs[i].elem_type == TensorElemType::Bool)
        continue;
      if(P.spec.inputs[i].shape.size() > 2)
        continue;
      const auto r = Onnx::classifyAux(
          P.spec.inputs[i].name, P.spec.inputs[i].shape, P.spec.inputs[i].elem_type);
      if(r == AuxRole::InputLength || r == AuxRole::SpeakerId
         || r == AuxRole::Mask || r == AuxRole::TokenTypes)
        continue; // these are aux inputs, not the token sequence
      P.token_index = i;
      break;
    }
  }
  if(P.token_index < 0)
    P.token_index = 0;

  P.token_in = Onnx::TokenInput::fromInputShape(P.spec.inputs[P.token_index].shape);

  // Aux plans for every other input.
  int next_param = 0; // Param 1..4 round-robin for unmapped scale-like roles
  auto takeParam = [&]() -> int
  { return (next_param < 4) ? next_param++ : -1; };

  for(int i = 0; i < (int)P.spec.inputs.size(); ++i)
  {
    if(i == P.token_index)
      continue;
    TokenAuxPlan p;
    p.model_index = i;
    p.dtype = P.spec.inputs[i].elem_type;
    p.shape = resolveShape(P.spec.inputs[i].shape);
    p.role = Onnx::classifyAux(
        P.spec.inputs[i].name, P.spec.inputs[i].shape, p.dtype);

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
    P.aux.push_back(std::move(p));
  }

  // Output routing: honour an explicit task override; else classify output 0.
  P.produces_audio = false;
  P.wave_out_index = 0;
  for(int o = 0; o < (int)P.spec.outputs.size(); ++o)
  {
    const auto r = Onnx::classifyTokenOutput(
        P.spec.outputs[o].name, P.spec.outputs[o].shape, P.spec.outputs[o].elem_type);
    if(r == TokenOutputRole::Waveform)
    {
      P.produces_audio = true;
      P.wave_out_index = o;
      break;
    }
  }

  if(P.produces_audio)
  {
    P.out_shape
        = Onnx::WaveformShape::fromInputShape(P.spec.outputs[P.wave_out_index].shape);
    if(P.out_shape.channels < 1)
      P.out_shape.channels = 1;
    P.model_rate = ttsRate(P.ctx->session, P.path, P.arch);
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

void TextToken::requestBuild()
{
  building = true;
  requested = std::string(inputs.model.file.filename);
  auto job = JobPool<TokenInferJob>::instance().acquire();
  job->kind = TokenInferJob::Kind::Build;
  job->build_path = requested;
  job->build_host_rate = host_rate;
  worker.request(std::move(job));
}

// On the audio thread: the new model replaces the running one, which goes
// back to the worker to be freed, with the utterance it was playing.
void TextToken::install(std::shared_ptr<TokenPipeline> p)
{
  std::swap(pipe, p);
  last_tokens.clear();
  pending = false;
  ++gen; // a job still running on the previous model is dropped
  inferenceInProgress = false;
  TtsUtterance old;
  std::swap(old, utterance);
  dispose(std::move(p), std::move(old));
}

void TextToken::dispose(std::shared_ptr<TokenPipeline> p, TtsUtterance u)
{
  if(!p && u.samples.empty())
    return;
  auto job = JobPool<TokenInferJob>::instance().acquire();
  job->kind = TokenInferJob::Kind::Dispose;
  job->pipeline = std::move(p);
  job->utterance = std::move(u);
  worker.request(std::move(job));
}

void TextToken::operator()(int frames)
try
{
  auto silence = [&] {
    for(int c = 0; c < outputs.audio.channels; ++c)
      std::fill_n(outputs.audio.samples[c], frames, 0.f);
  };
  if(!available || inputs.model.current_model_invalid
     || inputs.model.file.bytes.empty())
  {
    silence();
    return;
  }

  // A new file: its pipeline is built on the worker.
  if(!building && (!pipe || pipe->path != inputs.model.file.filename)
     && requested != inputs.model.file.filename)
    requestBuild();
  if(!pipe || pipe->refused || pipe->spec.inputs.empty() || pipe->spec.outputs.empty())
  {
    silence();
    return;
  }
  auto& P = *pipe;

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
        ids.data(), ids.size(), P.token_in, token_buf, explicit_len);
    last_tokens.assign(ids.begin(), ids.end());
    pending = false;

    // A style-conditioned TTS without voices would only make noise.
    if(P.style_dim > 0 && !styleFor((int64_t)ids.size()))
    {
      failures.failed(name(), P.path, "needs a voices file (Voices port)");
      return;
    }

    // Every model runs on the worker: a text encoder's result reaches Data a
    // tick later, and nothing heavy runs on the audio thread.
    dispatchInfer(L);
  }

  if(P.produces_audio)
    playUtterance(frames);
  else
    silence();
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

// The style vector of the voice Param 4 picks, from the Voices port or a
// voices.bin next to the model: `style_rows` rows of `style_dim` floats per
// voice; with several rows (Kokoro) the row follows the token count.
bool TextToken::styleFor(int64_t token_count)
{
  const auto& P = *pipe;
  std::span<const float> voices;
  const auto& file = inputs.voices.file.bytes;
  if(file.size() >= sizeof(float))
    voices = {reinterpret_cast<const float*>(file.data()), file.size() / sizeof(float)};
  else
    voices = P.sibling_voices;
  const std::size_t per_voice = (std::size_t)P.style_rows * P.style_dim;
  if(P.style_dim <= 0 || voices.size() < per_voice)
    return false;
  const int nvoices = (int)(voices.size() / per_voice);
  const int v = std::clamp((int)std::lround(inputs.param4.value), 0, nvoices - 1);
  const int r = P.style_rows > 1 ? (int)std::min<int64_t>(token_count, P.style_rows - 1) : 0;
  const float* row = voices.data() + (std::size_t)v * per_voice + (std::size_t)r * P.style_dim;
  style_buf.assign(row, row + P.style_dim);
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

void TextToken::dispatchInfer(int64_t token_len)
{
  if(inferenceInProgress)
    return;
  const auto& P = *pipe;
  inferenceInProgress = true;
  // Pooled job: lock-free acquire; recycled vectors keep their capacity so
  // the assignments below don't allocate in steady state.
  auto job = JobPool<TokenInferJob>::instance().acquire();
  job->kind = TokenInferJob::Kind::Infer;
  job->ctx = P.ctx;
  job->tokens = token_buf;
  job->token_shape = P.token_in.tensorShape((int64_t)token_buf.size());
  job->token_index = P.token_index;
  job->token_len = token_len;
  job->aux = P.aux;
  // The job is recycled: size (not push_back onto) the stale vector.
  job->aux_values.resize(P.aux.size());
  for(std::size_t k = 0; k < P.aux.size(); ++k)
    job->aux_values[k] = P.aux[k].param_index >= 0
                             ? paramValue(P.aux[k].param_index)
                             : P.aux[k].default_value;
  job->wave_out_index = P.wave_out_index;
  job->data_out_index = 0;
  job->produces_audio = P.produces_audio;
  // Snapshot all three VITS scales for the packed-scales models.
  job->scales[0] = inputs.param1.value;
  job->scales[1] = inputs.param2.value;
  job->scales[2] = inputs.param3.value;
  job->out_shape = P.out_shape;
  job->model_rate = P.model_rate;
  job->host_rate = host_rate;
  job->gen = gen;
  job->style.assign(style_buf.begin(), style_buf.end());
  worker.request(std::move(job));
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
      {
        j->ctx.reset(); // don't keep the ORT session alive from the pool
        j->pipeline.reset();
        j->utterance = {};
        j->kind = TokenInferJob::Kind::Infer;
      }
      JobPool<TokenInferJob>::instance().release(std::move(j));
    }
  } recycle{job};

  if(job && job->kind == TokenInferJob::Kind::Dispose)
  {
    // The old session, pipeline and utterance are freed here.
    job->pipeline.reset();
    job->utterance = {};
    return {};
  }
  if(job && job->kind == TokenInferJob::Kind::Build)
  {
    try
    {
      auto p = makeTokenPipeline(job->build_path, job->build_host_rate);
      return [p = std::move(p)](TextToken& self) mutable
      {
        self.building = false;
        if(p->path != self.inputs.model.file.filename)
        {
          self.requested.clear(); // another file was picked meanwhile
          self.dispose(std::move(p), {});
          return;
        }
        self.failures.succeeded();
        self.install(std::move(p));
      };
    }
    catch(const std::exception& e)
    {
      return [what = std::string(e.what()), path = job->build_path](TextToken& self)
      {
        self.building = false;
        if(path != self.inputs.model.file.filename)
          return;
        self.failures.failed(TextToken::name(), path, "cannot load the model: " + what);
        self.inputs.model.current_model_invalid = true;
      };
    }
  }

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
        self.failures.succeeded();
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
      self.failures.succeeded();
      if(gen == self.gen)
      {
        std::swap(self.utterance, u);
        // The utterance it replaced is freed on the worker.
        self.dispose(nullptr, std::move(u));
      }
    };
  }
  catch(const std::exception& e)
  {
    return [what = std::string(e.what())](TextToken& self)
    {
      self.inferenceInProgress = false;
      self.failures.failed(TextToken::name(), self.inputs.model.file.filename, what);
    };
  }
  catch(...)
  {
    return [](TextToken& self)
    {
      self.inferenceInProgress = false;
      self.failures.failed(TextToken::name(), self.inputs.model.file.filename, "unknown error");
    };
  }
}

}
