#pragma once
// Modality-agnostic model classifier: the shared keystone for the specialized
// ONNX node set (see docs/onnx-node-set-PLAN.md). Generalises classifyImage to
// audio / video / sequence / geometry / generative by tagging every input and
// output port with a PortArchetype (from shape + name + dtype), detecting
// recurrent statefulness, and suggesting which specialized node should host the
// model. Dependency-free (reuses Onnx::detail::nameContains); standalone-testable.
#include <Onnx/helpers/ModelRole.hpp>  // detail::nameContains
#include <Onnx/helpers/TensorType.hpp> // TensorElemType

#include <cstdint>
#include <string>
#include <vector>

namespace Onnx
{
enum class PortArchetype : uint8_t
{
  Unknown = 0,
  Image,          // [.,C,H,W] / [.,H,W,C], C in {1,3,4}
  Waveform,       // [.,1,N] / [.,2,N] / [.,N] audio samples
  Spectrogram,    // [.,F,T] mel / STFT magnitude (or [.,2,F,T] complex)
  PointSet,       // [.,N,3] / [.,3,N] point cloud / vertices
  Sequence,       // [.,T,F] generic time x features
  Vector,         // [.,D] fixed-size embedding / scores / params-out
  Latent,         // [.,Z] generative latent input
  TokenSeq,       // [.,L] integer token ids
  Scalar,         // [] / [1] / [1,1] scalar / control
  RecurrentState, // matches an output, or named state/hidden/cache/r#
};

enum class NodeKind : uint8_t
{
  Unknown = 0,
  ImageProcessor,    // image -> image / mask / depth / data
  ImageGenerator,    // latent -> image
  VideoProcessor,    // image (+ recurrent state) -> image
  AudioProcessor,    // audio -> audio
  AudioAnalyzer,     // audio -> data / control
  SequenceProcessor, // sequence -> sequence
  GeometryProcessor, // point set -> point set / data
  TextToken,         // tokens -> audio / data / tokens
};

// Dependency-free I/O view with dtype (built from Onnx::ModelSpec by the nodes).
struct ArchIO
{
  struct Port
  {
    std::string name;
    std::vector<int64_t> shape;
    TensorElemType dtype = TensorElemType::Float;
  };
  std::vector<Port> inputs, outputs;
};

struct ArchPort
{
  std::string name;
  std::vector<int64_t> shape;
  TensorElemType dtype = TensorElemType::Float;
  PortArchetype arch = PortArchetype::Unknown;
};

struct ModelArchetype
{
  std::vector<ArchPort> inputs, outputs;
  NodeKind suggested = NodeKind::Unknown;
  bool stateful = false; // has recurrent-state ports threaded internally

  bool valid() const noexcept { return suggested != NodeKind::Unknown; }
};

namespace detail
{
inline bool isChanDim(int64_t d) noexcept { return d == 1 || d == 3 || d == 4; }

inline int64_t flatPos(const std::vector<int64_t>& s)
{ // product of positive dims (treat dynamic as 1)
  int64_t p = 1;
  for(auto d : s)
    if(d > 0)
      p *= d;
  return p;
}
inline bool isIntType(TensorElemType t) noexcept
{
  switch(t)
  {
    case TensorElemType::Uint8:
    case TensorElemType::Int8:
    case TensorElemType::Uint16:
    case TensorElemType::Int16:
    case TensorElemType::Uint32:
    case TensorElemType::Int32:
    case TensorElemType::Uint64:
    case TensorElemType::Int64:
    case TensorElemType::Bool:
      return true;
    default:
      return false;
  }
}
inline bool anyName(const std::string& n, std::initializer_list<const char*> ks)
{
  for(auto k : ks)
    if(nameContains(n, k))
      return true;
  return false;
}

// Classify a single port from shape + name + dtype. `is_input` disambiguates
// Latent (generative input) vs Vector (embedding output).
inline PortArchetype classifyPort(
    const std::string& name, const std::vector<int64_t>& s, TensorElemType dt,
    bool is_input)
{
  const int rank = (int)s.size();
  const bool audioName
      = anyName(name, {"wav", "audio", "pcm", "signal", "mix", "sound", "speech"});
  // NOTE: bare "mag" must not be a substring match — it matches "iMAGe" (the
  // single most common image port name). Allow it only as an exact name or a
  // _mag suffix/prefix component (stft_mag, mag_spec).
  const bool specName
      = anyName(name, {"mel", "spec", "stft", "fbank", "magnit", "_mag", "mag_"})
        || name == "mag";
  const bool pointName = anyName(name, {"point", "cloud", "xyz", "vert", "pcd"});
  const bool stateName
      = anyName(name, {"state", "hidden", "cache", "memory", "context_"});
  // Small int metadata (image sizes / shapes) must not be read as token ids.
  const bool metaName
      = anyName(name, {"size", "shape", "_dim", "width", "height"});

  if(stateName)
    return PortArchetype::RecurrentState;

  // Integer sequences are tokens (phonemes, BPE, REMI, codec codes). A dynamic
  // length [1,-1] has flatPos==1 but is still a token sequence (VITS/Piper).
  if(isIntType(dt) && rank >= 1 && rank <= 2 && !metaName)
  {
    bool dynamic = false;
    for(auto d : s)
      if(d <= 0)
        dynamic = true;
    // rank-2 [1,-1] is a dynamic token sequence; a rank-1 dynamic [-1] is a
    // batch-length scalar list, not tokens.
    if(flatPos(s) > 1 || (dynamic && rank == 2))
      return PortArchetype::TokenSeq;
  }

  if(rank <= 1)
    return PortArchetype::Scalar;

  // rank >= 5 ([N,C,T,H,W] clip stacks, attention caches): out of scope for the
  // current node set — refuse explicitly instead of falling through to the
  // rank-2 branch and accidentally reading mars [1,3,T,H,W] as a Latent.
  if(rank > 4)
    return PortArchetype::Unknown;

  if(rank == 4)
  {
    // A mel/STFT-named 4-D tensor is a spectrogram even when it's 1-channel and
    // thus image-shaped (e.g. CLAP mel_fusion [B,1,T,F]); check name first.
    if(specName)
      return PortArchetype::Spectrogram;
    if(isChanDim(s[1]) || isChanDim(s[3]))
      return PortArchetype::Image;       // NCHW / NHWC
    if(s[1] <= 0)
      return PortArchetype::Image;       // symbolic CHANNEL (e.g. depth_pro)
    // Concrete non-image channel (16/32/...) is a feature / recurrent-state map,
    // even when its spatial dims are symbolic (e.g. RVM r1i [1,16,-1,-1]).
    return PortArchetype::Unknown;
  }

  if(rank == 3)
  {
    if(specName)
      return PortArchetype::Spectrogram;
    if(pointName || s[2] == 3 || s[1] == 3)
      return PortArchetype::PointSet;
    // [.,1,N] / [.,2,N] with a long last axis -> waveform; small front dims.
    if((s[1] == 1 || s[1] == 2) && (s[2] <= 0 || s[2] > 32))
      return PortArchetype::Waveform;
    if(audioName)
      return PortArchetype::Waveform;
    if(s[0] == 3 || s[0] == 4)
      return PortArchetype::Image; // CHW (no batch) — real image channels only
    return PortArchetype::Sequence; // [.,T,F]
  }

  // rank == 2
  if(audioName)
    return PortArchetype::Waveform; // [.,N]
  if(specName)
    return PortArchetype::Spectrogram;
  {
    const int64_t d = (s[1] > 0) ? s[1] : (s[0] > 0 ? s[0] : 0);
    // Big front-loaded vector with audio/no name and large N could be waveform;
    // otherwise it's a latent (input) or an embedding/scores (output).
    if(is_input)
      return PortArchetype::Latent;
    return PortArchetype::Vector;
    (void)d;
  }
}
} // namespace detail

inline ModelArchetype classifyModel(const ArchIO& io)
{
  ModelArchetype m;
  m.inputs.reserve(io.inputs.size());
  m.outputs.reserve(io.outputs.size());
  for(const auto& p : io.inputs)
    m.inputs.push_back(
        {p.name, p.shape, p.dtype,
         detail::classifyPort(p.name, p.shape, p.dtype, /*is_input*/ true)});
  for(const auto& p : io.outputs)
    m.outputs.push_back(
        {p.name, p.shape, p.dtype,
         detail::classifyPort(p.name, p.shape, p.dtype, /*is_input*/ false)});

  // Recurrent state: an input whose shape matches some output, or a name-paired
  // r#i/r#o / *_in -> *_out. Retag and mark stateful. Dynamic dims (<=0) act as
  // wildcards — real exports routinely lose the batch dim on the output side
  // (dtln2: in [1,2,128,2] vs out [N,2,128,2]) — but at least two concrete dims
  // must agree so fully-symbolic shapes don't match everything.
  auto sameShape = [](const std::vector<int64_t>& a, const std::vector<int64_t>& b)
  {
    if(a.size() != b.size() || a.empty())
      return false;
    int concrete = 0;
    for(size_t i = 0; i < a.size(); ++i)
    {
      if(a[i] <= 0 || b[i] <= 0)
        continue; // wildcard
      if(a[i] != b[i])
        return false;
      ++concrete;
    }
    return concrete >= 2;
  };
  // r#i<->r#o or *_in<->*_out name pairing: recovers recurrent states even when
  // their shapes are fully symbolic and indistinguishable from the image input
  // (real RVM exports r1i..r4i as [?,?,?,?] — shape-matching alone tags them
  // Image). Name pairing is shape-independent.
  auto endsWith = [](const std::string& s, const char* suf)
  {
    const std::string t(suf);
    return s.size() >= t.size()
           && s.compare(s.size() - t.size(), t.size(), t) == 0;
  };
  auto outNamed = [&](const std::string& nm)
  {
    for(const auto& o : m.outputs)
      if(o.name == nm)
        return true;
    return false;
  };
  auto nameStatePair = [&](const std::string& nm)
  {
    if(nm.empty())
      return false;
    if(nm.back() == 'i' && outNamed(nm.substr(0, nm.size() - 1) + "o"))
      return true; // r1i -> r1o
    if(endsWith(nm, "_in") && outNamed(nm.substr(0, nm.size() - 3) + "_out"))
      return true; // foo_in -> foo_out
    return false;
  };
  // Retag the OUTPUT side of a state pair too: a state output left as
  // Sequence/Vector would otherwise drive NodeKind routing (silero's hn/cn made
  // the whole model look like a SequenceProcessor).
  auto retagPairedOutput = [&](const ArchPort& in)
  {
    for(auto& o : m.outputs)
    {
      if(o.arch == PortArchetype::RecurrentState)
        continue; // already claimed by another state input (h->hn, then c->cn)
      const auto& nm = in.name;
      const bool named
          = (!nm.empty() && nm.back() == 'i'
             && o.name == nm.substr(0, nm.size() - 1) + "o")
            || (endsWith(nm, "_in") && o.name == nm.substr(0, nm.size() - 3) + "_out")
            || (o.name == nm + "n"); // h -> hn, c -> cn (silero / LSTM exports)
      if(named || (o.arch != PortArchetype::Image && sameShape(in.shape, o.shape)))
      {
        o.arch = PortArchetype::RecurrentState;
        return;
      }
    }
  };
  for(size_t i = 0; i < m.inputs.size(); ++i)
  {
    auto& in = m.inputs[i];
    if(in.arch == PortArchetype::RecurrentState) // name-flagged (state/hidden/...)
    {
      m.stateful = true;
      retagPairedOutput(in);
      continue;
    }
    if(nameStatePair(in.name)) // r#i/r#o, *_in/*_out — any shape, any index
    {
      in.arch = PortArchetype::RecurrentState;
      m.stateful = true;
      retagPairedOutput(in);
      continue;
    }
    // Shape-matched state only for SECONDARY inputs: input 0 is the primary media
    // (so a [1,T,F]->[1,T,F] sequence/AE model is not falsely stateful), and a 2nd
    // image input is an image-pair (flow/inpaint/stereo), never recurrent state.
    if(i == 0 || in.arch == PortArchetype::Image)
      continue;
    if(detail::flatPos(in.shape) <= 1)
      continue;
    for(const auto& out : m.outputs)
    {
      if(sameShape(in.shape, out.shape))
      {
        in.arch = PortArchetype::RecurrentState;
        m.stateful = true;
        retagPairedOutput(in);
        break;
      }
    }
  }

  // --- suggest a node kind from the dominant non-state archetypes ---
  auto hasIn = [&](PortArchetype a)
  {
    for(auto& p : m.inputs)
      if(p.arch == a)
        return true;
    return false;
  };
  auto hasOut = [&](PortArchetype a)
  {
    for(auto& p : m.outputs)
      if(p.arch == a)
        return true;
    return false;
  };

  // Audio-domain context: with a Waveform/Spectrogram input and no Image input,
  // a rank-4 output that was only "Image" via the symbolic-channel fallback is
  // really audio-domain data: a stem/waveform stack for waveform-in models
  // (demucs exports x as [?,?,?,?]) or a separated magnitude for
  // spectrogram-in ones (audiosep's output [?,?,?,?]), not a picture.
  if((hasIn(PortArchetype::Waveform) || hasIn(PortArchetype::Spectrogram))
     && !hasIn(PortArchetype::Image))
  {
    const auto audio_arch = hasIn(PortArchetype::Waveform)
                                ? PortArchetype::Waveform
                                : PortArchetype::Spectrogram;
    for(auto& o : m.outputs)
      if(o.arch == PortArchetype::Image && o.shape.size() == 4 && o.shape[1] <= 0
         && !detail::isChanDim(o.shape[3]))
        o.arch = audio_arch;
  }

  const bool audioIn = hasIn(PortArchetype::Waveform) || hasIn(PortArchetype::Spectrogram);
  const bool audioOut = hasOut(PortArchetype::Waveform) || hasOut(PortArchetype::Spectrogram);

  if(hasIn(PortArchetype::PointSet))
    m.suggested = NodeKind::GeometryProcessor;
  else if(hasIn(PortArchetype::TokenSeq))
    m.suggested = NodeKind::TextToken;
  else if(audioIn)
    m.suggested = audioOut ? NodeKind::AudioProcessor : NodeKind::AudioAnalyzer;
  else if(hasIn(PortArchetype::Image))
    m.suggested = m.stateful ? NodeKind::VideoProcessor : NodeKind::ImageProcessor;
  else if((hasIn(PortArchetype::Latent) || hasIn(PortArchetype::Vector)
           || hasIn(PortArchetype::Sequence))
          && hasOut(PortArchetype::Image))
    // Latent z, or a StyleGAN w+ matrix [1,18,512] (Sequence-shaped), -> image.
    m.suggested = NodeKind::ImageGenerator;
  else if(hasOut(PortArchetype::Waveform))
    m.suggested = NodeKind::AudioProcessor; // vocoder: mel/sequence/latent -> wav
  else if(hasIn(PortArchetype::Sequence) || hasOut(PortArchetype::Sequence))
    m.suggested = NodeKind::SequenceProcessor;
  else if(hasIn(PortArchetype::Latent) && hasOut(PortArchetype::Vector))
    // Vector-in -> vector-out MLP head (CLIP-embedding classifiers etc.):
    // SequenceProcessor's [1,D] vector path runs these.
    m.suggested = NodeKind::SequenceProcessor;
  else
    m.suggested = NodeKind::Unknown;

  return m;
}
} // namespace Onnx
