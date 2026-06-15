#pragma once
// Token adapter for the generic ONNX TextToken node (token-sequence -> audio /
// data / tokens). Single-forward token models only: VITS/Piper-style TTS
// (int64 phoneme/token ids [1,L] + scalar scales -> waveform [1,1,N]) and text
// encoders (CLIP/SigLIP text: token ids [1,L] -> embedding [1,D]). This file
// provides the pure, dependency-free logic the node needs:
//
//   - buildTokenTensor : assemble an int64 [1,L] token id buffer from an
//                        incoming id list (+ optional explicit length)
//   - AuxRole / classifyAux : tag each *non-token* model input by name+shape so
//                             the 4 generic float Params can be mapped to the
//                             right scalar (input_lengths, scales[3], sid)
//   - OutputRole / classifyTokenOutput : route output 0 (waveform -> audio,
//                                        vector -> data, tokens -> data)
//   - isAutoregressive : detect KV-cache / past_* / step inputs so the node can
//                        REFUSE autoregressive decode loops (Whisper decoder,
//                        REMI/MIDI transformers) which are OUT OF SCOPE here.
//
// SCOPE / CONTRACT (documented, enforced):
//   * NO TOKENIZER. The node consumes RAW integer token ids; tokenization is the
//     host/user's responsibility. A Text string port may exist for a future
//     tokenizer but is a pass-through no-op until one lands.
//   * NO AUTOREGRESSIVE DECODE. Models with KV-cache / past_* / cache_position /
//     a token-step loop are detected by isAutoregressive() and refused by the
//     node (no-op), since a single forward cannot drive their decode loop.
//
// Everything here is dependency-free (only <cstdint>/<string>/<vector> and the
// dependency-free TensorType.hpp + ModelArchetype.hpp's detail::nameContains),
// so the logic compiles and runs in a standalone test.

#include <Onnx/helpers/ModelArchetype.hpp> // detail::nameContains, isIntType
#include <Onnx/helpers/TensorType.hpp>     // TensorElemType

#include <cstdint>
#include <string>
#include <vector>

namespace Onnx
{

// ---------------------------------------------------------------------------
// Token tensor builder. Most single-forward token models take the ids as a
// 2-D int64 tensor [1, L] (batch 1). Some declare a bare [L]; we honour the
// model's rank via tensorShape(). The id list comes from the host (already
// tokenized); we clamp negatives to 0 and, if the model declares a fixed L>0,
// pad with `pad_id` / truncate to that length.
// ---------------------------------------------------------------------------
struct TokenInput
{
  int rank = 2;       // 2 -> [1,L]; 1 -> [L]
  int64_t fixed_len = 0; // model's declared L (>0 means fixed; else dynamic)

  static TokenInput fromInputShape(const std::vector<int64_t>& s) noexcept
  {
    TokenInput t;
    switch(s.size())
    {
      case 2: // [B,L]
        t.rank = 2;
        t.fixed_len = s[1];
        break;
      case 1: // [L]
        t.rank = 1;
        t.fixed_len = s[0];
        break;
      default:
        t.rank = 2;
        t.fixed_len = s.empty() ? 0 : s.back();
        break;
    }
    return t;
  }

  // The tensor shape for L tokens, matching the model's declared rank.
  std::vector<int64_t> tensorShape(int64_t L) const
  {
    if(rank == 1)
      return {L};
    return {1, L};
  }
};

// Build the int64 token buffer + its shape from an incoming id list. If the
// model declares a fixed length and `respect_fixed` is set, the output is
// padded/truncated to that length (pad_id default 0). Returns the actual L.
// RT-safe: writes into the caller-owned `out` (reused across frames).
template <typename Int>
inline int64_t buildTokenTensor(
    const Int* ids, std::size_t n, const TokenInput& ti,
    std::vector<int64_t>& out, int64_t explicit_len = -1, int64_t pad_id = 0,
    bool respect_fixed = true)
{
  int64_t L = (explicit_len >= 0) ? explicit_len : (int64_t)n;
  if(L < 0)
    L = 0;
  if(respect_fixed && ti.fixed_len > 0)
    L = ti.fixed_len;
  if(L == 0)
    L = (int64_t)n; // never emit an empty token tensor if we have ids

  out.resize((std::size_t)L);
  for(int64_t i = 0; i < L; ++i)
  {
    if((std::size_t)i < n)
    {
      const int64_t v = (int64_t)ids[i];
      out[(std::size_t)i] = (v < 0) ? 0 : v;
    }
    else
    {
      out[(std::size_t)i] = pad_id;
    }
  }
  return L;
}

// ---------------------------------------------------------------------------
// Auxiliary (non-token) input roles. VITS/Piper take, besides the token ids:
//   - input_lengths : int64 [1]  (== number of valid tokens)
//   - scales        : float [3]  (noise_scale, length_scale, noise_scale_w)
//     (some exports split these into separate scalar inputs)
//   - sid / speaker : int64 [1]  speaker id for multi-speaker models
// We classify each non-token input so the node can fill it from the model's
// declared aux inputs by name/shape, mapping the 4 generic float Params on top.
// ---------------------------------------------------------------------------
enum class AuxRole : uint8_t
{
  None = 0,
  TokenIds,    // the primary int token sequence (handled separately)
  InputLength, // input_lengths / text_lengths / x_lengths : int [1]
  Scales,      // a packed float scales vector [N] (noise/length/noise_w)
  NoiseScale,  // individual float scalar
  LengthScale, // individual float scalar
  NoiseScaleW, // individual float scalar
  SpeakerId,   // sid / speaker / spk : int scalar
  GenericFloat,// any other small float scalar -> mapped to a Param
  GenericInt,  // any other small int scalar
  Autoregress, // KV-cache / past_* / step : node must REFUSE
  Unknown,
};

namespace detail
{
inline int64_t tokFlatPos(const std::vector<int64_t>& s) noexcept
{
  int64_t p = 1;
  for(auto d : s)
    if(d > 0)
      p *= d;
  return p;
}

// True for the input names that mark an autoregressive decode loop with a KV
// cache / a per-step iteration. Such models can't be driven by one forward.
inline bool isAutoregName(const std::string& n) noexcept
{
  return anyName(
      n, {"past_key", "past_value", "past_seq", "past.", "past_", "present",
          "cache_position", "kv_cache", "past_key_values", "decoder_input_ids",
          "use_cache", "position_id", "cache"});
}
} // namespace detail

// Classify a single non-token input port into an aux role from name + shape +
// dtype. `dt` distinguishes int (lengths / sid) from float (scales).
inline AuxRole classifyAux(
    const std::string& name, const std::vector<int64_t>& shape,
    TensorElemType dt)
{
  if(detail::isAutoregName(name))
    return AuxRole::Autoregress;

  const bool isInt = detail::isIntType(dt);
  const int64_t pos = detail::tokFlatPos(shape);

  // The token sequence itself: int, rank<=2, flat>1.
  if(isInt && (int)shape.size() >= 1 && (int)shape.size() <= 2 && pos > 1)
    return AuxRole::TokenIds;

  // Specific float scale scalars FIRST, so "length_scale" / "noise_scale" aren't
  // swallowed by the broad "length"/... substrings in the input-length check.
  if(detail::anyName(name, {"noise_scale_w", "noise_w"}))
    return AuxRole::NoiseScaleW;
  if(detail::anyName(name, {"length_scale", "duration", "speed", "rate"}))
    return AuxRole::LengthScale;
  if(detail::anyName(name, {"noise_scale", "noise"}))
    return AuxRole::NoiseScale;

  if(detail::anyName(
         name, {"input_lengths", "text_lengths", "x_lengths", "token_length",
                "length", "lengths", "seq_len"}))
    return AuxRole::InputLength;

  if(detail::anyName(name, {"sid", "speaker", "spk", "voice_id", "spk_id"}))
    return AuxRole::SpeakerId;

  // A small float vector named "scales" packs (noise, length, noise_w).
  if(!isInt && detail::anyName(name, {"scales", "scale"}) && pos >= 1)
    return AuxRole::Scales;

  // A bare float [3] with no name hint is, by VITS convention, the scales.
  if(!isInt && pos == 3 && (int)shape.size() <= 2)
    return AuxRole::Scales;

  if(isInt)
    return AuxRole::GenericInt;
  return AuxRole::GenericFloat;
}

// ---------------------------------------------------------------------------
// Output routing. Output 0 of a token model is either an audio waveform
// ([1,1,N] / [1,N] / [N] with a long axis -> TTS) or a feature vector /
// embedding ([1,D] -> CLIP/SigLIP text) / logits / token ids. We route the
// waveform to the audio port and everything else to the Data port.
// ---------------------------------------------------------------------------
enum class TokenOutputRole : uint8_t
{
  Waveform, // TTS audio [1,1,N] / [1,N] / [N]
  Vector,   // embedding / logits / scores [1,D]
  Tokens,   // int token ids out (e.g. unit/codec codes) -> Data
};

inline TokenOutputRole classifyTokenOutput(
    const std::string& name, const std::vector<int64_t>& shape,
    TensorElemType dt)
{
  const bool isInt = detail::isIntType(dt);
  const bool audioName = detail::anyName(
      name, {"wav", "audio", "pcm", "signal", "waveform", "output", "y_hat",
             "speech", "sound"});
  const int rank = (int)shape.size();
  const int64_t last = shape.empty() ? 0 : shape.back();

  // Int output of token ids -> Data (unit/codec/code sequence).
  if(isInt)
    return TokenOutputRole::Tokens;

  // [1,1,N] / [1,C,N] with a long last axis -> waveform.
  if(rank == 3 && (shape[1] == 1 || shape[1] == 2)
     && (last <= 0 || last > 32))
    return TokenOutputRole::Waveform;
  // [1,N] / [N] with a long axis + audio-ish name -> waveform.
  if((rank == 2 || rank == 1) && (last <= 0 || last > 256) && audioName)
    return TokenOutputRole::Waveform;
  // A long, single-axis float output is almost certainly audio.
  if(rank == 1 && last > 4096)
    return TokenOutputRole::Waveform;

  return TokenOutputRole::Vector;
}

// ---------------------------------------------------------------------------
// Whole-model gate: is this token model an autoregressive decode loop we must
// refuse? True if ANY input is a KV-cache / past_* / step input, OR the model
// is stateful in the classifyModel sense with a token input (transformer
// decoder threading its own cache).
// ---------------------------------------------------------------------------
inline bool isAutoregressive(const ArchIO& io)
{
  for(const auto& p : io.inputs)
    if(detail::isAutoregName(p.name))
      return true;
  for(const auto& p : io.outputs)
    if(detail::anyName(p.name, {"present", "past_key", "past_value"}))
      return true;
  return false;
}

// Convenience: does this model look like a single-forward TTS (a waveform out)?
inline bool isTtsModel(const ArchIO& io)
{
  for(const auto& p : io.outputs)
    if(classifyTokenOutput(p.name, p.shape, p.dtype)
       == TokenOutputRole::Waveform)
      return true;
  return false;
}

} // namespace Onnx
