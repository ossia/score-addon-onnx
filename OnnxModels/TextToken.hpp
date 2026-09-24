#pragma once
// Generic ONNX TextToken node (token-sequence -> audio / data / tokens).
// Mirrors ImageProcessor / AudioProcessor structure: an OnnxObject base, an
// inputs/outputs struct, an operator() that (re)loads + classifies the model
// then runs a SINGLE forward with the same error-handling and async-worker
// pattern. The token / aux-input / output-routing logic lives in the
// dependency-free <Onnx/helpers/TokenIO.hpp>; the waveform output path reuses
// <Onnx/helpers/AudioIO.hpp>'s WaveformOutput (resample model SR -> host SR).
//
// SCOPE (single-forward token models ONLY):
//   * VITS / Piper TTS         : int64 token ids [1,L] (+ input_lengths, scales,
//                                 speaker id) -> waveform [1,1,N] -> audio Out.
//   * CLIP / SigLIP text enc.  : int64 token ids [1,L] -> embedding [1,D] -> Data.
//
// OUT OF SCOPE (detected + refused, node no-ops):
//   * Autoregressive decode loops with a KV-cache / past_* inputs (Whisper
//     decoder, REMI/MIDI transformers). isAutoregressive() flags them; the node
//     marks the model invalid and produces nothing.
//   * A real text TOKENIZER. The Tokens port carries RAW integer ids already
//     produced by the host/user. The optional Text string port is a documented
//     pass-through no-op reserved for a future tokenizer.
#include <OnnxModels/Utils.hpp>

#include <Onnx/helpers/AudioIO.hpp>
#include <Onnx/helpers/ModelArchetype.hpp>
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/TokenIO.hpp>

#include <halp/audio.hpp>
#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Onnx
{
struct OnnxRunContext;
}

namespace OnnxModels
{
struct TextToken;

// One model input, fully resolved: how to fill it for a forward pass. Built once
// per model load in resolveIO(); consumed every frame (sync) and snapshotted
// into the worker job (async, TTS).
struct TokenAuxPlan
{
  int model_index = 0;            // which model input this is
  Onnx::AuxRole role = Onnx::AuxRole::Unknown;
  std::vector<int64_t> shape;     // resolved (dynamic -> 1) tensor shape
  Onnx::TensorElemType dtype = Onnx::TensorElemType::Float;
  int param_index = -1;           // which of Param 1..4 maps here (-1 == none)
  float default_value = 0.f;      // fallback / VITS-sane default
};

// Off-thread inference job for heavy (TTS) token models: the assembled token
// buffer + the aux scalar values, plus everything work() needs to run + decode
// the waveform without touching the node. ctx is shared so it outlives the node
// frame; readModelSpec() is re-read in work() for thread-stable name ptrs.
//
// Jobs travel through score::TaskPool's 128-byte smallfun queue, so they are
// heap-allocated and recycled through the lock-free JobPool (only the
// unique_ptr crosses the queue; recycled vectors keep their capacity, so the
// steady-state request path does not allocate). See OnnxModels/JobPool.hpp.
struct TokenInferJob
{
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  std::vector<int64_t> tokens;      // assembled int64 token ids
  std::vector<int64_t> token_shape; // [1,L] or [L]
  int token_index = 0;              // which model input is the token sequence
  int64_t token_len = 0;            // valid token count (for input_lengths)
  std::vector<TokenAuxPlan> aux;    // non-token inputs to fill
  std::vector<float> aux_values;    // resolved Param values, parallel to `aux`
  int wave_out_index = 0;           // output routed to audio
  int data_out_index = 0;           // output routed to the Data port (encoders)
  bool produces_audio = false;      // route to the audio outlet vs outputs.data
  float scales[3] = {0.667f, 1.0f, 0.8f}; // VITS packed [noise,length,noise_w]
  Onnx::WaveformShape out_shape;    // waveform layout
  double model_rate = 22050.0;      // TTS native rate
  double host_rate = 48000.0;       // the utterance is resampled to it here
  uint32_t gen = 0;                 // the node's generation at dispatch
  std::vector<float> style;         // the voice row, for AuxRole::Style
};

// A synthesised utterance at host rate, planar [channel][frame], played once.
// It is built whole on the worker and swapped in, so it has no size limit and
// playing it does not allocate.
struct TtsUtterance
{
  std::vector<float> samples;
  int channels = 0;
  std::size_t frames = 0;
  std::size_t pos = 0;
};

struct TextToken : OnnxObject
{
public:
  halp_meta(name, "Text Token Processor");
  halp_meta(c_name, "text_token");
  halp_meta(category, "AI/Text Processing");
  halp_meta(author, "ossia team");
  halp_meta(
      description,
      "Generic ONNX token-sequence processor: single-forward TTS (VITS / Piper: "
      "token ids -> waveform) and text encoders (CLIP / SigLIP text: token ids "
      "-> embedding). Consumes RAW integer token ids (no built-in tokenizer). "
      "Autoregressive decode models (KV-cache / past_*) are detected and "
      "refused.");
  halp_meta(uuid, "c3d4e5f6-a7b8-4901-bcde-f01234567890");

  struct
  {
    // RAW integer token ids (host/user tokenizes; see scope note above).
    halp::val_port<"Tokens", std::vector<int>> tokens;
    ModelPort<"Model"> model;
    // Optional explicit valid-length; <=0 means "use the token list size".
    halp::spinbox_i32<"Length", halp::range{-1, 8192, -1}> length;
    // Generic params mapped to scales / speaker id / length-scale by the model's
    // declared aux inputs (resolveIO decides the mapping). VITS-sane defaults.
    halp::hslider_f32<"Param 1", halp::range{0., 4., 0.667}> param1; // noise
    halp::hslider_f32<"Param 2", halp::range{0., 4., 1.0}> param2;   // length
    halp::hslider_f32<"Param 3", halp::range{0., 4., 0.8}> param3;   // noise_w
    halp::hslider_f32<"Param 4", halp::range{0., 256., 0.}> param4;  // speaker
    // Reserved pass-through for a FUTURE tokenizer; ignored today (raw ids win).
    halp::lineedit<"Text", ""> text;
    halp::impulse_button<"Reset"> reset;
    // Appended: the style vectors of a style-conditioned TTS (Kitten, Kokoro
    // voices.bin, raw float32). Param 4 picks the voice. When empty, a
    // voices.bin next to the model is used.
    struct : halp::file_port<"Voices", halp::mmap_file_view>
    {
      halp_meta(extensions, "*.bin");
    } voices;
  } inputs;

  struct
  {
    halp::dynamic_audio_bus<"Out", float> audio; // TTS waveform
    halp::val_port<"Data", std::vector<float>> data; // embedding / logits / tokens
  } outputs;

  TextToken() noexcept;
  ~TextToken();

  void prepare(halp::setup info);
  void operator()(int frames);

  // Heavy (TTS) models run off the render thread; light text encoders run inline.
  struct worker
  {
    std::function<void(std::unique_ptr<TokenInferJob>)> request;
    static std::function<void(TextToken&)>
    work(std::unique_ptr<TokenInferJob> job);
  } worker;

private:
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  Onnx::ModelSpec spec;
  Onnx::ModelArchetype arch;
  std::string lastModelPath;
  bool inferenceInProgress = false;
  bool refused = false; // autoregressive -> permanently no-op this model

  // Host audio config (from prepare()).
  double host_rate = 48000.0;
  std::size_t max_frames = 4096;

  // Resolved model I/O.
  int token_index = -1;
  int wave_out_index = -1;
  bool produces_audio = false;
  Onnx::TokenInput token_in;
  Onnx::WaveformShape out_shape;
  double model_rate = 22050.0;
  std::vector<TokenAuxPlan> aux;

  TtsUtterance utterance; // the one playing
  std::vector<int64_t> token_buf; // assembled int64 token ids
  // Dispatch state. The model runs when the ids change or on Reset, once per
  // request: last_tokens is committed on dispatch, a request made while a job
  // is in flight waits for it (latest wins), and each request bumps gen so a
  // result it superseded is dropped.
  std::vector<int> last_tokens;
  bool pending = false;
  uint32_t gen = 0;

  // Style-conditioned TTS: `style_rows` rows of `style_dim` floats per voice
  // (Kokoro: 511 rows, indexed by the token count; Kitten: 1).
  int style_rows = 1, style_dim = 0;
  std::vector<float> sibling_voices; // voices.bin next to the model
  std::vector<float> style_buf;
  bool styleFor(int64_t token_count);

  // Per-input owned backing buffers so each Ort::Value references a distinct,
  // alive store for the whole infer() call (reused across frames).
  std::vector<std::vector<int64_t>> aux_int_bufs;
  std::vector<std::vector<float>> aux_flt_bufs;
  std::vector<std::vector<uint8_t>> int_narrow_bufs; // int32/bool/... copies
  std::vector<float> out_scratch; // dtype->float for the audio output

  void reloadModel();
  void resolveIO();
  float paramValue(int idx) const;
  void run();
  void dispatchInfer(int64_t token_len, bool force_async);
  void playUtterance(int frames);
};

}
