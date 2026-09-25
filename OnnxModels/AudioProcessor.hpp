#pragma once
// Generic ONNX audio->audio node. Mirrors ImageProcessor's structure: an
// OnnxObject base, an inputs/outputs struct, an operator() that (re)loads +
// classifies the model then runs inference with the same error-handling and
// async-worker pattern. The audio glue lives in <Onnx/helpers/AudioIO.hpp>: a
// per-channel resampler + ring buffer fires the model at its fixed block size,
// builds the [1,C,N] waveform tensor, and streams the result back to the host
// sample-rate. Recurrent state (DeepFilterNet / RAVE-style streaming models) is
// detected via classifyModel and threaded internally (zero-init, output->input
// each call, re-zeroed on Reset).
#include <OnnxModels/Utils.hpp>

#include <Onnx/helpers/AudioIO.hpp>
#include <Onnx/helpers/AuxInputs.hpp>
#include <Onnx/helpers/MelFrontend.hpp>
#include <Onnx/helpers/ModelArchetype.hpp>
#include <Onnx/helpers/ModelSpec.hpp>

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
struct AudioProcessor;

// A recurrent state buffer (flattened), per state port.
struct AudioStatePort
{
  int in_index = 0;
  int out_index = -1;
  std::vector<int64_t> shape;
  std::vector<float> data;
};

// What the node's model and settings resolve to: the ORT session, the I/O
// routing, the resamplers and rings, the vocoder tables, the recurrent state
// and the synchronous path's scratch. Building one creates the session and
// allocates all of this, so it is done by a worker job (buildPipeline), never
// on the audio thread; the audio thread swaps the new one in and sends the
// old one back to the worker to be freed.
struct AudioBuildParams
{
  std::string path;          // the model file, read again on the worker
  std::size_t model_bytes = 0;
  double host_rate = 48000.0;
  int host_channels = 0;
  std::size_t max_frames = 4096;
  int rate_override = 0;
  int overlap = 0;           // AudioOverlap
  int mel_style = 0;         // AudioMelStyle
  int block = 0;

  bool sameSettings(const AudioBuildParams& o) const noexcept
  {
    return host_rate == o.host_rate && host_channels == o.host_channels
           && max_frames == o.max_frames && rate_override == o.rate_override
           && overlap == o.overlap && mel_style == o.mel_style && block == o.block;
  }
};

struct AudioPipeline
{
  AudioBuildParams params;
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  Onnx::ModelSpec spec;
  Onnx::ModelArchetype arch;
  std::string refusal; // non-empty: the model is not something this node runs

  double model_rate = 48000.0;
  int wave_in_index = -1;
  int wave_out_index = -1;
  std::vector<int> wave_out_indices; // separation stems (Param 1 selects)
  // Heavy models run on the worker, one block per job; blocks that become
  // ready meanwhile wait in the input ring (sized for a backlog).
  bool async_model = false;
  std::vector<Onnx::AuxPlan> aux; // inputs other than the waveform and states:
                                  // scalars take Params 2..4
  std::vector<std::vector<uint8_t>> aux_store;
  std::vector<std::vector<int64_t>> aux_shapes;
  Onnx::WaveformShape in_shape, out_shape;

  // A vocoder's [1,n_mels,T] input is fed the log-mel spectrogram of the
  // incoming audio.
  bool mel_input = false;
  std::shared_ptr<Onnx::MelFrontend> mel;
  std::vector<int64_t> mel_shape;
  // A vocoder that returns a spectrum (Vocos: mag, x = cos, y = sin).
  std::shared_ptr<Onnx::SpectrumSynth> synth;
  std::vector<float> synth_acc;
  int spec_out[3]{-1, -1, -1};

  Onnx::WaveformInput audio_in;
  Onnx::WaveformOutput audio_out;
  std::vector<float> staged;      // planar input [C,N]
  std::vector<float> out_planar;  // planar output [C,N]
  std::vector<float> out_scratch; // dtype->float
  std::vector<int64_t> ishape;    // synchronous input shape
  std::vector<int64_t> oshape;    // synchronous output shape
  std::vector<Ort::Value> ins, outs;
  std::vector<AudioStatePort> states;
};

// Off-thread inference job for heavy audio models (separation / vocoders): a
// preprocessed waveform tensor plus the recurrent-state snapshot. ctx is shared
// so it outlives the node frame; readModelSpec() is re-read in work().
// The same job type builds a pipeline (Kind::Build) and frees an old one
// (Kind::Dispose) off the audio thread.
//
// Jobs travel through score::TaskPool's 128-byte smallfun queue, so they are
// heap-allocated and recycled through the lock-free JobPool (only the
// unique_ptr crosses the queue; recycled vectors keep their capacity, so the
// steady-state request path does not allocate). See OnnxModels/JobPool.hpp.
struct AudioInferJob
{
  enum class Kind : uint8_t
  {
    Infer,
    Build,
    Dispose
  } kind = Kind::Infer;
  AudioBuildParams build;                  // Kind::Build
  std::shared_ptr<AudioPipeline> pipeline; // Kind::Dispose: its last owner

  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  std::vector<float> input;    // planar [C,N]
  std::vector<int64_t> ishape; // waveform input shape
  int wave_in_index = 0;                 // which model input is the waveform
  int wave_out_index = 0;                 // which model output is the waveform
  Onnx::WaveformShape out_shape;          // output waveform layout
  // Recurrent state, addressed by model-input index -> flattened tensor.
  std::vector<std::vector<float>> state_in;
  std::vector<std::vector<int64_t>> state_in_shape;
  std::vector<int> state_in_index;  // model input index of each state port
  std::vector<int> state_out_index; // model output index feeding it back
  uint32_t gen = 0;                 // the node's generation at dispatch
  float stem = 0.f;                 // Param 1: the stem of a [B,S,C,N] output
  std::vector<Onnx::AuxPlan> aux;   // the other inputs
  float params[3]{};                // Params 2..4, for the scalar ones
  double model_rate = 48000.0;

  // Vocoder: `input` holds audio and the job computes its log-mel first. The
  // output is a waveform, of which samples [keep_from, keep_from +
  // keep_count) are kept (the context frames' are dropped), or a spectrum
  // (spec_out: magnitude, cos, sin) synthesised over frames [keep_from,
  // keep_from + keep_count) with the node's overlap state synth_acc.
  std::shared_ptr<const Onnx::MelFrontend> mel;
  std::shared_ptr<const Onnx::SpectrumSynth> synth;
  std::vector<float> synth_acc;
  int spec_out[3]{-1, -1, -1};
  int64_t keep_from = 0;
  int64_t keep_count = -1;
};

// The log-mel features a vocoder expects (see MelFrontend.hpp).
enum class AudioMelStyle
{
  Auto,    // Vocos for 100 mel bins, else HiFi-GAN
  HiFiGAN, // Slaney, 0..8 kHz, log floor 1e-5 (HiFi-GAN, MelGAN, Matcha)
  Vocos    // HTK, 0..Nyquist, log floor 1e-7 (vocos-mel-24khz)
};

// Frame-based streaming models (DTLN-style enhancement) take a block every
// hop samples and their outputs are overlap-added.
enum class AudioOverlap
{
  None,         // a new block every block (hop = block)
  Half,         // hop = block / 2
  ThreeQuarters // hop = block / 4
};

struct AudioProcessor : OnnxObject
{
public:
  halp_meta(name, "Audio Processor");
  halp_meta(c_name, "audio_processor");
  halp_meta(category, "AI/Audio Processing");
  halp_meta(author, "ossia team");
  halp_meta(
      description,
      "Generic ONNX audio->audio processor: source separation (stem output), "
      "denoise/enhancement (recurrent state threaded internally), neural "
      "vocoders and voice conversion. Auto-detects the waveform layout, channel "
      "count and model sample-rate; resamples to/from the host rate.");
  halp_meta(uuid, "b1c2d3e4-f5a6-4789-9abc-def012345678");

  struct
  {
    halp::dynamic_audio_bus<"In", float> audio;
    ModelPort<"Model"> model;
    halp::hslider_f32<"Param 1", halp::range{0., 1., 0.}> param1;
    halp::hslider_f32<"Param 2", halp::range{0., 1., 0.}> param2;
    halp::hslider_f32<"Param 3", halp::range{0., 1., 0.}> param3;
    halp::hslider_f32<"Param 4", halp::range{0., 1., 0.}> param4;
    halp::impulse_button<"Reset"> reset;
    // Appended: the model's sample rate when Auto (0) guesses wrong.
    struct : halp::spinbox_i32<"Model Rate", halp::range{0, 192000, 0}>
    {
      halp_meta(description, "Sample rate the model runs at; 0 = auto");
    } model_rate;
    struct : halp::enum_t<AudioOverlap, "Overlap">
    {
      halp_meta(
          description,
          "Run the model on overlapping blocks and overlap-add its outputs "
          "(Hann window); for models that return a block as long as their "
          "input");
    } overlap;
    struct : halp::enum_t<AudioMelStyle, "Mel">
    {
      halp_meta(
          description,
          "Log-mel features fed to a vocoder ([1,n_mels,T] input); the "
          "model's metadata sets the sample rate, n_fft and hop");
    } mel_style;
    struct : halp::spinbox_i32<"Block", halp::range{0, 1 << 21, 0}>
    {
      halp_meta(
          description,
          "Samples per inference, at the model's rate, for a model whose "
          "length is free; 0 = 1024. Demucs wants its training segment, "
          "343980 at 44.1 kHz");
    } block;
  } inputs;

  struct
  {
    halp::dynamic_audio_bus<"Out", float> audio;
    halp::val_port<"Data", std::vector<float>> data;
  } outputs;

  AudioProcessor() noexcept;
  ~AudioProcessor();

  void prepare(halp::setup info);
  void operator()(int frames);

  struct worker
  {
    std::function<void(std::unique_ptr<AudioInferJob>)> request;
    static std::function<void(AudioProcessor&)>
    work(std::unique_ptr<AudioInferJob> job);
  } worker;

private:
  // The running pipeline, and whether a build is in flight. While one is,
  // the current pipeline keeps running (a settings change has no gap).
  std::shared_ptr<AudioPipeline> pipe;
  bool building = false;
  AudioBuildParams requested; // what the last build was asked for
  bool inferenceInProgress = false;
  // Reset, and each new pipeline, bump gen: a job they outdated does not
  // bring back its states or its block.
  uint32_t gen = 0;

  // Host audio config (from prepare()).
  double host_rate = 48000.0;
  int host_in_channels = 0;
  std::size_t max_frames = 4096;

  AudioBuildParams currentParams() const;
  void requestBuild(const AudioBuildParams& p, bool reuse_session);
  void install(std::shared_ptr<AudioPipeline> p);
  void dispose(std::shared_ptr<AudioPipeline> p);
  void zeroStates();
  void runBlock();
  void dispatchInfer(int64_t n, bool force_async);
};

}
