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

// Off-thread inference job for heavy audio models (separation / vocoders): a
// preprocessed waveform tensor plus the recurrent-state snapshot. ctx is shared
// so it outlives the node frame; readModelSpec() is re-read in work().
//
// Jobs travel through score::TaskPool's 128-byte smallfun queue, so they are
// heap-allocated and recycled through the lock-free JobPool (only the
// unique_ptr crosses the queue; recycled vectors keep their capacity, so the
// steady-state request path does not allocate). See OnnxModels/JobPool.hpp.
struct AudioInferJob
{
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
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  Onnx::ModelSpec spec;
  Onnx::ModelArchetype arch;
  std::string lastModelPath;
  bool inferenceInProgress = false;
  // Heavy models run on the worker, one block per job. Blocks that become
  // ready meanwhile wait in the input ring (sized for a backlog) instead of
  // being consumed and dropped; Reset bumps gen so a job it outdated does not
  // bring back the old states.
  bool async_model = false;
  uint32_t gen = 0;

  // Host audio config (from prepare()).
  double host_rate = 48000.0;
  int host_in_channels = 0;
  std::size_t max_frames = 4096;

  // Resolved model I/O.
  double model_rate = 48000.0;
  int wave_in_index = -1;
  int wave_out_index = -1;
  std::vector<int> wave_out_indices; // separation stems (Param 1 selects)
  int lastRateOverride = 0;
  AudioOverlap lastOverlap{};
  AudioMelStyle lastMelStyle{};
  int lastBlock = 0;
  std::vector<Onnx::AuxPlan> aux; // inputs other than the waveform and states:
                                  // scalars take Params 2..4
  std::vector<std::vector<uint8_t>> aux_store;
  Onnx::WaveformShape in_shape, out_shape;

  // A vocoder's [1,n_mels,T] input is fed the log-mel spectrogram of the
  // incoming audio (HiFi-GAN settings).
  bool mel_input = false;
  std::shared_ptr<Onnx::MelFrontend> mel;
  std::vector<int64_t> mel_shape;
  // A vocoder that returns a spectrum (Vocos: mag, x = cos, y = sin).
  std::shared_ptr<Onnx::SpectrumSynth> synth;
  std::vector<float> synth_acc;
  int spec_out[3]{-1, -1, -1};

  Onnx::WaveformInput audio_in;
  Onnx::WaveformOutput audio_out;
  std::vector<float> staged;     // planar input [C,N]
  std::vector<float> out_planar; // planar output [C,N]
  std::vector<float> out_scratch;         // dtype->float
  std::vector<std::vector<int64_t>> aux_shapes;

  // Recurrent state buffers (flattened), per state port.
  struct StatePort
  {
    int in_index = 0;
    int out_index = -1;
    std::vector<int64_t> shape;
    std::vector<float> data;
  };
  std::vector<StatePort> states;

  void reloadModel();
  void resolveIO();
  void zeroStates();
  void runBlock();
  void dispatchInfer(int64_t n, bool force_async);
};

}
