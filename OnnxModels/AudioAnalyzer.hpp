#pragma once
// Generic ONNX audio->data/control node. Same skeleton as AudioProcessor, but
// the model emits descriptors instead of audio: F0/pitch (CREPE), voice-activity
// probability (Silero-VAD, recurrent), audio tags (YAMNet), or embeddings
// (CLAP). The waveform input is accumulated + resampled like AudioProcessor; the
// (possibly recurrent) outputs are flattened into a Data vector, with the first
// two scalar-ish results also exposed as float Value ports (e.g. F0 + confidence,
// or VAD probability). The node's kind is FIXED audio-analyzer, so the audio
// input is treated as a waveform even when classifyModel mistags it (Silero-VAD's
// bare [1,512] 'input').
#include <OnnxModels/Utils.hpp>

#include <Onnx/helpers/AudioIO.hpp>
#include <Onnx/helpers/AuxInputs.hpp>
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

// How to reduce the model output(s) to the two scalar Value ports.
enum class AnalyzerReduce
{
  Auto,    // pitch->argmax bin + confidence; vad->prob; else value[0]/[1]
  Argmax,  // value0 = argmax index, value1 = max value
  FirstTwo, // value0 = out[0], value1 = out[1]
  MeanPeak, // value0 = mean, value1 = peak
};

struct AudioAnalyzer;

// What the node's model and settings resolve to: the session, the I/O
// routing, the resampler and ring, the recurrent state and the inference
// scratch. Built by a worker job, never on the audio thread; the audio thread
// swaps the new one in and hands the old one back to the worker to be freed.
// (The model itself runs inline: analyzers are small, and their controls
// should follow the audio closely.)
struct AnalyzerBuildParams
{
  std::string path; // the model file, read again on the worker
  double host_rate = 48000.0;
  int host_channels = 0;
  std::size_t max_frames = 4096;
  int rate_override = 0;

  bool sameSettings(const AnalyzerBuildParams& o) const noexcept
  {
    return host_rate == o.host_rate && host_channels == o.host_channels
           && max_frames == o.max_frames && rate_override == o.rate_override;
  }
};

struct AnalyzerStatePort
{
  int in_index = 0;
  int out_index = -1;
  std::vector<int64_t> shape;
  std::vector<float> data;
};

struct AnalyzerPipeline
{
  AnalyzerBuildParams params;
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  Onnx::ModelSpec spec;
  Onnx::ModelArchetype arch;
  std::string refusal; // non-empty: the model is not something this node runs

  double model_rate = 16000.0;
  int wave_in_index = -1;
  Onnx::WaveformShape in_shape;
  Onnx::WaveformInput audio_in;
  std::vector<float> staged;
  std::vector<float> out_scratch;
  std::vector<float> flat; // the primary output, as floats
  // The other inputs (sr, lengths, masks, flags, scalars), typed as declared.
  std::vector<Onnx::AuxPlan> aux;
  std::vector<std::vector<uint8_t>> aux_store;
  std::vector<std::vector<int64_t>> aux_shapes;
  bool normalize_frames = false; // CREPE: zero mean, unit variance per frame
  std::vector<AnalyzerStatePort> states;
  std::vector<int64_t> ishape;
  std::vector<Ort::Value> ins, outs;
};

struct AnalyzerJob
{
  enum class Kind : uint8_t
  {
    Build,
    Dispose
  } kind = Kind::Build;
  AnalyzerBuildParams build;
  std::shared_ptr<Onnx::OnnxRunContext> ctx; // Build: reuse this session
  std::shared_ptr<AnalyzerPipeline> pipeline; // Dispose: its last owner
};

struct AudioAnalyzer : OnnxObject
{
public:
  halp_meta(name, "Audio Analyzer");
  halp_meta(c_name, "audio_analyzer");
  halp_meta(category, "AI/Audio Analysis");
  halp_meta(author, "ossia team");
  halp_meta(
      description,
      "Generic ONNX audio->data/control analyzer: pitch (CREPE), voice activity "
      "(Silero-VAD, recurrent), audio tagging (YAMNet) and embeddings (CLAP). "
      "Accumulates + resamples the input to the model rate, threads recurrent "
      "state internally, and exposes the result as a Data vector plus two scalar "
      "controls.");
  halp_meta(uuid, "c2d3e4f5-a6b7-4890-abcd-ef0123456789");

  struct
  {
    halp::dynamic_audio_bus<"In", float> audio;
    ModelPort<"Model"> model;
    halp::hslider_f32<"Param", halp::range{0., 1., 0.5}> param;
    halp::enum_t<AnalyzerReduce, "Reduce"> reduce;
    halp::impulse_button<"Reset"> reset;
    // Appended: the model's sample rate when Auto (0) guesses wrong.
    struct : halp::spinbox_i32<"Model Rate", halp::range{0, 192000, 0}>
    {
      halp_meta(description, "Sample rate the model runs at; 0 = auto");
    } model_rate;
  } inputs;

  struct
  {
    halp::val_port<"Data", std::vector<float>> data;
    halp::val_port<"Value 1", float> value1;
    halp::val_port<"Value 2", float> value2;
  } outputs;

  AudioAnalyzer() noexcept;
  ~AudioAnalyzer();

  void prepare(halp::setup info);
  void operator()(int frames);

  struct worker
  {
    std::function<void(std::unique_ptr<AnalyzerJob>)> request;
    static std::function<void(AudioAnalyzer&)> work(std::unique_ptr<AnalyzerJob> job);
  } worker;

private:
  std::shared_ptr<AnalyzerPipeline> pipe;
  bool building = false;
  AnalyzerBuildParams requested;

  double host_rate = 48000.0;
  int host_in_channels = 0;
  std::size_t max_frames = 4096;

  AnalyzerBuildParams currentParams() const;
  void requestBuild(const AnalyzerBuildParams& p, bool reuse_session);
  void install(std::shared_ptr<AnalyzerPipeline> p);
  void dispose(std::shared_ptr<AnalyzerPipeline> p);
  void zeroStates();
  void runBlock();
  void reduceOutputs(const std::vector<float>& flat, int primary_out);
};

}
