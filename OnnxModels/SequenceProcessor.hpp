#pragma once
#include <OnnxModels/Utils.hpp>

#include <Onnx/helpers/AuxInputs.hpp>
#include <Onnx/helpers/DataIO.hpp>
#include <Onnx/helpers/ModelArchetype.hpp>
#include <Onnx/helpers/ModelSpec.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <cstdint>
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
struct SequenceProcessor;

// Off-thread inference job for the sequence node (mirrors ImageProcessor's
// InferJob). Carries a preprocessed contiguous input buffer ([1,T,F] / [1,D]),
// its shape, the model's input dtype, and any recurrent-state buffers to feed
// back. ctx is shared so it outlives the node frame; readModelSpec() is re-read
// in work() for thread-stable name pointers.
//
// Jobs travel through score::TaskPool's 128-byte smallfun queue, so they are
// heap-allocated and recycled through the lock-free JobPool (only the
// unique_ptr crosses the queue; recycled vectors keep their capacity, so the
// steady-state request path does not allocate). See OnnxModels/JobPool.hpp.
// A recurrent state: an input fed from the output it is paired with.
struct SeqState
{
  int in_index = 0;
  int out_index = 0;
  std::vector<int64_t> shape;
  Onnx::TensorElemType dt = Onnx::TensorElemType::Float;
  std::vector<float> values;
};

// Staging for one inference, reused from run to run on the processing thread
// (tensors are non-owning views over these).
struct SeqScratch
{
  std::vector<int64_t> i64;
  std::vector<int32_t> i32;
  std::vector<double> f64;
  std::vector<std::vector<int64_t>> s_i64, f_i64;
  std::vector<std::vector<int32_t>> s_i32, f_i32;
  std::vector<std::vector<double>> s_f64, f_f64;
  std::vector<std::vector<float>> filler;
  std::vector<std::vector<uint8_t>> aux_store;
  std::vector<std::vector<int64_t>> aux_shape;
  std::vector<Ort::Value> ins, outs;
  std::vector<float> decode;
  std::vector<int64_t> oshape;
};

// A decoded result: the primary output (Out), an optional secondary Data
// output, and the new recurrent-state values.
struct SeqResult
{
  std::vector<float> out;
  std::vector<float> data;
  bool has_data = false;
  struct NewState
  {
    int in_index = 0;
    std::vector<float> values;
  };
  std::vector<NewState> states;
};

// What the node's model resolves to: the session, the I/O routing, the aux
// plans, the batch the probe found, the recurrent states, the window ring and
// the synchronous path's scratch. Built by a worker job (creating the session
// and running the probe inference there), never on the processing thread;
// the processing thread swaps it in and hands the old one back to the worker
// to be freed.
struct SeqPipeline
{
  std::string path;
  std::size_t model_bytes = 0;
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  Onnx::ModelSpec spec;
  Onnx::ModelArchetype arch;
  std::string refusal; // non-empty: the model does not run

  int primaryIn = 0;
  int primaryOut = 0;
  int dataOut = -1;
  bool stateful = false;
  std::vector<Onnx::AuxPlan> aux; // inputs other than the primary and states
  // Exports that bake a batch size into the graph (Informer: 2, while the
  // input declares it dynamic) are fed that many copies of the input.
  int64_t batch = 1;
  std::vector<SeqState> states;

  // Sized here when the model's T and F are known, so the processing thread
  // does not allocate it on the first frames.
  Onnx::FrameWindow window;
  std::vector<float> in_scratch;
  std::vector<float> input; // the primary input of the next run
  std::vector<int64_t> ishape;
  SeqScratch scratch;       // the synchronous path's
  SeqResult result;         // the synchronous path's
};

struct SeqInferJob
{
  // Infer runs the model; Build makes a pipeline for build_path; Dispose frees
  // `pipeline` here, off the processing thread.
  enum class Kind : uint8_t
  {
    Infer,
    Build,
    Dispose
  } kind = Kind::Infer;
  std::string build_path;
  std::size_t build_bytes = 0;
  std::shared_ptr<SeqPipeline> pipeline;

  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  std::vector<float> input;        // primary input, flattened [1,T,F]/[1,D]
  std::vector<int64_t> ishape;     // primary input shape (batch == 1)
  Onnx::TensorElemType in_dt = Onnx::TensorElemType::Float;
  int primary_in_index = 0;        // which model input is the data port
  int primary_out_index = 0;       // which model output is the data result
  int data_out_index = -1;         // secondary "Data" output (-1 == same as primary)
  int64_t batch = 1;               // > 1: the input is replicated, slice 0 read
  uint32_t gen = 0;                // the node's generation at dispatch

  // Recurrent state threaded internally: for each (input_index -> output_index)
  // pair, the current state values + shape. work() feeds `values` in and reads
  // the matching output back out into the returned applicator.
  using State = SeqState;
  std::vector<State> states;

  // The other inputs (Silero sr, scalar controls): planned at load, filled here.
  std::vector<Onnx::AuxPlan> aux;
  float params[2]{};
};

// How a model that declares a fixed time dimension is fed from the upstream.
enum class SeqWindowMode
{
  Auto,        // Sliding if the model needs a fixed T and payload is one frame
  Passthrough, // upstream already provides a full [T,F] block per tick
  Sliding,     // host buffers the last T frames
};

// Per-frame normalisation of the input (each F-vector of [1,T,F], or the
// whole [1,D] vector): embedding heads trained on unit-norm CLIP embeddings
// saturate on raw ones.
enum class SeqNormalize
{
  None,
  L2,
  ZScore,
};

struct SequenceProcessor : OnnxObject
{
public:
  halp_meta(name, "Sequence Processor");
  halp_meta(c_name, "sequence_processor");
  halp_meta(category, "AI/Data processing");
  halp_meta(author, "ossia team");
  halp_meta(
      description,
      "Generic ONNX sequence/vector processor: runs any [1,T,F] (time x "
      "features) or [1,D] vector model — forecasting, anomaly detection, "
      "gesture/IMU recognition, 2D->3D pose lifting, motion in-betweening, "
      "small RNN/transformer. Auto-detects layout and recurrent state; "
      "windows streaming input to a fixed T; threads hidden state internally.");
  halp_meta(uuid, "b2d4e6f8-0a1c-4e3d-9b8a-7c6d5e4f3a2b");

  struct
  {
    halp::val_port<"In", std::vector<float>> in;
    ModelPort<"Model"> model;
    halp::enum_t<SeqWindowMode, "Window"> window_mode;
    halp::hslider_f32<"Param 1", halp::range{-10., 10., 0.}> param1;
    halp::hslider_f32<"Param 2", halp::range{-10., 10., 0.}> param2;
    halp::impulse_button<"Reset"> reset;
    halp::enum_t<SeqNormalize, "Normalize"> normalize;
  } inputs;

  struct
  {
    halp::val_port<"Out", std::vector<float>> out;
    halp::val_port<"Data", std::vector<float>> data;
  } outputs;

  SequenceProcessor() noexcept;
  ~SequenceProcessor();

  void operator()();

  // Worker-thread inference (avendish pattern): heavy models run off the render
  // thread. Cheap ones stay inline.
  struct worker
  {
    std::function<void(std::unique_ptr<SeqInferJob>)> request;
    static std::function<void(SequenceProcessor&)>
    work(std::unique_ptr<SeqInferJob> job);
  } worker;

  // The running model's recurrent states (for tests; empty before a model).
  const std::vector<SeqState>& currentStates() const noexcept
  {
    static const std::vector<SeqState> none;
    return pipe ? pipe->states : none;
  }

  // Public for applyResult() (a free function in the .cpp, reached from the
  // worker completion lambda).
  std::shared_ptr<SeqPipeline> pipe;

private:
  bool building = false;
  std::string requested; // the file the last build was for
  bool inferenceInProgress = false;
  Onnx::WindowMode resolvedWindow = Onnx::WindowMode::Passthrough;
  SeqWindowMode lastWindowMode{};
  // Bumped by Reset and by a new model: a job dispatched before either must
  // not bring its state and output back.
  uint32_t gen = 0;

  void reportError(std::string_view what);
  void requestBuild();
  void install(std::shared_ptr<SeqPipeline> p);
  void dispose(std::shared_ptr<SeqPipeline> p);

  void resolveWindow();
  void resetState();
  void dispatchInfer(bool force_async);
};

}
