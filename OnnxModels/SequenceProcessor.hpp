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
struct SeqInferJob
{
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  std::vector<float> input;        // primary input, flattened [1,T,F]/[1,D]
  std::vector<int64_t> ishape;     // primary input shape (batch == 1)
  Onnx::TensorElemType in_dt = Onnx::TensorElemType::Float;
  int primary_in_index = 0;        // which model input is the data port
  int primary_out_index = 0;       // which model output is the data result
  int data_out_index = -1;         // secondary "Data" output (-1 == same as primary)
  int64_t batch = 1;               // > 1: the input is replicated, slice 0 read

  // Recurrent state threaded internally: for each (input_index -> output_index)
  // pair, the current state values + shape. work() feeds `values` in and reads
  // the matching output back out into the returned applicator.
  struct State
  {
    int in_index = 0;
    int out_index = 0;
    std::vector<int64_t> shape;
    Onnx::TensorElemType dt = Onnx::TensorElemType::Float;
    std::vector<float> values;
  };
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

  // Recurrent-state buffers, persistent across ticks (zero-init; Reset
  // re-zeros). Public: applyResult() (a free function in the .cpp, reached from
  // the worker completion lambda) swaps fresh values back in.
  struct StateBuf
  {
    int in_index = 0;
    int out_index = 0;
    std::vector<int64_t> shape;
    Onnx::TensorElemType dt = Onnx::TensorElemType::Float;
    std::vector<float> values;
  };
  std::vector<StateBuf> states;

private:
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  Onnx::ModelSpec spec;
  Onnx::ModelArchetype arch;
  std::string lastModelPath;
  bool inferenceInProgress = false;

  // Resolved I/O routing (computed at reload from the archetype).
  int primaryIn = 0;
  int primaryOut = 0;
  int dataOut = -1;
  bool stateful = false;
  Onnx::WindowMode resolvedWindow = Onnx::WindowMode::Passthrough;
  SeqWindowMode lastWindowMode{};
  std::vector<Onnx::AuxPlan> aux; // inputs other than the primary and states
  // Exports that bake a batch size into the graph (Informer: 2, while the
  // input declares it dynamic) are fed that many copies of the input.
  int64_t batch = 1;
  std::string lastError; // printed once, not on every failing tick

  void reportError(std::string_view what);
  bool probe();

  // Hot-path scratch (reused).
  Onnx::FrameWindow window;
  std::vector<float> in_scratch;

  void reloadModel();
  void resolveWindow();
  void resetState();
  void dispatchInfer(
      std::vector<float> input, std::vector<int64_t> ishape, bool force_async);
};

}
