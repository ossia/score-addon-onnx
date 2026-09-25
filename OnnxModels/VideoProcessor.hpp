#pragma once
// VIDEO PROCESSOR — streaming / recurrent image models (e.g. Robust Video
// Matting). Structurally this is ImageProcessor + INTERNAL RECURRENT STATE + a
// Reset: models like RVM carry recurrent state tensors (r1i..r4i / r1o..r4o)
// that the host must allocate, zero-init, and feed back frame-to-frame.
//
// Mechanism (see VideoProcessor.cpp): after readModelSpec() we run classifyModel
// (ModelArchetype) to find which INPUTS are RecurrentState and which OUTPUT each
// maps to (by shape; RVM also uses r1i/r1o naming). A zero-initialized buffer is
// held per state input (dtype + element count from the declared shape, or sized
// from the first inference's concrete output shapes when dims are dynamic). Each
// frame: feed the held state buffers, run, copy the matching state OUTPUTS back.
// The Reset impulse re-zeros all state. The primary image input + image/mask/
// depth outputs use the normal ImageProcessor paths (ImageOps + TensorToTexture).
#include <OnnxModels/ImageEnums.hpp>
#include <OnnxModels/Utils.hpp>

#include <ossia/detail/pod_vector.hpp>
#include <ossia/detail/small_vector.hpp>

#include <Onnx/helpers/ImageModelRole.hpp>
#include <Onnx/helpers/ModelArchetype.hpp>
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/TensorToTexture.hpp>

#include <boost/container/vector.hpp>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

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
struct VideoProcessor;

// A single recurrent-state slot: an input port fed each frame, paired with the
// output port whose value replaces it after inference. The buffer holds the
// running state (dtype-typed bytes); zeroed on construction / Reset.
struct StateSlot
{
  int input_index = -1;        // index into spec.inputs
  int output_index = -1;       // index into spec.outputs (matching state out)
  Onnx::TensorElemType dtype = Onnx::TensorElemType::Float;
  std::vector<int64_t> shape;  // concrete shape (resolved; dynamic dims -> 1)
  int64_t count = 0;           // element count (product of shape)
  std::vector<uint8_t> buffer; // count * elemSize(dtype) bytes, zero-init
};

// Off-thread inference job: a preprocessed image input + the held recurrent
// state buffers, plus everything work() needs to run and decode without touching
// the node. ctx is shared so it outlives the node frame.
//
// Jobs travel through score::TaskPool's 128-byte smallfun queue, so they are
// heap-allocated and recycled through the lock-free JobPool (only the
// unique_ptr crosses the queue; recycled vectors keep their capacity, so the
// steady-state request path does not allocate). See OnnxModels/JobPool.hpp.
struct VideoInferJob
{
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  boost::container::vector<float> input; // preprocessed primary image
  std::vector<int64_t> ishape;
  int image_input_index = 0;
  Onnx::TensorElemType in_dt = Onnx::TensorElemType::Float;
  int output_index = 0; // primary visual output to decode
  Onnx::ImageModelKind kind = Onnx::ImageModelKind::ImageToImage;
  Onnx::WriteMode wm = Onnx::WriteMode::DirectClamp;
  float param0 = 0.f, param1 = 0.f; // scalar params (e.g. downsample_ratio)
  int param0_input_index = -1, param1_input_index = -1;
  std::vector<StateSlot> states; // copied in (held state -> fed); copied back out
  std::vector<Onnx::ImageModelKind> out_kinds; // role of each output
  uint32_t gen = 0; // the node's generation at dispatch
};

// (Identical knobs to ImageProcessor: see its header for rationale.)
// Shared image/video enums live in OnnxModels/ImageEnums.hpp (included below).

struct VideoProcessor : OnnxObject
{
public:
  halp_meta(name, "Video Processor");
  halp_meta(c_name, "video_processor");
  halp_meta(category, "AI/Image Processing");
  halp_meta(author, "ossia team");
  halp_meta(
      description,
      "Generic streaming/recurrent ONNX video processor: runs image->image / "
      "mask / depth models that carry internal recurrent state across frames "
      "(e.g. Robust Video Matting r1..r4). The host allocates, zero-inits and "
      "feeds the state tensors back frame-to-frame; Reset re-zeros them.");
  halp_meta(uuid, "b7c8d9e0-1234-5678-9abc-def012345678");

  struct
  {
    halp::fixed_texture_input<"In"> image;
    ModelPort<"Model"> model;

    // Two generic scalar params, mapped to the model's scalar inputs in order
    // (RVM: param0 -> downsample_ratio; the second is spare for other models).
    halp::hslider_f32<"Param 1", halp::range{0., 1., 0.25}> param0;
    halp::hslider_f32<"Param 2", halp::range{0., 1., 1.0}> param1;

    halp::impulse_button<"Reset"> reset;

    // Image config (mirrors ImageProcessor).
    halp::xy_spinboxes_i32<"Resolution", halp::range{1, 4096, 512}> resolution;
    halp::enum_t<InputNormalization, "Normalization"> normalization;
    halp::enum_t<ChannelOrder, "Channel Order"> channel_order;
    halp::enum_t<ResizeMode, "Resize"> resize_mode;
    halp::spinbox_i32<"Output Index", halp::range{0, 16, 0}> output_index;
    halp::enum_t<TaskMode, "Task"> task;
    // Value-range -> pixel mapping (Auto picks the right one per model). Not the
    // PoseDetector "Output Mode" (skeleton vis); it rescales pixel values.
    halp::enum_t<OutputMode, "Pixel Mapping"> output_mode;
  } inputs;

  struct
  {
    halp::texture_output<"Image"> image;             // e.g. fgr
    halp::texture_output<"Mask", halp::r8_texture> mask;   // e.g. pha / alpha
    halp::texture_output<"Depth", halp::r32f_texture> depth;
    halp::val_port<"Data", std::vector<float>> data;
  } outputs;

  VideoProcessor() noexcept;
  ~VideoProcessor();

  void operator()();

  // Worker-thread inference (avendish pattern): heavy recurrent models run off
  // the render thread. Strict per-frame ordering + latest-wins (drop frames
  // while a job is in flight) keeps the recurrent state coherent.
  struct worker
  {
    std::function<void(std::unique_ptr<VideoInferJob>)> request;
    static std::function<void(VideoProcessor&)>
    work(std::unique_ptr<VideoInferJob> job);
  } worker;

private:
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  Onnx::ModelSpec spec;
  Onnx::ImageModelRole role;
  Onnx::ModelArchetype arch;
  std::string lastModelPath;
  int lastOutputIndex = -1;
  bool inferenceInProgress = false;

  // Primary image input port index (the non-state, non-scalar image input).
  int imageInputIndex = 0;
  // Scalar param input indices (downsample_ratio etc.), in declaration order.
  int param0InputIndex = -1;
  int param1InputIndex = -1;

  // Recurrent state slots (one per RecurrentState input), held across frames.
  std::vector<StateSlot> states;
  // The buffers of the states before the last result, handed to the next
  // job so its snapshot reuses their capacity (see worker::work).
  std::shared_ptr<std::vector<StateSlot>> spareStates;
  // classifyImage's kind for every output, to route the non-primary ones.
  std::vector<Onnx::ImageModelKind> out_kinds;
  // Reset and model changes bump gen: a job dispatched before does not bring
  // its states back. A Reset during a job waits for it (resetPending).
  uint32_t gen = 0;
  bool resetPending = false;
  // Set when a synchronous run took longer than a frame's budget: the model
  // then runs on the worker until the model or its geometry changes.
  bool preferAsync = false;
  // The geometry the carried recurrent state was produced at. RVM-style models
  // size their state by (input WxH, downsample_ratio); feeding a state from a
  // different geometry makes an internal Expand fail to broadcast (crash). When
  // any of these change we must re-zero the state so the model re-inits it.
  int stateGeomW = -1, stateGeomH = -1;
  float stateGeomP0 = 1e30f, stateGeomP1 = 1e30f;

  // Input preprocessing buffer (mirrors ImageProcessor).
  boost::container::vector<float> storage;
  std::vector<uint16_t> half_buf;
  std::vector<uint8_t> u8_buf;
  std::vector<float> out_scratch;

  void reloadModel();
  void detectStates();
  void zeroStates();
  void runImage();
  void dispatchInfer(
      std::vector<int64_t> ishape, Onnx::ImageModelKind kind, Onnx::WriteMode wm,
      bool force_async);
};

}
