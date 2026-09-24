#pragma once
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
struct ImageProcessor;

// Off-thread inference job: a preprocessed input plus everything work() needs to
// run + decode without touching the node. ctx is shared so it outlives the node
// frame; readModelSpec() is re-read inside work() for thread-stable name ptrs.
//
// Jobs travel through score::TaskPool's 128-byte smallfun queue, so they are
// heap-allocated and recycled through the lock-free JobPool (only the
// unique_ptr crosses the queue; recycled vectors keep their capacity, so the
// steady-state request path does not allocate). See OnnxModels/JobPool.hpp.
struct InferJob
{
  uint32_t gen = 0; // the node's generation at dispatch
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  boost::container::vector<float> input;
  std::vector<int64_t> ishape;
  Onnx::TensorElemType in_dt = Onnx::TensorElemType::Float;
  int output_index = 0;
  Onnx::ImageModelKind kind = Onnx::ImageModelKind::ImageToImage;
  Onnx::WriteMode wm = Onnx::WriteMode::DirectClamp;
  // Auto role of every model output, to route the secondary ones.
  std::vector<Onnx::ImageModelKind> out_kinds;
};

// Pixel value normalization applied to the input image before inference.
// Shared image/video enums (InputNormalization, ChannelOrder, ResizeMode,
// TaskMode, OutputMode) live in one header to avoid duplicate definitions.

struct ImageProcessor : OnnxObject
{
public:
  halp_meta(name, "Image Processor");
  halp_meta(c_name, "image_processor");
  halp_meta(category, "AI/Image Processing");
  halp_meta(author, "ossia team");
  halp_meta(
      description,
      "Generic ONNX image processor: runs any image->image, image->mask/depth, "
      "or image->data model. Auto-detects layout (NCHW/NHWC), channels and "
      "element type; configurable normalization, resize and output mapping.");
  halp_meta(uuid, "f4a5b6c7-d8e9-0123-4567-89abcdef0123");

  struct
  {
    halp::fixed_texture_input<"In"> image;
    // Optional second image: bound to a model's 2nd image input (image-pair
    // models — inpainting img+mask, optical flow img0+img1, stereo L+R).
    halp::fixed_texture_input<"Aux"> aux;
    ModelPort<"Model"> model;
    halp::xy_spinboxes_i32<"Resolution", halp::range{1, 4096, 512}> resolution;
    halp::enum_t<InputNormalization, "Normalization"> normalization;
    halp::enum_t<ChannelOrder, "Channel Order"> channel_order;
    halp::enum_t<ResizeMode, "Resize"> resize_mode;
    halp::spinbox_i32<"Output Index", halp::range{0, 16, 0}> output_index;
    halp::enum_t<TaskMode, "Task"> task;
    // Value-range -> pixel mapping (Auto picks the right one per model). This is
    // NOT the PoseDetector "Output Mode" (skeleton vis); it rescales pixels.
    halp::enum_t<OutputMode, "Pixel Mapping"> output_mode;
    // Bound in order to a model's extra scalar inputs (timestep, strength, etc.).
    halp::hslider_f32<"Param 1", halp::range{0., 1., 0.}> param1;
    halp::hslider_f32<"Param 2", halp::range{0., 1., 0.}> param2;

    // Latent / generative controls — used only for latent->image models.
    halp::val_port<"Latent Vector", ossia::small_pod_vector<float, 16>> latent;
    halp::hslider_f32<"Latent Scale", halp::range{-10., 10., 1.}> latent_scale;
    halp::spinbox_i32<"Seed", halp::range{0, 999999, 42}> seed;
  } inputs;

  struct
  {
    halp::texture_output<"Image"> image;
    halp::texture_output<"Mask", halp::r8_texture> mask;
    halp::texture_output<"Depth", halp::r32f_texture> depth;
    halp::val_port<"Data", std::vector<float>> data;
  } outputs;

  ImageProcessor() noexcept;
  ~ImageProcessor();

  void operator()();

  // Worker-thread inference (avendish pattern): heavy models run off the render
  // thread so a slow inference can't freeze the graph. Cheap models stay inline.
  struct worker
  {
    std::function<void(std::unique_ptr<InferJob>)> request;
    static std::function<void(ImageProcessor&)>
    work(std::unique_ptr<InferJob> job);
  } worker;

private:
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  Onnx::ModelSpec spec;
  Onnx::ImageModelRole role;
  // Auto role of every model output (index = output index). The Output Index
  // output goes where Task says; the others fill the outlets it left free.
  std::vector<Onnx::ImageModelKind> out_kinds;
  std::string lastModelPath;
  int lastOutputIndex = -1;
  bool inferenceInProgress = false;
  // Bumped by a model reload: a job of the previous model finishing later
  // must not publish its result (nor, for Auto, its mapping).
  uint32_t gen = 0;

  // Input preprocessing buffer (float), reused across frames; the dtype-specific
  // staging buffers are only used for fp16/uint8-input models.
  boost::container::vector<float> storage;
  std::vector<uint16_t> half_buf;
  std::vector<uint8_t> u8_buf;
  // Output dtype->float scratch (only used when a result is not fp32).
  std::vector<float> out_scratch;

  // Multi-input (image-pair / parametric) support. Roles are resolved once per
  // model from classifyModel; single-input models leave multi_input=false and
  // take the unchanged fast path.
  Onnx::ModelArchetype march;
  bool multi_input = false;
  int image_input_index = 0;
  bool has_image_input = false; // no image input: a latent -> image model
  int aux_input_index = -1;
  int param_in[2] = {-1, -1};
  boost::container::vector<float> aux_storage;
  std::vector<uint16_t> aux_half_buf;
  std::vector<uint8_t> aux_u8_buf;

  // Latent->image state.
  std::vector<float> latent_vector;
  std::vector<float> last_latent;
  bool produced_latent = false;

  void reloadModel();
  void runImage();
  void runImageMulti();
  void runLatent();
  void dispatchInfer(
      std::vector<int64_t> ishape, Onnx::ImageModelKind kind,
      Onnx::WriteMode wm, bool force_async);
};

}
