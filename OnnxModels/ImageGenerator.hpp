#pragma once
// Generic latent -> image generator for the specialized ONNX node set (the
// successor to GenerativeImageGAN). Drives the StyleGAN-style two-stage chain:
//
//   z -> [optional mapping net] -> w/style -> [synthesis/main net] -> image
//
// Unlike GenerativeImageGAN this carries NO per-architecture hardcoding (no fixed
// truncation_psi reshape, no FBAnime [1,16,1024] reshape): the two networks are
// chained generically by feeding the mapping output 0 straight into the synthesis
// input 0 (flat sizes must match). Structurally it mirrors ImageProcessor's
// runLatent() path: seed-driven latent build + user Latent overlay + Scale,
// vec_to_tensor for the [1,Z] input, TensorToTexture writeRgb decode, and an async
// worker so a heavy generation never freezes the render thread.
#include <OnnxModels/Utils.hpp>

#include <ossia/detail/pod_vector.hpp>
#include <ossia/detail/small_vector.hpp>

#include <Onnx/helpers/AuxInputs.hpp>
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/TensorToTexture.hpp>
#include <Onnx/helpers/TensorType.hpp>

#include <boost/container/vector.hpp>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Onnx
{
struct OnnxRunContext;
}

namespace OnnxModels
{
struct ImageGenerator;

// Output value -> pixel mapping. Auto == MinMaxNormalize, which fits the
// dynamic-range GANs (MobileStyleGAN, EigenGAN) the orchestrator targets; the
// explicit modes mirror the per-architecture denormalizations from GAN.cpp.
enum class GenOutputMode
{
  Auto,            // -> MinMaxNormalize (dynamic-range GANs)
  MinMaxNormalize, // (x-min)/(max-min)*255
  Denormalize,     // (x+1)*127.5            — [-1,1] outputs (classic StyleGAN)
  Half255,         // clamp(0.5+255*x,0,255) — PyTorchGAN
  DirectClamp,     // clamp(x,0,1)*255       — [0,1] outputs
};

// Off-thread generation job: carries BOTH context pointers (mapping is optional)
// plus the pre-built z buffer and everything work() needs to chain + decode
// without touching the node. ctx pointers are shared so they outlive the frame.
//
// Jobs travel through score::TaskPool's 128-byte smallfun queue, so they are
// heap-allocated and recycled through the lock-free JobPool (only the
// unique_ptr crosses the queue; recycled vectors keep their capacity, so the
// steady-state request path does not allocate). See OnnxModels/JobPool.hpp.
struct GenJob
{
  std::shared_ptr<Onnx::OnnxRunContext> synth_ctx; // synthesis / main generator
  std::shared_ptr<Onnx::OnnxRunContext> map_ctx;   // optional mapping network
  boost::container::vector<float> z;               // seeded + overlaid + scaled
  std::vector<int64_t> z_shape;                    // declared shape of synth/map in 0
  Onnx::TensorElemType synth_in_dt = Onnx::TensorElemType::Float;
  Onnx::TensorElemType map_in_dt = Onnx::TensorElemType::Float;
  int output_index = 0;
  Onnx::WriteMode wm = Onnx::WriteMode::MinMaxNormalize;
  bool pick_auto = false; // Auto: choose from this output's range

  // The inputs after input 0 of each stage (EigenGAN's z_*): planned at load.
  std::vector<Onnx::AuxPlan> synth_aux, map_aux;
  uint32_t seed = 0;
  float scale = 1.f;
  float params[2]{};
};

struct ImageGenerator : OnnxObject
{
public:
  halp_meta(name, "Image Generator");
  halp_meta(c_name, "image_generator");
  halp_meta(category, "AI/Generative");
  halp_meta(author, "ossia team");
  halp_meta(
      description,
      "Generic ONNX image generator: latent -> image, with an optional mapping "
      "network for the StyleGAN-style z -> mapping -> w -> synthesis -> image "
      "chain. Auto-detects the latent size and output denormalization; "
      "dtype-aware (fp16) and async (generative models are heavy).");
  halp_meta(uuid, "b2d3e4f5-a6c7-4890-1234-567890abcdef");

  struct
  {
    // The synthesis / main generator. Its first input's flat non-batch size is
    // the latent size we build z from (or the w size the mapping must produce).
    ModelPort<"Model"> model;
    // OPTIONAL mapping net: when loaded, z is fed through it and its output 0
    // (= w / style) is fed to the synthesis model instead of z.
    ModelPort<"Mapping model"> mapping_model;

    halp::val_port<"Latent", ossia::small_pod_vector<float, 16>> latent;
    halp::spinbox_i32<"Seed", halp::range{0, 999999, 42}> seed;
    halp::hslider_f32<"Scale", halp::range{-10., 10., 1.}> scale;

    // Generic extra controls (class label, truncation psi, ...). They overlay the
    // tail of the latent / are appended as additional scalar params; harmless when
    // the model has no second input.
    halp::hslider_f32<"Param 1", halp::range{-10., 10., 0.}> param1;
    halp::hslider_f32<"Param 2", halp::range{-10., 10., 0.}> param2;

    halp::enum_t<GenOutputMode, "Output Mode"> output_mode;
  } inputs;

  struct
  {
    halp::texture_output<"Image"> image;
    halp::val_port<"Data", std::vector<float>> data;
  } outputs;

  ImageGenerator() noexcept;
  ~ImageGenerator();

  void operator()();

  // Worker-thread generation (avendish pattern): generative models are heavy, so
  // the chain runs off the render thread. Latest-wins: while a job is in flight
  // new frames are dropped.
  struct worker
  {
    std::function<void(std::unique_ptr<GenJob>)> request;
    static std::function<void(ImageGenerator&)> work(std::unique_ptr<GenJob> job);
  } worker;

  // --- standalone-testable core (no ORT / Qt / ossia deps) ------------------

  // Build z: seeded normal RNG, overlaid with the user Latent dims, then * Scale.
  // Defined here so the test can exercise it without the node infrastructure.
  static void buildLatent(
      std::vector<float>& z, int dim, uint32_t seed, const float* user,
      int user_n, float scale);

  // Mapping-out flat size must equal synthesis-in flat size for the generic chain
  // (no reshape, no truncation hardcoding). Treats dynamic (<=0) dims as 1.
  static bool chainCompatible(
      const std::vector<int64_t>& map_out, const std::vector<int64_t>& synth_in);

private:
  std::shared_ptr<Onnx::OnnxRunContext> ctx;     // synthesis / main
  std::shared_ptr<Onnx::OnnxRunContext> map_ctx; // optional mapping
  Onnx::ModelSpec spec;                          // synthesis spec
  Onnx::ModelSpec map_spec;                      // mapping spec (when loaded)
  std::string lastModelPath;
  std::string lastMappingPath;
  int latent_dim = 0;
  bool inferenceInProgress = false;
  std::vector<Onnx::AuxPlan> synth_aux, map_aux; // inputs after input 0
  std::optional<Onnx::WriteMode> autoMode; // Auto's choice for this model

  // z build state (re-infer only when the effective latent changed).
  std::vector<float> z_vector;
  std::vector<float> last_z;
  bool produced = false;

  void reloadModel();
  void runGenerate();
};

}
