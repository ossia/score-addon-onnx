#pragma once
#include <ossia/detail/pod_vector.hpp>
#include <ossia/detail/small_vector.hpp>

#include <OnnxModels/Utils.hpp>
#include <cmath>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <array>

namespace Onnx
{
class StyleGANModel;
class SingleNetworkGAN;
class GANFactory;
}

namespace OnnxModels
{

static constexpr halp::range latent_range = halp::range{-1.0, 1.0, 0.0};
struct GenerativeImageGAN : OnnxObject
{
public:
  halp_meta(name, "Generative Image GAN");
  halp_meta(c_name, "generative_image_gan");
  halp_meta(category, "AI/Generative");
  halp_meta(author, "StyleGAN authors, Onnxruntime");
  halp_meta(description, "Generate images from latent space using GAN models like StyleGAN.");
  halp_meta(uuid, "68c5c370-a9b8-4d97-8dc8-d370470a0d1c");

  enum ModelType
  {
    FBAnime,
    EigenGAN,
    MobileStyleGAN,
    PyTorchGAN
  };

  struct
  {
    halp::enum_t<ModelType, "Model Type"> model_type;
    ModelPort<"Mapping model"> mapping_model;
    ModelPort<"Synthesis model"> synthesis_model;

    halp::val_port<"Latent Vector", ossia::small_pod_vector<float, 16>>
        latent_dims;

    halp::hslider_f32<"Scale", halp::range{-10., 10., 1.}> scale;

    // Seed for random generation
    struct : halp::spinbox_i32<"Seed", halp::range{0, 999999, 42}>
    {
      void update(GenerativeImageGAN& self) { self.randomizeLatent(); }
    } seed;
  } inputs;

  struct
  {
    halp::texture_output<"Generated Image"> image;
  } outputs;

  GenerativeImageGAN() noexcept;
  ~GenerativeImageGAN();

  void operator()();

  // Worker thread infrastructure
  struct worker
  {
    std::function<void(
        std::vector<float>,
        std::shared_ptr<Onnx::StyleGANModel>,
        std::shared_ptr<Onnx::SingleNetworkGAN>,
        std::string)>
        request;

    // Called back in a worker thread
    // The returned function will be later applied in this object's processing thread
    static std::function<void(GenerativeImageGAN&)> work(
        std::vector<float> latent_vector,
        std::shared_ptr<Onnx::StyleGANModel> stylegan_model,
        std::shared_ptr<Onnx::SingleNetworkGAN> singlenet_model,
        std::string model_type);
  } worker;

private:
  std::shared_ptr<Onnx::StyleGANModel> stylegan_model;
  std::shared_ptr<Onnx::SingleNetworkGAN> singlenet_model;
  std::vector<float> latent_vector;
  std::vector<float> scaled_latent_vector;
  std::string current_model_type;
  
  std::string lastMappingModelPath;
  std::string lastSynthesisModelPath;
  ModelType lastModelType;
  
  bool inferenceInProgress = false;
  
  void initializeLatentVector();
  void updateLatentFromControls();
  void randomizeLatent();
  void createModelFromFile();
  bool needsReinitialization() const;
  void requestInference();
};
}
