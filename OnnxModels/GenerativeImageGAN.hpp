#pragma once
#include <OnnxModels/Utils.hpp>
#include <cmath>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

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
    halp::enum_t<ModelType, "ModelType"> model_type;
    ModelPort mapping_model;
    ModelPort synthesis_model;

    // Latent space controls - 512 dimensions for StyleGAN
    halp::hslider_f32<"Z0", latent_range> z0;
    halp::hslider_f32<"Z1", latent_range> z1;
    halp::hslider_f32<"Z2", latent_range> z2;
    halp::hslider_f32<"Z3", latent_range> z3;
    halp::hslider_f32<"Z4", latent_range> z4;
    halp::hslider_f32<"Z5", latent_range> z5;
    halp::hslider_f32<"Z6", latent_range> z6;
    halp::hslider_f32<"Z7", latent_range> z7;
    halp::hslider_f32<"Z8", latent_range> z8;
    halp::hslider_f32<"Z9", latent_range> z9;
    halp::hslider_f32<"Z10", latent_range> z10;
    halp::hslider_f32<"Z11", latent_range> z11;
    halp::hslider_f32<"Z12", latent_range> z12;
    halp::hslider_f32<"Z13", latent_range> z13;
    halp::hslider_f32<"Z14", latent_range> z14;
    halp::hslider_f32<"Z15", latent_range> z15;

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
