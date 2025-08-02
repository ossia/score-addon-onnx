#pragma once
#include <OnnxModels/Utils.hpp>
#include <cmath>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

namespace Onnx
{
class ImageTranslationGAN;
class GANFactory;
}

namespace OnnxModels
{

struct ImageToImageGAN : OnnxObject
{
public:
  halp_meta(name, "Image-to-Image GAN");
  halp_meta(c_name, "image_to_image_gan");
  halp_meta(category, "AI/Image Processing");
  halp_meta(author, "Image-to-Image GAN authors, Onnxruntime");
  halp_meta(description, "Transform images using GAN models like AnimeGANv3, Fast-SRGAN, DeblurGAN.");
  halp_meta(uuid, "b2c3d4e5-f6a7-8901-bcde-f23456789012");

  enum ModelType
  {
    AnimeGANv3,
    FastSRGAN,
    DeblurGANv2
  };

  struct
  {
    struct : halp::texture_input<"Input Image">
    {
      // Request computation when image changes
      void update(ImageToImageGAN& g)
      {
        if (g.available && g.translation_model)
        {
          g.requestInference();
        }
      }
    } input_image;

    halp::enum_t<ModelType, "ModelType"> model_type;
    ModelPort model;

  } inputs;

  struct
  {
    halp::texture_output<"Transformed Image"> image;
  } outputs;

  ImageToImageGAN() noexcept;
  ~ImageToImageGAN();

  void operator()();

  // Worker thread infrastructure
  struct worker
  {
    std::function<void(
        QImage,
        std::shared_ptr<Onnx::ImageTranslationGAN>)>
        request;

    static std::function<void(ImageToImageGAN&)> work(
        QImage input_image,
        std::shared_ptr<Onnx::ImageTranslationGAN> translation_model);
  } worker;

private:
  std::shared_ptr<Onnx::ImageTranslationGAN> translation_model;
  
  std::string lastModelPath;
  ModelType lastModelType;
  
  bool inferenceInProgress = false;
  
  void createModelFromFile();
  bool needsReinitialization() const;
  void requestInference();
};
}
