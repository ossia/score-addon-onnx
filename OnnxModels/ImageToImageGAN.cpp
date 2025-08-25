#include "ImageToImageGAN.hpp"

#include <QImage>

#include <Onnx/helpers/GAN.hpp>

namespace OnnxModels
{

ImageToImageGAN::ImageToImageGAN() noexcept
  : lastModelType(ModelType::AnimeGANv3)
{
}

ImageToImageGAN::~ImageToImageGAN() = default;

bool ImageToImageGAN::needsReinitialization() const
{
  return !translation_model 
         || lastModelPath != inputs.model.file.filename
         || lastModelType != inputs.model_type.value;
}

void ImageToImageGAN::createModelFromFile()
{
  if (inputs.model.file.filename.empty())
    return;

  try {
    std::vector<std::string_view> model_paths = {inputs.model.file.bytes};

    // Get appropriate config based on selected model type
    Onnx::GANConfig config;
    switch (inputs.model_type.value) {
      case ModelType::AnimeGANv3:
        config = Onnx::GANFactory::getAnimeGANv3Config(model_paths);
        break;
      case ModelType::FastSRGAN:
        config = Onnx::GANFactory::getFastSRGANConfig(model_paths);
        break;
      case ModelType::DeblurGANv2:
        config = Onnx::GANFactory::getDeblurGANv2Config(model_paths);
        break;
      default:
        config = Onnx::GANFactory::getAnimeGANv3Config(model_paths);
        break;
    }
    
    // Create image translation model
    translation_model = std::make_shared<Onnx::ImageTranslationGAN>(config);
    
    // Update last known paths
    lastModelPath = inputs.model.file.filename;
    lastModelType = inputs.model_type.value;
    
  } catch (const std::exception& e) {
    // Model creation failed
    translation_model.reset();
  }
}

void ImageToImageGAN::requestInference()
{
  // Don't start new inference if one is already in progress
  if (inferenceInProgress)
    return;

  auto& in_tex = inputs.input_image.texture;
  if (!in_tex.bytes || in_tex.width <= 0 || in_tex.height <= 0)
    return;

  // Convert texture to QImage
  QImage input_image(
      reinterpret_cast<const uchar*>(in_tex.bytes),
      in_tex.width,
      in_tex.height,
      QImage::Format_RGBA8888);

  if (input_image.isNull())
    return;

  // Mark inference as in progress
  inferenceInProgress = true;

  // Start worker thread computation
  worker.request(input_image, translation_model);
}

void ImageToImageGAN::operator()()
{
  if (!available)
    return;

  if (needsReinitialization())
  {
    createModelFromFile();
  }

  if (!translation_model)
    return;

  // The actual processing happens in the worker thread
  // This operator() just handles initialization and model management
  if (!inferenceInProgress)
  {
    requestInference();
  }
}

// Worker thread implementation
std::function<void(ImageToImageGAN&)> ImageToImageGAN::worker::work(
    QImage input_image,
    std::shared_ptr<Onnx::ImageTranslationGAN> translation_model)
{
  if (input_image.isNull() || !translation_model)
  {
    return [](ImageToImageGAN& node)
    {
      node.inferenceInProgress = false;
    };
  }

  try
  {
    QImage result;

    if (translation_model->isReady())
    {
      result = translation_model->transformImage(input_image);
    }

    if (!result.isNull())
    {
      // Convert QImage to RGBA format for ossia
      QImage rgba_image = result.convertToFormat(QImage::Format_RGBA8888);
      
      // Return a function that will be executed in the main thread
      return [rgba_image = std::move(rgba_image)](ImageToImageGAN& node) mutable
      {
        node.inferenceInProgress = false;
        
        node.outputs.image.create(rgba_image.width(), rgba_image.height());
        memcpy(
            node.outputs.image.texture.bytes,
            rgba_image.constBits(),
            rgba_image.width() * rgba_image.height() * 4);
        node.outputs.image.texture.changed = true;
      };
    }
    else
    {
      return [](ImageToImageGAN& node)
      {
        node.inferenceInProgress = false;
      };
    }
  }
  catch (const std::exception& e)
  {
    return [](ImageToImageGAN& node)
    {
      node.inferenceInProgress = false;
      // Keep previous output on error
    };
  }
  catch (...)
  {
    return [](ImageToImageGAN& node)
    {
      node.inferenceInProgress = false;
    };
  }
}

}
