#include "GenerativeImageGAN.hpp"

#include <QImage>

#include <Onnx/helpers/GAN.hpp>

#include <random>

namespace OnnxModels
{

GenerativeImageGAN::GenerativeImageGAN() noexcept
  : lastModelType(ModelType::MobileStyleGAN)
{
  // Initialize with 512-dimensional latent vector (standard for StyleGAN)
  latent_vector.resize(512, 0.0f);
  scaled_latent_vector.resize(512, 0.0f);

  initializeLatentVector();
}

GenerativeImageGAN::~GenerativeImageGAN() = default;

void GenerativeImageGAN::initializeLatentVector()
{
  // Initialize with small random values
  std::mt19937 gen(inputs.seed.value);
  std::normal_distribution<float> dist(-1.0f, 1.0f);

  for (auto& val : latent_vector)
  {
    val = dist(gen);
  }
}

void GenerativeImageGAN::updateLatentFromControls()
{
  // Update first 16 dimensions from array input
  const int N
      = std::min(latent_vector.size(), inputs.latent_dims.value.size());

  for (int i = 0; i < N; ++i)
  {
    latent_vector[i] = inputs.latent_dims.value[i];
  }
}

void GenerativeImageGAN::randomizeLatent()
{
  std::random_device rd;
  std::mt19937 gen(inputs.seed.value);
  std::normal_distribution<float> dist(-1.0f, 1.0f);

  for (auto& val : latent_vector)
  {
    val = dist(gen);
  }

  // Update the array input with the first 16 dimensions
  if (latent_vector.size() >= 16)
  {
    for (size_t i = 0; i < 16; ++i) {
      inputs.latent_dims.value[i] = latent_vector[i];
    }
  }
}

bool GenerativeImageGAN::needsReinitialization() const
{
  return (!stylegan_model && !singlenet_model) 
         || lastMappingModelPath != inputs.mapping_model.file.filename
         || lastSynthesisModelPath != inputs.synthesis_model.file.filename
         || lastModelType != inputs.model_type.value;
}

void GenerativeImageGAN::createModelFromFile()
{
  if (inputs.synthesis_model.file.filename.empty())
    return;

  try {
    std::vector<std::string> model_paths;
    if (!inputs.mapping_model.file.filename.empty())
      model_paths.push_back(std::string(inputs.mapping_model.file.filename));
    model_paths.push_back(std::string(inputs.synthesis_model.file.filename));

    // Get appropriate config based on selected model type
    Onnx::GANConfig config;
    switch (inputs.model_type.value) {
      case ModelType::FBAnime:
        config = Onnx::GANFactory::getFBAnimeConfig(model_paths);
        break;
      case ModelType::EigenGAN:
        config = Onnx::GANFactory::getEigenGANConfig(model_paths);
        break;
      case ModelType::MobileStyleGAN:
        config = Onnx::GANFactory::getMobileStyleGANConfig(model_paths);
        break;
      case ModelType::PyTorchGAN:
        config = Onnx::GANFactory::getPyTorchGANConfig(model_paths);
        break;
      default:
        config = Onnx::GANFactory::getMobileStyleGANConfig(model_paths);
        break;
    }
    
    // Create appropriate model based on config
    if (config.requires_mapping_network) {
      // StyleGAN-like model with mapping network
      stylegan_model = std::make_shared<Onnx::StyleGANModel>(config);
      singlenet_model.reset();
      current_model_type = "stylegan";
      
      // Resize latent vector to match model's requirement
      size_t required_size = stylegan_model->getLatentSize();
      if (required_size > 0) {
        latent_vector.resize(required_size);
        scaled_latent_vector.resize(required_size);
        initializeLatentVector();
      }
    } else {
      // Single network GAN
      singlenet_model = std::make_shared<Onnx::SingleNetworkGAN>(config);
      stylegan_model.reset();
      current_model_type = "singlenet";
      
      // Resize latent vector to match model's requirement
      size_t required_size = singlenet_model->getLatentSize();
      if (required_size > 0) {
        latent_vector.resize(required_size);
        scaled_latent_vector.resize(required_size);
        initializeLatentVector();
      }
    }
    
    // Update last known paths
    lastMappingModelPath = inputs.mapping_model.file.filename;
    lastSynthesisModelPath = inputs.synthesis_model.file.filename;
    lastModelType = inputs.model_type.value;
    
  } catch (const std::exception& e) {
    // Model creation failed
    stylegan_model.reset();
    singlenet_model.reset();
    current_model_type.clear();
  }
}

void GenerativeImageGAN::requestInference()
{
  // Don't start new inference if one is already in progress
  if (inferenceInProgress)
    return;

  // Update latent vector from controls and apply scaling
  updateLatentFromControls();
  scaled_latent_vector = latent_vector;
  for (int i = 0; i < 16 && i < static_cast<int>(scaled_latent_vector.size()); i++)
    scaled_latent_vector[i] = latent_vector[i] * inputs.scale.value;

  // Mark inference as in progress
  inferenceInProgress = true;

  // Start worker thread computation
  worker.request(scaled_latent_vector, stylegan_model, singlenet_model, current_model_type);
}

void GenerativeImageGAN::operator()()
{
  if (!available)
    return;

  if (needsReinitialization())
  {
    createModelFromFile();
  }

  if (!stylegan_model && !singlenet_model)
    return;

  // The actual processing happens in the worker thread
  // This operator() just handles initialization and model management
  if (!inferenceInProgress)
  {
    requestInference();
  }
}

// Worker thread implementation
std::function<void(GenerativeImageGAN&)> GenerativeImageGAN::worker::work(
    std::vector<float> latent_vector,
    std::shared_ptr<Onnx::StyleGANModel> stylegan_model,
    std::shared_ptr<Onnx::SingleNetworkGAN> singlenet_model,
    std::string model_type)
{
  if (latent_vector.empty() || (!stylegan_model && !singlenet_model))
  {
    return [](GenerativeImageGAN& node)
    {
      node.inferenceInProgress = false;
    };
  }

  try
  {
    QImage result;

    if (model_type == "stylegan" && stylegan_model && stylegan_model->isReady())
    {
      result = stylegan_model->generateFromLatent(latent_vector);
    }
    else if (model_type == "singlenet" && singlenet_model && singlenet_model->isReady())
    {
      result = singlenet_model->generateFromLatent(latent_vector);
    }

    if (!result.isNull())
    {
      // Convert QImage to RGBA format for ossia
      QImage rgba_image = result.convertToFormat(QImage::Format_RGBA8888);
      
      // Return a function that will be executed in the main thread
      return [rgba_image = std::move(rgba_image)](GenerativeImageGAN& node) mutable
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
      return [](GenerativeImageGAN& node)
      {
        node.inferenceInProgress = false;
      };
    }
  }
  catch (const std::exception& e)
  {
    return [](GenerativeImageGAN& node)
    {
      node.inferenceInProgress = false;
      // Keep previous output on error
    };
  }
  catch (...)
  {
    return [](GenerativeImageGAN& node)
    {
      node.inferenceInProgress = false;
    };
  }
}

}
