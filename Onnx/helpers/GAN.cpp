#include "GAN.hpp"

#include <QFile>
#include <QString>

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>
#include <cmath>

#include <algorithm>
namespace Onnx
{

// StyleGAN Model Implementation
StyleGANModel::StyleGANModel(const GANConfig& config)
    : env_(ORT_LOGGING_LEVEL_WARNING, "stylegan_model")
    , config_(config)
    , rng_(std::random_device{}())
{
  Onnx::Options oopts;
  session_options_ = Onnx::create_session_options(oopts);

  if (config_.model_bytes.size() >= 2)
  {
    auto& model0_str = config_.model_bytes[0];
    auto& model1_str = config_.model_bytes[1];
    // Load mapping and synthesis networks
    mapping_session_ = std::make_unique<Ort::Session>(
        env_, model0_str.data(), model0_str.size(), session_options_);
    synthesis_session_ = std::make_unique<Ort::Session>(
        env_, model1_str.data(), model1_str.size(), session_options_);
  }
}

QImage StyleGANModel::generateRandom()
{
  auto latent = generateRandomLatent(
      config_.latent_dim, config_.input_mean, config_.input_std);
  return generateFromLatent(latent);
}

QImage StyleGANModel::generateFromLatent(const std::vector<float>& latent)
{
  if (!isReady() || latent.size() != config_.latent_dim)
  {
    qDebug() << "StyleGAN model not ready or invalid latent size";
    return QImage();
  }

  // Run through mapping network
  auto w_vector = runMapping(latent);

  // Run through synthesis network
  return runSynthesis(w_vector);
}

std::vector<float>
StyleGANModel::runMapping(const std::vector<float>& z_vector)
{
  auto memory_info
      = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create Z tensor
  std::vector<int64_t> z_shape = {1, static_cast<int64_t>(z_vector.size())};
  std::vector<float> z_copy(z_vector); // vec_to_tensor needs non-const data
  auto z_tensor = vec_to_tensor<float>(z_copy, z_shape);

  // Create truncation psi tensor (for StyleGAN)
  std::vector<float> truncation_psi = {0.7f, 0.7f};
  std::vector<int64_t> psi_shape = {2};
  auto psi_tensor = vec_to_tensor<float>(truncation_psi, psi_shape);

  // Get input/output names
  auto input0_name = mapping_session_->GetInputNameAllocated(0, allocator_);
  auto input1_name = mapping_session_->GetInputNameAllocated(1, allocator_);
  auto output0_name = mapping_session_->GetOutputNameAllocated(0, allocator_);

  const char* input_names[] = {input0_name.get(), input1_name.get()};
  const char* output_names[] = {output0_name.get()};

  // Prepare inputs
  std::vector<Ort::Value> inputs;
  inputs.push_back(std::move(z_tensor));
  inputs.push_back(std::move(psi_tensor));

  // Run mapping
  auto outputs = mapping_session_->Run(
      Ort::RunOptions{nullptr},
      input_names,
      inputs.data(),
      inputs.size(),
      output_names,
      1);

  // Extract W vector
  auto& w_tensor = outputs[0];
  auto w_shape_info = w_tensor.GetTensorTypeAndShapeInfo();
  auto w_shape = w_shape_info.GetShape();
  const float* w_data = w_tensor.GetTensorData<float>();

  size_t w_size = 1;
  for (auto dim : w_shape)
    w_size *= dim;

  return std::vector<float>(w_data, w_data + w_size);
}

QImage StyleGANModel::runSynthesis(const std::vector<float>& w_vector)
{
  auto memory_info
      = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create W tensor (reshape to [1, 16, 1024] for StyleGAN)
  std::vector<int64_t> w_shape = {1, 16, 1024};
  std::vector<float> w_copy(w_vector); // vec_to_tensor needs non-const data
  auto w_tensor = vec_to_tensor<float>(w_copy, w_shape);

  // Create noise tensor
  std::vector<float> noise = {0.0f};
  std::vector<int64_t> noise_shape = {1};
  auto noise_tensor = vec_to_tensor<float>(noise, noise_shape);

  // Get input/output names
  auto input0_name = synthesis_session_->GetInputNameAllocated(0, allocator_);
  auto input1_name = synthesis_session_->GetInputNameAllocated(1, allocator_);
  auto output0_name
      = synthesis_session_->GetOutputNameAllocated(0, allocator_);

  const char* input_names[] = {input0_name.get(), input1_name.get()};
  const char* output_names[] = {output0_name.get()};

  // Prepare inputs
  std::vector<Ort::Value> inputs;
  inputs.push_back(std::move(w_tensor));
  inputs.push_back(std::move(noise_tensor));

  // Run synthesis
  auto outputs = synthesis_session_->Run(
      Ort::RunOptions{nullptr},
      input_names,
      inputs.data(),
      inputs.size(),
      output_names,
      1);

  // Convert output to image
  auto& img_tensor = outputs[0];
  auto img_shape_info = img_tensor.GetTensorTypeAndShapeInfo();
  auto img_shape = img_shape_info.GetShape();
  const float* img_data = img_tensor.GetTensorData<float>();

  return tensorToImage(img_data, img_shape);
}

QImage StyleGANModel::tensorToImage(
    const float* data,
    const std::vector<int64_t>& shape)
{
  if (shape.size() != 4)
    return QImage();

  return normalizeToImage(data, shape, config_.output_min, config_.output_max);
}

// Single Network GAN Implementation
SingleNetworkGAN::SingleNetworkGAN(const GANConfig& config)
    : env_(ORT_LOGGING_LEVEL_WARNING, "single_gan_model")
    , config_(config)
    , rng_(std::random_device{}())
{
  Onnx::Options oopts;
  session_options_ = Onnx::create_session_options(oopts);

  if (!config_.model_bytes.empty())
  {
    auto& model0_str = config_.model_bytes[0];
    generator_session_ = std::make_unique<Ort::Session>(
        env_, model0_str.data(), model0_str.size(), session_options_);
  }
}

QImage SingleNetworkGAN::generateRandom()
{
  // Generate model-specific random inputs
  std::vector<std::vector<float>> inputs;

  if (config_.model_type == "EigenGAN")
  {
    // eps: [1,512] shape (first input)
    inputs.push_back(generateRandomLatent(512, 0.0f, 1.0f));
    // z_ to z_5: [1,6] shape each
    for (int i = 0; i < 6; ++i)
    {
      inputs.push_back(generateRandomLatent(6, 0.0f, 1.0f));
    }
  }
  else if (config_.model_type == "PyTorchGAN")
  {
    // PyTorchGAN uses uniform random [0,1]
    inputs.push_back(
        generateUniformRandom(config_.latent_dim, 0.0f, 1.0f));
  }
  else
  {
    // Default: single latent vector with normal distribution
    inputs.push_back(
        generateRandomLatent(
            config_.latent_dim, config_.input_mean, config_.input_std));
  }

  return runGenerator(inputs);
}

QImage SingleNetworkGAN::generateFromLatent(const std::vector<float>& latent)
{
  if (!isReady())
  {
    qDebug() << "Single GAN model not ready";
    return QImage();
  }

  std::vector<std::vector<float>> inputs = {latent};
  return runGenerator(inputs);
}

QImage
SingleNetworkGAN::runGenerator(const std::vector<std::vector<float>>& inputs)
{
  auto memory_info
      = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create input tensors
  std::vector<Ort::Value> ort_inputs;
  std::vector<std::string> input_name_strings;
  std::vector<const char*> input_names;
  std::vector<float> input_copy;

  for (size_t i = 0; i < inputs.size() && i < config_.input_shapes.size(); ++i)
  {
    // Get actual input name from session
    auto name_alloc = generator_session_->GetInputNameAllocated(i, allocator_);
    std::string name_str = name_alloc.get();

    // If name is empty, use a default
    if (name_str.empty())
    {
      name_str = "input_" + std::to_string(i);
    }

    input_name_strings.push_back(name_str);
    input_names.push_back(input_name_strings.back().c_str());

    std::vector<int64_t> shape = config_.input_shapes[i];

    // Fix for PyTorchGAN: Handle dynamic batch dimensions (-1)
    // Dynamic dimensions in ONNX models need to be set to concrete values
    if (config_.model_type == "PyTorchGAN")
    {
      qDebug() << "PyTorchGAN: Original shape for input" << i << ":";
      for (size_t j = 0; j < shape.size(); ++j)
      {
        qDebug() << "  [" << j << "]:" << shape[j];
      }

      // Check if first dimension is dynamic (-1) and set it to batch size 1
      if (!shape.empty() && shape[0] == -1)
      {
        shape[0] = 1;
        qDebug() << "PyTorchGAN: Fixed dynamic batch dimension from -1 to 1";
      }

      qDebug() << "PyTorchGAN: Final shape for input" << i << ":";
      for (size_t j = 0; j < shape.size(); ++j)
      {
        qDebug() << "  [" << j << "]:" << shape[j];
      }
      qDebug() << "PyTorchGAN: Input data size:" << inputs[i].size();
    }

    // FIXME multi-input??
    input_copy = inputs[i]; // vec_to_tensor needs non-const data

    ort_inputs.push_back(vec_to_tensor<float>(input_copy, shape));
  }

  // Get output name
  auto output_name_alloc
      = generator_session_->GetOutputNameAllocated(0, allocator_);
  std::string output_name_str = output_name_alloc.get();
  if (output_name_str.empty())
  {
    output_name_str = "output_0";
  }
  const char* output_names[] = {output_name_str.c_str()};

  // Run inference
  std::vector<Ort::Value> outputs;
  try
  {
    // Use actual input/output names
    outputs = generator_session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        ort_inputs.data(),
        ort_inputs.size(),
        output_names,
        1);
  }
  catch (const Ort::Exception& e)
  {
    qDebug() << "Run with names failed:" << e.what();
    throw; // Re-throw the exception instead of trying alternative approaches that may not work
  }

  // Convert output to image
  auto& img_tensor = outputs[0];
  auto img_shape_info = img_tensor.GetTensorTypeAndShapeInfo();
  auto img_shape = img_shape_info.GetShape();
  const float* img_data = img_tensor.GetTensorData<float>();

  return tensorToImage(img_data, img_shape);
}

QImage SingleNetworkGAN::tensorToImage(
    const float* data,
    const std::vector<int64_t>& shape)
{
  if (config_.model_type == "PyTorchGAN" && shape.size() == 4)
  {
    // PyTorchGAN uses specific normalization: (0.5 + 255*output_data)
    // Output format: NCHW [1, 3, height, width]
    int channels = static_cast<int>(shape[1]);
    int height = static_cast<int>(shape[2]);
    int width = static_cast<int>(shape[3]);

    if (channels == 3)
    {
      QImage image(width, height, QImage::Format_RGB888);

      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
          // NCHW format: data[channel][y][x]
          float r = data[0 * height * width + y * width + x];
          float g = data[1 * height * width + y * width + x];
          float b = data[2 * height * width + y * width + x];

          // Apply PyTorchGAN normalization: (0.5 + 255*output_data)
          int red
              = static_cast<int>(std::clamp(0.5f + 255.0f * r, 0.0f, 255.0f));
          int green
              = static_cast<int>(std::clamp(0.5f + 255.0f * g, 0.0f, 255.0f));
          int blue
              = static_cast<int>(std::clamp(0.5f + 255.0f * b, 0.0f, 255.0f));

          image.setPixel(x, y, qRgb(red, green, blue));
        }
      }
      return image;
    }
  }
  else if (config_.model_type == "MobileStyleGAN" && shape.size() == 4)
  {
    // MobileStyleGAN outputs RGB in NCHW format
    // PINTO model zoo ONNX export uses dynamic range (not fixed [-1, 1])
    // Use dynamic normalization based on actual min/max values
    int channels = static_cast<int>(shape[1]);
    int height = static_cast<int>(shape[2]);
    int width = static_cast<int>(shape[3]);

    if (channels == 3)
    {
      // Find actual min/max for dynamic normalization
      float actual_min = data[0], actual_max = data[0];
      size_t total_pixels = channels * height * width;
      for (size_t i = 0; i < total_pixels; ++i)
      {
        actual_min = std::min(actual_min, data[i]);
        actual_max = std::max(actual_max, data[i]);
      }

      QImage image(width, height, QImage::Format_RGB888);

      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
          // NCHW format: data[channel * H * W + y * W + x]
          float r = data[0 * height * width + y * width + x];
          float g = data[1 * height * width + y * width + x];
          float b = data[2 * height * width + y * width + x];

          // Dynamic normalization to [0, 255]
          int red = static_cast<int>(std::clamp(
              (r - actual_min) / (actual_max - actual_min) * 255.0f,
              0.0f,
              255.0f));
          int green = static_cast<int>(std::clamp(
              (g - actual_min) / (actual_max - actual_min) * 255.0f,
              0.0f,
              255.0f));
          int blue = static_cast<int>(std::clamp(
              (b - actual_min) / (actual_max - actual_min) * 255.0f,
              0.0f,
              255.0f));

          image.setPixel(x, y, qRgb(red, green, blue));
        }
      }
      return image;
    }
  }
  else if (config_.model_type == "EigenGAN" && shape.size() == 4)
  {
    // EigenGAN outputs in HWC format [1,256,256,3]
    int height = static_cast<int>(shape[1]);
    int width = static_cast<int>(shape[2]);
    int channels = static_cast<int>(shape[3]);

    if (channels == 3)
    {
      QImage image(width, height, QImage::Format_RGB888);

      // Find min/max for normalization
      float min_val = data[0], max_val = data[0];
      size_t total_pixels = height * width * channels;
      for (size_t i = 0; i < total_pixels; ++i)
      {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
      }

      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
          // HWC format: data[y][x][c]
          float r = data[y * width * channels + x * channels + 0];
          float g = data[y * width * channels + x * channels + 1];
          float b = data[y * width * channels + x * channels + 2];

          // Normalize to [0,255]
          int red = static_cast<int>(std::clamp(
              (r - min_val) / (max_val - min_val) * 255.0f, 0.0f, 255.0f));
          int green = static_cast<int>(std::clamp(
              (g - min_val) / (max_val - min_val) * 255.0f, 0.0f, 255.0f));
          int blue = static_cast<int>(std::clamp(
              (b - min_val) / (max_val - min_val) * 255.0f, 0.0f, 255.0f));

          image.setPixel(x, y, qRgb(red, green, blue));
        }
      }
      return image;
    }
  }

  // Default NCHW format handling
  return normalizeToImage(
      data, shape, config_.output_min, config_.output_max);
}

GANConfig
GANFactory::getFBAnimeConfig(const std::vector<std::string_view>& model_bytes)
{
  GANConfig config;
  config.model_type = "FBAnime";
  config.model_bytes = model_bytes;
  config.latent_dim = 1024;
  config.output_width = 512;
  config.output_height = 1024;
  config.requires_mapping_network = true;
  config.input_mean = 0.0f;
  config.input_std = 1.0f;
  config.output_min = -1.0f;
  config.output_max = 1.0f;
  return config;
}

GANConfig
GANFactory::getEigenGANConfig(const std::vector<std::string_view>& model_bytes)
{
  GANConfig config;
  config.model_type = "EigenGAN";
  config.model_bytes = model_bytes;
  config.latent_dim = 512; // eps dimension
  config.output_width = 256;
  config.output_height = 256;
  config.requires_mapping_network = false;
  config.input_names = {"eps", "z_", "z_1", "z_2", "z_3", "z_4", "z_5"};
  config.input_shapes
      = {{1, 512}, {1, 6}, {1, 6}, {1, 6}, {1, 6}, {1, 6}, {1, 6}};
  config.input_mean = 0.0f;
  config.input_std = 1.0f;
  config.output_min = -1.0f;
  config.output_max = 1.0f;
  return config;
}

GANConfig GANFactory::getMobileStyleGANConfig(
    const std::vector<std::string_view>& model_bytes)
{
  GANConfig config;
  config.model_type = "MobileStyleGAN";
  config.model_bytes = model_bytes;
  config.requires_mapping_network = false; // Single ONNX file
  config.is_generative = true;
  config.input_mean = 0.0f;
  config.input_std = 1.0f;
  config.output_min = 0.0f; // MobileStyleGAN likely outputs [0,1] range
  config.output_max = 1.0f;
  
  // Read actual model specifications
  if (!model_bytes.empty())
  {
    auto spec = readModelSpec(model_bytes[0]);
    updateConfigWithModelSpec(config, spec);
  }

  // Fallback defaults if model spec reading failed
  if (config.latent_dim == 0) config.latent_dim = 512;
  if (config.output_width == 0) config.output_width = 1024;
  if (config.output_height == 0) config.output_height = 1024;
  
  return config;
}

GANConfig GANFactory::getPyTorchGANConfig(
    const std::vector<std::string_view>& model_bytes)
{
  GANConfig config;
  config.model_type = "PyTorchGAN";
  config.model_bytes = model_bytes;
  config.latent_dim = 512;
  config.output_width = 128; // CelebA model
  config.output_height = 128;
  config.requires_mapping_network = false; // Single ONNX file
  config.input_names = {"0"}; // PyTorchGAN uses numeric input names
  config.input_shapes
      = {{-1, 512}};        // Keep dynamic batch dimension to test fix
  config.input_mean = 0.5f; // Uniform random [0,1]
  config.input_std = 0.5f;  // For normalization to [-1,1] if needed
  config.output_min = 0.0f; // PyTorchGAN outputs are normalized differently
  config.output_max = 1.0f;
  return config;
}

GANConfig GANFactory::getAnimeGANv3Config(
    const std::vector<std::string_view>& model_bytes)
{
  GANConfig config;
  config.model_type = "AnimeGANv3";
  config.model_bytes = model_bytes;
  config.latent_dim = 0;    // No latent space for image translation
  config.requires_mapping_network = false;
  config.is_generative = false; // Image-to-image translation
  config.input_mean = 0.0f; // Images normalized to [0,1]
  config.input_std = 1.0f;
  config.output_min = 0.0f; // Output normalized to [0,1]
  config.output_max = 1.0f;
  
  // Read actual model specifications
  if (!model_bytes.empty())
  {
    ModelSpec spec = readModelSpec(model_bytes[0]);
    updateConfigWithModelSpec(config, spec);
  }

  // Fallback defaults if model spec reading failed
  if (config.input_width == 0) config.input_width = 256;
  if (config.input_height == 0) config.input_height = 256;
  if (config.output_width == 0) config.output_width = 256;
  if (config.output_height == 0) config.output_height = 256;
  
  return config;
}

GANConfig GANFactory::getFastSRGANConfig(
    const std::vector<std::string_view>& model_bytes)
{
  GANConfig config;
  config.model_type = "FastSRGAN";
  config.model_bytes = model_bytes;
  config.latent_dim = 0;    // No latent space for super-resolution
  config.requires_mapping_network = false;
  config.is_generative = false; // Image-to-image translation (super-resolution)
  config.input_mean = 0.5f; // Images normalized to [0,1], then centered to [-1,1]
  config.input_std = 0.5f;   // For (x - 0.5) / 0.5 normalization
  config.output_min = -1.0f; // Output in [-1,1] range based on inspection
  config.output_max = 1.0f;
  
  // Read actual model specifications
  if (!model_bytes.empty())
  {
    auto spec = readModelSpec(model_bytes[0]);
    updateConfigWithModelSpec(config, spec);
  }

  // Fallback defaults if model spec reading failed
  if (config.input_width == 0) config.input_width = 256;
  if (config.input_height == 0) config.input_height = 256;
  if (config.output_width == 0) config.output_width = 1024;  // 4x upscaling default
  if (config.output_height == 0) config.output_height = 1024;
  
  return config;
}

GANConfig GANFactory::getDeblurGANv2Config(
    const std::vector<std::string_view>& model_bytes)
{
  GANConfig config;
  config.model_type = "DeblurGANv2";
  config.model_bytes = model_bytes;
  config.latent_dim = 0;    // No latent space for deblurring
  config.input_width = 640; // Fixed input size for this model
  config.input_height = 480;
  config.output_width = 640; // Same size output (deblurring, not resizing)
  config.output_height = 480;
  config.requires_mapping_network = false;
  config.is_generative = false; // Image-to-image translation (deblurring)
  config.input_names = {"input.1"};
  config.output_names = {"645"};
  config.input_shapes
      = {{1, 3, 480, 640}}; // [batch, channels, height, width] - NCHW format
  config.input_mean = 0.0f; // Images already in [0,1] range
  config.input_std = 1.0f;  // No normalization needed
  config.output_min = 0.0f; // Output mostly in [0,1] range based on inspection
  config.output_max = 1.0f;
  return config;
}

// Image Translation GAN Implementation (AnimeGANv3)
ImageTranslationGAN::ImageTranslationGAN(const GANConfig& config)
    : env_(ORT_LOGGING_LEVEL_WARNING, "image_translation_model")
    , config_(config)
{
  Onnx::Options oopts;
  session_options_ = Onnx::create_session_options(oopts);

  if (!config_.model_bytes.empty())
  {
    const auto& model0_str = config_.model_bytes[0];
    translation_session_ = std::make_unique<Ort::Session>(
        env_, model0_str.data(), model0_str.size(), session_options_);
  }
}

QImage ImageTranslationGAN::transformImage(const QImage& input_image)
{
  if (!isReady())
  {
    qDebug() << "Image translation model not ready";
    return QImage();
  }

  return runTranslation(input_image);
}

QImage ImageTranslationGAN::runTranslation(const QImage& input_image)
{
  auto memory_info
      = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Resize input if needed for models with fixed input sizes
  QImage processed_image = input_image;
  
  // Determine if we need to resize based on whether model has fixed dimensions
  bool needsFixedSize = (config_.input_width > 0 && config_.input_height > 0);
  
  if (needsFixedSize && (input_image.width() != config_.input_width || input_image.height() != config_.input_height))
  {
    processed_image = input_image.scaled(
        config_.input_width,
        config_.input_height,
        Qt::IgnoreAspectRatio,
        Qt::SmoothTransformation);
  }

  // Convert QImage to tensor format
  auto tensor_data = Onnx::imageToTensor(
      processed_image,
      config_.tensor_format,
      config_.input_mean,
      config_.input_std);
  if (tensor_data.empty())
  {
    qDebug() << "Failed to convert image to tensor";
    return QImage();
  }

  // Create input tensor with format based on model configuration
  std::vector<int64_t> input_shape;
  if (config_.tensor_format == "NCHW")
  {
    // NCHW format: [batch, channels, height, width]
    input_shape = {1, 3, processed_image.height(), processed_image.width()};
  }
  else
  {
    // NHWC format: [batch, height, width, channels]
    input_shape = {1, processed_image.height(), processed_image.width(), 3};
  }

  auto input_tensor = vec_to_tensor<float>(tensor_data, input_shape);

  // Get input/output names
  auto input_name = translation_session_->GetInputNameAllocated(0, allocator_);
  auto output_name
      = translation_session_->GetOutputNameAllocated(0, allocator_);

  const char* input_names[] = {input_name.get()};
  const char* output_names[] = {output_name.get()};

  // Run inference
  std::vector<Ort::Value> inputs;
  inputs.push_back(std::move(input_tensor));

  try
  {
    auto outputs = translation_session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        inputs.data(),
        inputs.size(),
        output_names,
        1);

    // Convert output to image
    auto& img_tensor = outputs[0];
    auto img_shape_info = img_tensor.GetTensorTypeAndShapeInfo();
    auto img_shape = img_shape_info.GetShape();
    const float* img_data = img_tensor.GetTensorData<float>();

    return tensorToImage(img_data, img_shape);
  }
  catch (const Ort::Exception& e)
  {
    qDebug() << "Image translation inference failed:" << e.what();
    return QImage();
  }
}


QImage ImageTranslationGAN::tensorToImage(
    const float* data,
    const std::vector<int64_t>& shape)
{
  if (shape.size() != 4)
    return QImage();

  int batch = static_cast<int>(shape[0]);
  if (batch != 1)
    return QImage();

  if (config_.tensor_format == "NCHW")
  {
    // NCHW output format: [batch, channels, height, width]
    int channels = static_cast<int>(shape[1]);
    int height = static_cast<int>(shape[2]);
    int width = static_cast<int>(shape[3]);

    if (channels != 3)
      return QImage();

    QImage image(width, height, QImage::Format_RGB888);

    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        // NCHW format: data[0][c][y][x]
        float r = data[0 * height * width + y * width + x]; // R channel
        float g = data[1 * height * width + y * width + x]; // G channel
        float b = data[2 * height * width + y * width + x]; // B channel

        if (config_.model_type == "DeblurGANv2")
        {
          // DeblurGANv2 outputs in [0,1] range, just scale to [0,255]
          r = std::clamp(r, 0.0f, 1.0f);
          g = std::clamp(g, 0.0f, 1.0f);
          b = std::clamp(b, 0.0f, 1.0f);
        }
        else // FastSRGAN
        {
          // Convert from [-1,1] to [0,1]
          r = (r * config_.input_std + config_.input_mean);
          g = (g * config_.input_std + config_.input_mean);
          b = (b * config_.input_std + config_.input_mean);
        }

        int red = static_cast<int>(std::clamp(r * 255.0f, 0.0f, 255.0f));
        int green = static_cast<int>(std::clamp(g * 255.0f, 0.0f, 255.0f));
        int blue = static_cast<int>(std::clamp(b * 255.0f, 0.0f, 255.0f));

        image.setPixel(x, y, qRgb(red, green, blue));
      }
    }

    return image;
  }
  else
  {
    // NHWC output format: [batch, height, width, channels]
    int height = static_cast<int>(shape[1]);
    int width = static_cast<int>(shape[2]);
    int channels = static_cast<int>(shape[3]);

    if (channels != 3)
      return QImage();

    QImage image(width, height, QImage::Format_RGB888);
    auto ptr = image.bits();

    const int N = width * height * 3;
#pragma omp simd
    for (int i = 0; i < N; i++)
    {
      const float val = data[i] * 255.0f;
      ptr[i] = static_cast<int>(
          (val < 255.0f ? (val > 0.f ? val : 0.f) : 255.0f));
    }

    return image;
  }
}


// Helper function to update GANConfig with model specifications
void updateConfigWithModelSpec(GANConfig& config, const ModelSpec& spec, const QImage& input_image)
{
  // Update input/output names from model spec
  config.input_names = spec.input_names;
  config.output_names = spec.output_names;
  config.input_shapes.clear();
  for (auto& ins : spec.inputs)
    config.input_shapes.push_back(ins.shape);

  // Update input dimensions
  if (!config.input_shapes.empty())
  {
    const auto& input_shape = config.input_shapes[0];

    if (input_shape.size() == 4)
    { // Batch, Height, Width, Channels or Batch, Channels, Height, Width
      if (config.is_generative) {
        // For generative models, the input is usually latent space
        if (input_shape[1] > 4) { // Likely NCHW with channels > 4, so this is latent dim
          config.latent_dim = static_cast<int>(input_shape[1]);
        } else if (input_shape[3] > 4) { // Likely NHWC with channels > 4, so this is latent dim  
          config.latent_dim = static_cast<int>(input_shape[3]);
        }
      } else {
        // For image-to-image models, determine format and dimensions
        if (input_shape[1] == 3 || input_shape[1] == 1) {
          // NCHW format: [batch, channels, height, width]
          config.tensor_format = "NCHW";
          if (input_shape[2] != -1) config.input_height = static_cast<int>(input_shape[2]);
          if (input_shape[3] != -1) config.input_width = static_cast<int>(input_shape[3]);
        } else if (input_shape[3] == 3 || input_shape[3] == 1) {
          // NHWC format: [batch, height, width, channels]
          config.tensor_format = "NHWC";
          if (input_shape[1] != -1) config.input_height = static_cast<int>(input_shape[1]);
          if (input_shape[2] != -1) config.input_width = static_cast<int>(input_shape[2]);
        }
        
        // Use input image dimensions for dynamic sizes (-1)
        if (!input_image.isNull()) {
          if (config.input_width <= 0 || input_shape[config.tensor_format == "NCHW" ? 3 : 2] == -1) {
            config.input_width = input_image.width();
          }
          if (config.input_height <= 0 || input_shape[config.tensor_format == "NCHW" ? 2 : 1] == -1) {
            config.input_height = input_image.height();
          }
        }
      }
    }
  }

  // Update output dimensions
  if (!spec.outputs.empty())
  {
    const auto& output_shape = spec.outputs[0].shape;

    if (output_shape.size() == 4)
    { // Batch, Height, Width, Channels or Batch, Channels, Height, Width
      if (output_shape[1] == 3 || output_shape[1] == 1) {
        // NCHW format: [batch, channels, height, width]
        if (output_shape[2] != -1) config.output_height = static_cast<int>(output_shape[2]);
        if (output_shape[3] != -1) config.output_width = static_cast<int>(output_shape[3]);
      } else if (output_shape[3] == 3 || output_shape[3] == 1) {
        // NHWC format: [batch, height, width, channels]
        if (output_shape[1] != -1) config.output_height = static_cast<int>(output_shape[1]);
        if (output_shape[2] != -1) config.output_width = static_cast<int>(output_shape[2]);
      }
      
      // For image-to-image models with dynamic output, use input image scaling
      if (!config.is_generative && !input_image.isNull()) {
        if (config.output_width <= 0) {
          config.output_width = input_image.width();
        }
        if (config.output_height <= 0) {
          config.output_height = input_image.height();
        }
      }
    }
  }

  qDebug() << "Updated config for" << config.model_type.c_str()
           << "- Input:" << config.input_width << "x" << config.input_height
           << "- Output:" << config.output_width << "x" << config.output_height
           << "- Latent dim:" << config.latent_dim
           << "- Format:" << config.tensor_format.c_str();
}


} // namespace Onnx
