#include "GAN.hpp"

#include <QFile>

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
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  session_options_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  {
    using namespace Ort;

    auto api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* cuda_option_v2 = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_option_v2));
    const std::vector keys{
        "arena_extend_strategy",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "cudnn_conv_use_max_workspace",
        "cudnn_conv1d_pad_to_nc1d",
        "enable_cuda_graph",
        "enable_skip_layer_norm_strict_mode"};
    const std::vector values{
        "kNextPowerOfTwo", "EXHAUSTIVE", "1", "1", "1", "0", "1"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
        cuda_option_v2, keys.data(), values.data(), keys.size()));
    // FIXME release options
    session_options_.AppendExecutionProvider_CUDA_V2(*cuda_option_v2);
  }

  if (config_.model_paths.size() >= 2)
  {
    // Load mapping and synthesis networks
    mapping_session_ = std::make_unique<Ort::Session>(
        env_, config_.model_paths[0].c_str(), session_options_);
    synthesis_session_ = std::make_unique<Ort::Session>(
        env_, config_.model_paths[1].c_str(), session_options_);
    qDebug() << "Loaded StyleGAN models:" << config_.model_paths[0].c_str()
             << "and" << config_.model_paths[1].c_str();
  }
}

QImage StyleGANModel::generateRandom()
{
  auto latent = GANUtils::generateRandomLatent(
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
  auto z_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      const_cast<float*>(z_vector.data()),
      z_vector.size(),
      z_shape.data(),
      z_shape.size());

  // Create truncation psi tensor (for StyleGAN)
  std::vector<float> truncation_psi = {0.7f, 0.7f};
  std::vector<int64_t> psi_shape = {2};
  auto psi_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      truncation_psi.data(),
      truncation_psi.size(),
      psi_shape.data(),
      psi_shape.size());

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
  auto w_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      const_cast<float*>(w_vector.data()),
      w_vector.size(),
      w_shape.data(),
      w_shape.size());

  // Create noise tensor
  std::vector<float> noise = {0.0f};
  std::vector<int64_t> noise_shape = {1};
  auto noise_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      noise.data(),
      noise.size(),
      noise_shape.data(),
      noise_shape.size());

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

  int channels = static_cast<int>(shape[1]);
  int height = static_cast<int>(shape[2]);
  int width = static_cast<int>(shape[3]);

  return GANUtils::normalizeToImage(
      data, shape, config_.output_min, config_.output_max);
}

// Single Network GAN Implementation
SingleNetworkGAN::SingleNetworkGAN(const GANConfig& config)
    : env_(ORT_LOGGING_LEVEL_WARNING, "single_gan_model")
    , config_(config)
    , rng_(std::random_device{}())
{
  session_options_.SetIntraOpNumThreads(1);

  // For PyTorchGAN, configure for dynamic shape handling
  if (config_.model_type == "PyTorchGAN")
  {
    // Use only basic optimizations and sequential execution for stability
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_BASIC);
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options_.DisableMemPattern();
    session_options_.DisableCpuMemArena();
    qDebug() << "PyTorchGAN: Using basic optimizations with memory "
                "optimizations disabled";
  }
  else
  {
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
  }

  if (!config_.model_paths.empty())
  {
    generator_session_ = std::make_unique<Ort::Session>(
        env_, config_.model_paths[0].c_str(), session_options_);
    qDebug() << "Loaded single GAN model:" << config_.model_paths[0].c_str();
  }
}

QImage SingleNetworkGAN::generateRandom()
{
  // Generate model-specific random inputs
  std::vector<std::vector<float>> inputs;

  if (config_.model_type == "EigenGAN")
  {
    // eps: [1,512] shape (first input)
    inputs.push_back(GANUtils::generateRandomLatent(512, 0.0f, 1.0f));
    // z_ to z_5: [1,6] shape each
    for (int i = 0; i < 6; ++i)
    {
      inputs.push_back(GANUtils::generateRandomLatent(6, 0.0f, 1.0f));
    }
  }
  else if (config_.model_type == "PyTorchGAN")
  {
    // PyTorchGAN uses uniform random [0,1]
    inputs.push_back(
        GANUtils::generateUniformRandom(config_.latent_dim, 0.0f, 1.0f));
  }
  else
  {
    // Default: single latent vector with normal distribution
    inputs.push_back(
        GANUtils::generateRandomLatent(
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

    auto tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(inputs[i].data()),
        inputs[i].size(),
        shape.data(),
        shape.size());

    ort_inputs.push_back(std::move(tensor));
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
  return GANUtils::normalizeToImage(
      data, shape, config_.output_min, config_.output_max);
}
/*
// Factory Implementation
std::unique_ptr<GANModel> GANFactory::createModel(const std::string& model_type, const std::vector<std::string>& model_paths)
{
    if (model_type == "StyleGAN" || model_type == "FBAnime")
    {
        auto config = getFBAnimeConfig(model_paths);
        return std::make_unique<StyleGANModel>(config);
    }
    else if (model_type == "EigenGAN")
    {
        auto config = getEigenGANConfig(model_paths);
        return std::make_unique<SingleNetworkGAN>(config);
    }
    else if (model_type == "MobileStyleGAN")
    {
        auto config = getMobileStyleGANConfig(model_paths);
        return std::make_unique<SingleNetworkGAN>(config);
    }
    else if (model_type == "PyTorchGAN")
    {
        auto config = getPyTorchGANConfig(model_paths);
        return std::make_unique<SingleNetworkGAN>(config);
    }
    else if (model_type == "AnimeGANv3")
    {
        auto config = getAnimeGANv3Config(model_paths);
        return std::make_unique<ImageTranslationGAN>(config);
    }
    else if (model_type == "FastSRGAN")
    {
      auto config = getFastSRGANConfig(model_paths);
      return std::make_unique<ImageTranslationGAN>(config);
    }
    else if (model_type == "DeblurGANv2")
    {
        auto config = getDeblurGANv2Config(model_paths);
        return std::make_unique<ImageTranslationGAN>(config);
    }
    
    qDebug() << "Unknown model type:" << model_type.c_str();
    return nullptr;
}
*/
GANConfig
GANFactory::getFBAnimeConfig(const std::vector<std::string>& model_paths)
{
  GANConfig config;
  config.model_type = "FBAnime";
  config.model_paths = model_paths;
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
GANFactory::getEigenGANConfig(const std::vector<std::string>& model_paths)
{
  GANConfig config;
  config.model_type = "EigenGAN";
  config.model_paths = model_paths;
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
    const std::vector<std::string>& model_paths)
{
  GANConfig config;
  config.model_type = "MobileStyleGAN";
  config.model_paths = model_paths;
  config.latent_dim = 512;
  config.output_width = 1024;
  config.output_height = 1024;
  config.requires_mapping_network = false; // Single ONNX file
  config.input_names = {"latent_vector"};
  config.input_shapes = {{1, 512}};
  config.input_mean = 0.0f;
  config.input_std = 1.0f;
  config.output_min = 0.0f; // MobileStyleGAN likely outputs [0,1] range
  config.output_max = 1.0f;
  return config;
}

GANConfig
GANFactory::getPyTorchGANConfig(const std::vector<std::string>& model_paths)
{
  GANConfig config;
  config.model_type = "PyTorchGAN";
  config.model_paths = model_paths;
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

GANConfig
GANFactory::getAnimeGANv3Config(const std::vector<std::string>& model_paths)
{
  GANConfig config;
  config.model_type = "AnimeGANv3";
  config.model_paths = model_paths;
  config.latent_dim = 0;    // No latent space for image translation
  config.input_width = 256; // Default input size (can be dynamic)
  config.input_height = 256;
  config.output_width = 256; // Same as input for AnimeGANv3
  config.output_height = 256;
  config.requires_mapping_network = false;
  config.is_generative = false; // Image-to-image translation
  config.input_names = {"AnimeGANv3_input:0"};
  config.output_names = {"generator_1/main/out_layer:0"};
  config.input_shapes
      = {{1, -1, -1, 3}};   // [batch, height, width, channels] - dynamic HW
  config.input_mean = 0.0f; // Images normalized to [0,1]
  config.input_std = 1.0f;
  config.output_min = 0.0f; // Output normalized to [0,1]
  config.output_max = 1.0f;
  return config;
}

GANConfig
GANFactory::getFastSRGANConfig(const std::vector<std::string>& model_paths)
{
  GANConfig config;
  config.model_type = "FastSRGAN";
  config.model_paths = model_paths;
  config.latent_dim = 0;    // No latent space for super-resolution
  config.input_width = 256; // Fixed input size for this model
  config.input_height = 256;
  config.output_width = 1024; // 4x upscaling
  config.output_height = 1024;
  config.requires_mapping_network = false;
  config.is_generative
      = false; // Image-to-image translation (super-resolution)
  config.input_names = {"input_1"};
  config.output_names = {"model_2"};
  config.input_shapes
      = {{1, 3, 256, 256}}; // [batch, channels, height, width] - NCHW format
  config.input_mean
      = 0.5f; // Images normalized to [0,1], then centered to [-1,1]
  config.input_std = 0.5f;   // For (x - 0.5) / 0.5 normalization
  config.output_min = -1.0f; // Output in [-1,1] range based on inspection
  config.output_max = 1.0f;
  return config;
}

GANConfig
GANFactory::getDeblurGANv2Config(const std::vector<std::string>& model_paths)
{
  GANConfig config;
  config.model_type = "DeblurGANv2";
  config.model_paths = model_paths;
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
  qDebug() << "ImageTranslationGAN constructor start for"
           << config.model_type.c_str();
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  qDebug() << "Session options configured";

  // Add CUDA provider if available (temporarily disabled for debugging)

  {
    using namespace Ort;
    auto api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* cuda_option_v2 = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_option_v2));
    const std::vector keys{
        "arena_extend_strategy",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "cudnn_conv_use_max_workspace",
        "cudnn_conv1d_pad_to_nc1d",
        "enable_cuda_graph",
        "enable_skip_layer_norm_strict_mode"};
    const std::vector values{
        "kNextPowerOfTwo", "EXHAUSTIVE", "1", "1", "1", "0", "1"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
        cuda_option_v2, keys.data(), values.data(), keys.size()));
    session_options_.AppendExecutionProvider_CUDA_V2(*cuda_option_v2);
  }

  if (!config_.model_paths.empty())
  {
    translation_session_ = std::make_unique<Ort::Session>(
        env_, config_.model_paths[0].c_str(), session_options_);
    qDebug() << "Loaded image translation model:"
             << config_.model_paths[0].c_str();
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
  if (config_.model_type == "FastSRGAN" || config_.model_type == "DeblurGANv2")
  {
    if (input_image.width() != config_.input_width
        || input_image.height() != config_.input_height)
    {
      processed_image = input_image.scaled(
          config_.input_width,
          config_.input_height,
          Qt::IgnoreAspectRatio,
          Qt::SmoothTransformation);
    }
  }

  // Convert QImage to tensor format
  auto tensor_data = imageToTensor(processed_image);
  if (tensor_data.empty())
  {
    qDebug() << "Failed to convert image to tensor";
    return QImage();
  }

  // Create input tensor with format based on model type
  std::vector<int64_t> input_shape;
  if (config_.model_type == "FastSRGAN" || config_.model_type == "DeblurGANv2")
  {
    // NCHW format: [batch, channels, height, width]
    input_shape = {1, 3, processed_image.height(), processed_image.width()};
  }
  else
  {
    // NHWC format: [batch, height, width, channels] (AnimeGANv3)
    input_shape = {1, processed_image.height(), processed_image.width(), 3};
  }

  auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      tensor_data.data(),
      tensor_data.size(),
      input_shape.data(),
      input_shape.size());

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

std::vector<float> ImageTranslationGAN::imageToTensor(const QImage& image)
{
  if (image.isNull())
    return {};

  // Convert to RGB format if necessary
  QImage rgb_image = image.convertToFormat(QImage::Format_RGB888);

  int width = rgb_image.width();
  int height = rgb_image.height();
  std::vector<float> tensor_data(height * width * 3);

  if (config_.model_type == "FastSRGAN" || config_.model_type == "DeblurGANv2")
  {
    // NCHW format: [channels, height, width]
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        QRgb pixel = rgb_image.pixel(x, y);

        // Normalize based on config
        float r
            = (qRed(pixel) / 255.0f - config_.input_mean) / config_.input_std;
        float g = (qGreen(pixel) / 255.0f - config_.input_mean)
                  / config_.input_std;
        float b
            = (qBlue(pixel) / 255.0f - config_.input_mean) / config_.input_std;

        // NCHW format: tensor[c][y][x]
        tensor_data[0 * height * width + y * width + x] = r; // R channel
        tensor_data[1 * height * width + y * width + x] = g; // G channel
        tensor_data[2 * height * width + y * width + x] = b; // B channel
      }
    }
  }
  else
  {
    // NHWC format: [height, width, channels] (AnimeGANv3)
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        QRgb pixel = rgb_image.pixel(x, y);

        // Normalize to [0,1] range
        float r = qRed(pixel) / 255.0f;
        float g = qGreen(pixel) / 255.0f;
        float b = qBlue(pixel) / 255.0f;

        // NHWC format: tensor[y][x][c]
        size_t idx = y * width * 3 + x * 3;
        tensor_data[idx + 0] = r;
        tensor_data[idx + 1] = g;
        tensor_data[idx + 2] = b;
      }
    }
  }

  return tensor_data;
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

  if (config_.model_type == "FastSRGAN" || config_.model_type == "DeblurGANv2")
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
    // AnimeGANv3 output format: [batch, height, width, channels] (NHWC)
    int height = static_cast<int>(shape[1]);
    int width = static_cast<int>(shape[2]);
    int channels = static_cast<int>(shape[3]);

    if (channels != 3)
      return QImage();

    QImage image(width, height, QImage::Format_RGB888);

    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        // NHWC format: data[0][y][x][c]
        size_t idx = y * width * channels + x * channels;
        float r = data[idx + 0];
        float g = data[idx + 1];
        float b = data[idx + 2];

        // Convert from [0,1] to [0,255] and clamp
        int red = static_cast<int>(std::clamp(r * 255.0f, 0.0f, 255.0f));
        int green = static_cast<int>(std::clamp(g * 255.0f, 0.0f, 255.0f));
        int blue = static_cast<int>(std::clamp(b * 255.0f, 0.0f, 255.0f));

        image.setPixel(x, y, qRgb(red, green, blue));
      }
    }

    return image;
  }
}

// Utility Functions
std::vector<float>
GANUtils::generateRandomLatent(size_t size, float mean, float std)
{
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<float> dist(mean, std);

  std::vector<float> latent(size);
  for (auto& val : latent)
  {
    val = dist(gen);
  }
  return latent;
}

std::vector<float>
GANUtils::generateUniformRandom(size_t size, float min, float max)
{
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min, max);

  std::vector<float> latent(size);
  for (auto& val : latent)
  {
    val = dist(gen);
  }
  return latent;
}

std::vector<float> GANUtils::interpolateLatents(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float t)
{
  if (a.size() != b.size())
    return a;

  std::vector<float> result(a.size());
  for (size_t i = 0; i < a.size(); ++i)
  {
    result[i] = a[i] * (1.0f - t) + b[i] * t;
  }
  return result;
}

QImage GANUtils::normalizeToImage(
    const float* data,
    const std::vector<int64_t>& shape,
    float min_val,
    float max_val)
{
  if (shape.size() != 4)
    return QImage();

  int channels = static_cast<int>(shape[1]);
  int height = static_cast<int>(shape[2]);
  int width = static_cast<int>(shape[3]);

  if (channels != 3)
    return QImage();

  // Find actual min/max for better normalization
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
      // Get RGB values from NCHW tensor
      float r = data[0 * height * width + y * width + x];
      float g = data[1 * height * width + y * width + x];
      float b = data[2 * height * width + y * width + x];

      // Normalize to [0,255]
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

QImage
GANUtils::nchwToQImage(const float* data, int width, int height, int channels)
{
  if (channels != 3)
    return QImage();

  QImage image(width, height, QImage::Format_RGB888);

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      float r = data[0 * height * width + y * width + x];
      float g = data[1 * height * width + y * width + x];
      float b = data[2 * height * width + y * width + x];

      // Assume values are in [0,1] range
      int red = static_cast<int>(std::clamp(r * 255.0f, 0.0f, 255.0f));
      int green = static_cast<int>(std::clamp(g * 255.0f, 0.0f, 255.0f));
      int blue = static_cast<int>(std::clamp(b * 255.0f, 0.0f, 255.0f));

      image.setPixel(x, y, qRgb(red, green, blue));
    }
  }

  return image;
}

} // namespace GANInference
