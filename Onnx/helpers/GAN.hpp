#pragma once

#include <Onnx/helpers/ImageBuffer.hpp>

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <unordered_map>

namespace Onnx
{
struct ModelSpec;
// Configuration for different GAN-like architectures
struct GANConfig
{
  std::string model_type;
  std::vector<std::string_view> model_bytes;
  int latent_dim = 0; // 0 for image-to-image models
  int output_width;
  int output_height;
  int input_width = 0;  // For image-to-image models
  int input_height = 0; // For image-to-image models
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::vector<int64_t>> input_shapes;
  bool requires_mapping_network = false;
  bool is_generative = true; // false for image-to-image translation

  // Tensor format
  std::string tensor_format = "NCHW"; // "NCHW" or "NHWC"

  // Normalization parameters
  float input_mean = 0.0f;
  float input_std = 1.0f;
  float output_min = -1.0f;
  float output_max = 1.0f;
};

// Helper function to update GANConfig with model specifications  
void updateConfigWithModelSpec(GANConfig& config, const ModelSpec& spec, const Onnx::ImageData& input_image = Onnx::ImageData{});

// StyleGAN-like architecture (mapping + synthesis)
class StyleGANModel
{
private:
  std::unique_ptr<Ort::Session> mapping_session_;
  std::unique_ptr<Ort::Session> synthesis_session_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  GANConfig config_;
  std::mt19937 rng_;

public:
  StyleGANModel(const GANConfig& config);

  Onnx::ImageData generateRandom();
  Onnx::ImageData generateFromLatent(const std::vector<float>& latent);
  Onnx::ImageData transformImage(const Onnx::ImageData& input_image)
  {
    return Onnx::ImageData{};
  } // Not supported
  size_t getLatentSize() const { return config_.latent_dim; }
  std::pair<int, int> getOutputSize() const
  {
    return {config_.output_width, config_.output_height};
  }
  std::string getModelType() const { return config_.model_type; }
  bool isReady() const { return mapping_session_ && synthesis_session_; }
  bool isGenerativeModel() const { return true; }

private:
  std::vector<float> runMapping(const std::vector<float>& z_vector);
  Onnx::ImageData runSynthesis(const std::vector<float>& w_vector);
  Onnx::ImageData tensorToImage(const float* data, const std::vector<int64_t>& shape);
};

// Single-network GAN architecture (e.g., EigenGAN)
class SingleNetworkGAN
{
private:
  std::unique_ptr<Ort::Session> generator_session_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  GANConfig config_;
  std::mt19937 rng_;

public:
  SingleNetworkGAN(const GANConfig& config);

  Onnx::ImageData generateRandom();
  Onnx::ImageData generateFromLatent(const std::vector<float>& latent);
  Onnx::ImageData transformImage(const Onnx::ImageData& input_image)
  {
    return Onnx::ImageData{};
  } // Not supported
  size_t getLatentSize() const { return config_.latent_dim; }
  std::pair<int, int> getOutputSize() const
  {
    return {config_.output_width, config_.output_height};
  }
  std::string getModelType() const { return config_.model_type; }
  bool isReady() const { return generator_session_ != nullptr; }
  bool isGenerativeModel() const { return true; }

private:
  Onnx::ImageData runGenerator(const std::vector<std::vector<float>>& inputs);
  Onnx::ImageData tensorToImage(const float* data, const std::vector<int64_t>& shape);
};

// Image-to-image translation architecture (e.g., AnimeGANv3)
class ImageTranslationGAN
{
private:
  std::unique_ptr<Ort::Session> translation_session_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  GANConfig config_;

public:
  ImageTranslationGAN(const GANConfig& config);

  Onnx::ImageData generateRandom() { return Onnx::ImageData{}; } // Not supported
  Onnx::ImageData generateFromLatent(const std::vector<float>& latent)
  {
    return Onnx::ImageData{};
  } // Not supported
  Onnx::ImageData transformImage(const Onnx::ImageData& input_image);
  size_t getLatentSize() const { return 0; } // No latent space
  std::pair<int, int> getOutputSize() const
  {
    return {config_.output_width, config_.output_height};
  }
  std::string getModelType() const { return config_.model_type; }
  bool isReady() const { return translation_session_ != nullptr; }
  bool isGenerativeModel() const { return false; }

  // Additional methods for image translation
  std::pair<int, int> getInputSize() const
  {
    return {config_.input_width, config_.input_height};
  }

private:
  Onnx::ImageData runTranslation(const Onnx::ImageData& input_image);
  Onnx::ImageData tensorToImage(const float* data, const std::vector<int64_t>& shape);
  std::vector<float> imageToTensor(const Onnx::ImageData& image);
};

// Factory for creating different GAN models
class GANFactory
{
public:
  static GANConfig
  getFBAnimeConfig(const std::vector<std::string_view>& models);
  static GANConfig
  getEigenGANConfig(const std::vector<std::string_view>& models);
  static GANConfig
  getMobileStyleGANConfig(const std::vector<std::string_view>& models);
  static GANConfig
  getPyTorchGANConfig(const std::vector<std::string_view>& models);
  static GANConfig
  getAnimeGANv3Config(const std::vector<std::string_view>& models);
  static GANConfig
  getFastSRGANConfig(const std::vector<std::string_view>& models);
  static GANConfig
  getDeblurGANv2Config(const std::vector<std::string_view>& models);
};


} // namespace Onnx
