#pragma once

#include <QDebug>
#include <QImage>

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <unordered_map>

namespace Onnx
{

// Configuration for different GAN-like architectures
struct GANConfig
{
  std::string model_type;
  std::vector<std::string>
      model_paths;    // Support multi-file models (mapping + synthesis)
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

  // Normalization parameters
  float input_mean = 0.0f;
  float input_std = 1.0f;
  float output_min = -1.0f;
  float output_max = 1.0f;
};

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

  QImage generateRandom();
  QImage generateFromLatent(const std::vector<float>& latent);
  QImage transformImage(const QImage& input_image)
  {
    return QImage();
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
  QImage runSynthesis(const std::vector<float>& w_vector);
  QImage tensorToImage(const float* data, const std::vector<int64_t>& shape);
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

  QImage generateRandom();
  QImage generateFromLatent(const std::vector<float>& latent);
  QImage transformImage(const QImage& input_image)
  {
    return QImage();
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
  QImage runGenerator(const std::vector<std::vector<float>>& inputs);
  QImage tensorToImage(const float* data, const std::vector<int64_t>& shape);
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

  QImage generateRandom() { return QImage(); } // Not supported
  QImage generateFromLatent(const std::vector<float>& latent)
  {
    return QImage();
  } // Not supported
  QImage transformImage(const QImage& input_image);
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
  QImage runTranslation(const QImage& input_image);
  QImage tensorToImage(const float* data, const std::vector<int64_t>& shape);
  std::vector<float> imageToTensor(const QImage& image);
};

// Factory for creating different GAN models
class GANFactory
{
public:
  // static std::unique_ptr<GANModel> createModel(const std::string& model_type, const std::vector<std::string>& model_paths);
  static GANConfig
  getFBAnimeConfig(const std::vector<std::string>& model_paths);
  static GANConfig
  getEigenGANConfig(const std::vector<std::string>& model_paths);
  static GANConfig
  getMobileStyleGANConfig(const std::vector<std::string>& model_paths);
  static GANConfig
  getPyTorchGANConfig(const std::vector<std::string>& model_paths);
  static GANConfig
  getAnimeGANv3Config(const std::vector<std::string>& model_paths);
  static GANConfig
  getFastSRGANConfig(const std::vector<std::string>& model_paths);
  static GANConfig
  getDeblurGANv2Config(const std::vector<std::string>& model_paths);
};

// Utility functions
class GANUtils
{
public:
  // Generate random latent vector (normal distribution)
  static std::vector<float>
  generateRandomLatent(size_t size, float mean = 0.0f, float std = 1.0f);

  // Generate uniform random vector
  static std::vector<float>
  generateUniformRandom(size_t size, float min = 0.0f, float max = 1.0f);

  // Interpolate between two latent vectors
  static std::vector<float> interpolateLatents(
      const std::vector<float>& a,
      const std::vector<float>& b,
      float t);

  // Normalize tensor data to [0,255] range
  static QImage normalizeToImage(
      const float* data,
      const std::vector<int64_t>& shape,
      float min_val,
      float max_val);

  // Convert NCHW tensor to QImage
  static QImage
  nchwToQImage(const float* data, int width, int height, int channels);
};

} // namespace GANInference
