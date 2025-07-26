#pragma once

#include <QImage>

#include <onnxruntime_cxx_api.h>
#include <ortx_tokenizer.h>

#include <memory>
#include <string>
#include <vector>

namespace Onnx
{
class FastVLMInference
{
public:
  FastVLMInference(
      std::string_view visionEncoderPath,
      std::string_view embedTokensPath,
      std::string_view decoderPath,
      std::string_view tokenizerModelPath);

  ~FastVLMInference();

  std::string generateResponse(
      const QImage& image,
      const std::string& prompt,
      float temperature = 1.0f);

private:
  std::vector<int64_t> tokenizeImagePrompt(const std::string& prompt) const;
  std::vector<float> runVisionEncoder(std::span<float> imageData);
  std::vector<float> runEmbedTokens(std::span<int64_t> tokenIds);
  std::string decodeTokens(std::span<int64_t> tokens) const;

  std::string createPromptTemplate(std::string_view userPrompt) const;
  std::string formatImagePrompt(std::string_view prompt) const;

  std::vector<float> createMultimodalEmbeddings(
      std::span<int64_t> tokenIds,
      std::span<float> imageFeatures);
  std::vector<int64_t> generateWithONNXDecoder(
      std::span<float> embeddings,
      int maxTokens,
      float temperature = 0.0f);

  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  Ort::AllocatorWithDefaultOptions allocator;

  std::unique_ptr<Ort::Session> visionEncoderSession;
  std::unique_ptr<Ort::Session> embedTokensSession;
  std::unique_ptr<Ort::Session> decoderSession;

  OrtxTokenizer* tokenizer{};

  std::vector<std::vector<Ort::Float16_t>> keyCache
      = std::vector<std::vector<Ort::Float16_t>>(24);
  std::vector<std::vector<Ort::Float16_t>> valueCache
      = std::vector<std::vector<Ort::Float16_t>>(24);
  std::vector<std::vector<int64_t>> cacheShapes
      = std::vector<std::vector<int64_t>>(24);

  std::vector<std::string> decoderInputNames;
  std::vector<std::string> decoderOutputNames;
  std::vector<const char*> decoderInputNamePtrs;
  std::vector<const char*> decoderOutputNamePtrs;
  std::vector<int64_t> reusableAttentionMask;
  std::vector<int64_t> reusablePositionIds;
};
}
