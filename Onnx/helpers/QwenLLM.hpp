#pragma once

#include <onnxruntime_cxx_api.h>
#include <ortx_tokenizer.h>
#include <ortx_utils.h>

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace Onnx
{
class QwenLLMInference
{
public:
  QwenLLMInference(
      std::string_view modelPath,
      std::string_view tokenizerModelPath);

  ~QwenLLMInference();

  std::string generate(
      const std::string& prompt,
      int maxTokens = 512,
      float temperature = 0.7f,
      float topP = 0.9f,
      int topK = 40);

  void generateStreaming(
      const std::string& prompt,
      std::function<bool(const std::string&)> tokenCallback,
      int maxTokens = 100,
      float temperature = 0.7f,
      float topP = 0.9f,
      int topK = 40);

private:
  std::vector<int64_t> tokenize(const std::string& text) const;
  std::string decodeToken(int64_t tokenId) const;
  std::string decodeTokens(std::span<int64_t> tokens) const;
  
  int64_t sampleToken(
      std::span<float> logits,
      float temperature,
      float topP,
      int topK);

  std::vector<float> runModel(
      std::span<int64_t> inputIds,
      std::span<int64_t> attentionMask,
      std::span<std::vector<Ort::Float16_t>> pastKeyValues,
      std::vector<std::vector<Ort::Float16_t>>& newKeyValues);

  void applyTemperature(std::span<float> logits, float temperature);
  void applyTopK(std::span<float> logits, int k);
  void applyTopP(std::span<float> logits, float p);
  void softmax(std::span<float> logits);

  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  Ort::AllocatorWithDefaultOptions allocator;
  
  std::unique_ptr<Ort::Session> modelSession;
  OrtxTokenizer* tokenizer{};

  // Model configuration from JSON spec
  static constexpr const int vocabSize = 151936;
  static constexpr const int hiddenSize = 1536;
  static constexpr const int numLayers = 28;         // num_hidden_layers
  static constexpr const int numKeyValueHeads = 2;   // num_key_value_heads
  static constexpr const int numAttentionHeads = 12; // num_attention_heads
  static constexpr const int headDim = 128;          // head_size
  static constexpr const int maxPositionEmbeddings = 131072; // context_length

  // Special tokens
  static constexpr const int64_t eosTokenId = 151643;
  static constexpr const int64_t padTokenId = 151643;

  // Input/output names
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  std::vector<const char*> inputNamePtrs;
  std::vector<const char*> outputNamePtrs;
  
  // KV cache for efficient generation
  std::vector<std::vector<Ort::Float16_t>> pastKeyValues;
};
}
