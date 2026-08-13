#pragma once

#include <onnxruntime_cxx_api.h>
#include <ortx_tokenizer.h>
#include <ortx_utils.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace Onnx
{
// Runs a decoder-only chat model exported by transformers.js / optimum
// (onnx-community/*-ONNX) or onnxruntime-genai: inputs are input_ids,
// attention_mask, optionally position_ids, and one (key, value) pair per
// layer. The graph geometry (layer count, KV heads, head dimension) and the
// KV cache dtype (fp16 or fp32, which varies per model *and* per
// quantization variant) are read from the model at load time; the chat
// template comes from the tokenizer directory (tokenizer_config.json /
// chat_template.jinja, applied through onnxruntime-extensions' Jinja
// engine) and the stop tokens from generation_config.json / config.json,
// falling back to Qwen's ChatML conventions when absent. Generation is
// KV-cached: the prompt is prefilled once, then one token per step.
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

  // Runs the shared generation loop; onToken returns false to stop early.
  void generateLoop(
      const std::string& prompt,
      int maxTokens,
      float temperature,
      float topP,
      int topK,
      std::function<bool(int64_t)> onToken);

  void applyTemperature(std::span<float> logits, float temperature);
  void applyTopK(std::span<float> logits, int k);
  void applyTopP(std::span<float> logits, float p);
  void softmax(std::span<float> logits);

  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  Ort::AllocatorWithDefaultOptions allocator;

  std::unique_ptr<Ort::Session> modelSession;
  OrtxTokenizer* tokenizer{};

  // Applies the model's own chat template to a single user message;
  // falls back to Qwen-style ChatML when the model does not ship one.
  std::string applyChatTemplate(const std::string& userPrompt) const;

  // Token ids that end the reply, from generation_config.json /
  // config.json; defaults to Qwen's <|endoftext|> + <|im_end|>.
  std::vector<int64_t> stopTokenIds{151643, 151645};

  // Graph geometry discovered at load time.
  int numLayers{};
  int64_t kvHeads{};
  int64_t headDim{};
  ONNXTensorElementDataType kvType{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  bool hasPositionIds{};

  // Input/output names
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  std::vector<const char*> inputNamePtrs;
  std::vector<const char*> outputNamePtrs;

  // Per-layer KV cache, stored as raw bytes in the model's own dtype: it
  // only ever round-trips from the outputs to the inputs of the next step,
  // so it never needs converting.
  std::vector<std::vector<std::byte>> keyCache;
  std::vector<std::vector<std::byte>> valueCache;
  std::vector<std::vector<int64_t>> cacheShapes;

  std::vector<int64_t> reusableAttentionMask;
  std::vector<int64_t> reusablePositionIds;
};
}
