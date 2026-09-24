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
      int topK = 40,
      bool thinking = true);

  // thinking = false asks a reasoning model (see supportsThinking) for its
  // reply without the <think> block, as Qwen3's enable_thinking=false does.
  void generateStreaming(
      const std::string& prompt,
      std::function<bool(const std::string&)> tokenCallback,
      int maxTokens = 100,
      float temperature = 0.7f,
      float topP = 0.9f,
      int topK = 40,
      bool thinking = true);

  // The token ids the model receives for a prompt (for tests).
  std::vector<int64_t> promptTokens(const std::string& prompt, bool thinking = true) const
  {
    return tokenize(applyChatTemplate(prompt, thinking));
  }

  // The tokenizer knows <think> and </think>: Qwen3, DeepSeek-R1, ...
  bool supportsThinking() const noexcept { return thinkingModel; }

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
      bool thinking,
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
  // Then closes the think block right away when thinking is off.
  std::string
  applyChatTemplate(const std::string& userPrompt, bool thinking) const;

  // Token ids that end the reply, from generation_config.json /
  // config.json; defaults to Qwen's <|endoftext|> + <|im_end|>.
  std::vector<int64_t> stopTokenIds{151643, 151645};

  bool thinkingModel = false;
  // Replace SentencePiece's U+2581 left in the decoded text (see HfConfig).
  bool spaceMarker = false;

  // Every decoder input is bound by name, in the session's own order: the
  // exports do not agree on it, and a positional binding silently swapped
  // tensors of the same type and shape (key/value, or the mask and the
  // positions).
  enum class InputRole : uint8_t
  {
    Ids,          // input_ids
    Mask,         // attention_mask
    Positions,    // position_ids
    LogitsToKeep, // num_logits_to_keep: 1, only the last row is read
    State,        // a recurrent state, see StateSlot
  };
  struct InputSlot
  {
    InputRole role{};
    int state = -1; // index into states for InputRole::State
  };
  std::vector<InputSlot> inputSlots;

  // A state carried from an output to an input at every step: the KV cache
  // (past_key_values.N.key|value <- present.N.key|value), which grows by one
  // position per step, and LFM2's convolution state (past_conv.N <-
  // present_conv.N), which keeps its size. Raw bytes in the slot's own dtype:
  // it only round-trips, so it is never converted.
  struct StateSlot
  {
    std::string output;
    ONNXTensorElementDataType type{};
    std::vector<int64_t> initShape; // batch 1; the growing axis at 0
    std::vector<std::byte> data;
    std::vector<int64_t> shape;
  };
  std::vector<StateSlot> states;

  // Input/output names
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  std::vector<const char*> inputNamePtrs;
  // What each step asks for: "logits", then each state's output.
  std::vector<const char*> runOutputNames;

  std::vector<int64_t> reusableAttentionMask;
  std::vector<int64_t> reusablePositionIds;
  int64_t logitsToKeep = 1;
};
}
