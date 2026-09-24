#pragma once

#include <boost/container/vector.hpp>

#include <Onnx/helpers/ImageBuffer.hpp>

#include <onnxruntime_cxx_api.h>
#include <ortx_tokenizer.h>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace Onnx
{
// Runs the three-model VLM pipeline (vision encoder, token embeddings,
// merged decoder) of the onnx-community exports: FastVLM (LLaVA-Qwen2),
// SmolVLM (idefics3) and gemma3. The family (config.json model_type) decides
// the image preprocessing and how the prompt's image placeholder expands;
// see Family. The pipeline
// adapts to the export variant at load time: layer count, KV heads, head
// dimension and hidden size are read from the model graphs, and the KV cache
// dtype (fp16 for the _fp16 / _q4f16 exports, fp32 for all others) as well as
// fp16 embeddings / logits are handled transparently.
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
      const Onnx::ImageData& image,
      const std::string& prompt,
      float temperature = 1.0f,
      int maxTokens = 500);

private:
  std::vector<int64_t> tokenizeImagePrompt(const std::string& prompt) const;
  // Image -> the vision encoder's input tensor values ([3,H,W] planar).
  std::vector<float> preprocess(const Onnx::ImageData& image, int& w, int& h) const;
  std::vector<float>
  runVisionEncoder(std::span<float> imageData, int w, int h);
  std::vector<float> runEmbedTokens(std::span<int64_t> tokenIds);
  std::string decodeTokens(std::span<int64_t> tokens) const;

  // The model's own chat template (tokenizer_config.json /
  // chat_template.jinja) applied to one user message with the image
  // placeholder in front; Qwen2 ChatML when it ships none.
  std::string createPromptTemplate(std::string_view userPrompt) const;
  std::vector<int64_t> tokenizeText(const std::string& text) const;

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

  boost::container::vector<float> tensorValues;

  // From the files next to the tokenizer (config.json,
  // generation_config.json, processor_config.json); the defaults are
  // FastVLM's (LLaVA-Qwen2) values.
  std::string imageToken{"<image>"};
  int64_t imageTokenId{151646}; // a sentinel past the vocabulary
  std::vector<int64_t> stopTokenIds{151645};
  bool spaceMarker = false; // U+2581 left by the detokenizer (HfConfig)

  // How the image reaches the model:
  // - Llava (FastVLM): the native image size, 0..1; one image token that
  //   stands for all the features.
  // - Idefics3 (SmolVLM): one global 512² image, ±1, as pixel_values
  //   [1,1,3,512,512] with a pixel_attention_mask; the template's <image>
  //   expands to <fake_token_around_image><global-img>, image_seq_len image
  //   tokens, <fake_token_around_image>. The processor's tiling of large
  //   images (up to 16 more 512² tiles, 64 tokens each) is not done.
  // - Gemma3: 896², ±1; the template's <start_of_image> expands to
  //   \n\n<start_of_image>, mm_tokens_per_image <image_soft_token>s,
  //   <end_of_image>\n\n.
  // With several image tokens, each takes one row of the features.
  enum class Family : uint8_t
  {
    Llava,
    Idefics3,
    Gemma3
  } family{Family::Llava};
  int imageSize = 0; // square input side; 0 = the image's own size
  float imageMean[3]{0.f, 0.f, 0.f};
  float imageStd[3]{1.f, 1.f, 1.f};
  std::string placeholder;          // what the chat template emits
  std::string placeholderExpansion; // what it becomes; empty = kept

public:
  // Which family the model files describe.
  const char* familyName() const noexcept
  {
    return family == Family::Idefics3 ? "idefics3"
           : family == Family::Gemma3 ? "gemma3"
                                      : "llava";
  }

public:
  const std::string& imagePlaceholder() const noexcept { return imageToken; }
  int64_t imagePlaceholderId() const noexcept { return imageTokenId; }
  std::span<const int64_t> stopTokens() const noexcept { return stopTokenIds; }
  std::string promptFor(std::string_view userPrompt) const
  {
    return createPromptTemplate(userPrompt);
  }
  // The token ids the decoder receives for a prompt, the image tokens
  // included (for tests).
  std::vector<int64_t> promptTokens(std::string_view userPrompt) const
  {
    return tokenizeImagePrompt(createPromptTemplate(userPrompt));
  }

private:
  // Every decoder input is bound by name: exports differ in which of
  // position_ids / num_logits_to_keep they have, and in their order.
  enum class DecoderInput : uint8_t
  {
    Embeds,       // inputs_embeds
    Mask,         // attention_mask
    Positions,    // position_ids
    LogitsToKeep, // num_logits_to_keep
    Key,          // past_key_values.N.key
    Value,        // past_key_values.N.value
  };
  struct DecoderSlot
  {
    DecoderInput role{};
    int layer = -1;
  };
  std::vector<DecoderSlot> decoderSlots;
  int logitsOutput = 0;
  std::vector<int> presentKeyOutput, presentValueOutput; // per layer

  // Decoder graph properties discovered at load time.
  int numLayers{};
  int64_t kvHeads{2};
  int64_t headDim{64};
  int64_t hiddenSize{896};
  ONNXTensorElementDataType visionInputType{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  ONNXTensorElementDataType embedsType{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  ONNXTensorElementDataType kvType{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};

  // Per-layer KV cache, stored as raw bytes in the decoder's own dtype: the
  // cache only ever round-trips from the decoder's outputs to its inputs, so
  // it never needs converting.
  std::vector<std::vector<std::byte>> keyCache;
  std::vector<std::vector<std::byte>> valueCache;
  std::vector<std::vector<int64_t>> cacheShapes;

  std::vector<std::string> decoderInputNames;
  std::vector<std::string> decoderOutputNames;
  std::vector<const char*> decoderInputNamePtrs;
  std::vector<const char*> decoderOutputNamePtrs;
  std::vector<int64_t> reusableAttentionMask;
  std::vector<int64_t> reusablePositionIds;
  std::vector<Ort::Float16_t> reusableF16Scratch;
};
}
