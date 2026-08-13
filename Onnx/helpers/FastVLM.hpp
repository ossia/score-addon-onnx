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
// Runs the three-model FastVLM pipeline (vision encoder, token embeddings,
// merged decoder) exported by onnx-community/FastVLM-*-ONNX. The pipeline
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
  std::vector<float>
  runVisionEncoder(std::span<float> imageData, int w, int h);
  std::vector<float> runEmbedTokens(std::span<int64_t> tokenIds);
  std::string decodeTokens(std::span<int64_t> tokens) const;

  std::string createPromptTemplate(std::string_view userPrompt) const;

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
