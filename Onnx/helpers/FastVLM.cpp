#include "FastVLM.hpp"

// fmt is header-only and available in both the score and standalone builds;
// include it directly so this file stays free of any ossia/ include (it used to
// pull ossia/detail/fmt.hpp, a thin wrapper over <fmt/format.h>).
#if defined(check) // Thanks Unreal...
#undef check
#endif
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>
#include <cmath>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace Onnx
{
struct FastVLMTokenizerConstants
{
  static constexpr int BOS_TOKEN_ID = 151643;
  static constexpr int EOS_TOKEN_ID = 151645;
  static constexpr int IMAGE_TOKEN_INDEX
      = 151646; // <image> token from tokenizer config
  static constexpr int MAX_LENGTH = 8192;

  // String tokens
  static inline const std::string DEFAULT_IMAGE_TOKEN = "<image>";
};

static Onnx::FloatTensor preprocessImageForFastVLM(
    const Onnx::ImageData& image,
    boost::container::vector<float>& tensorValues)
{
  // Create a minimal ModelSpec::Port for the existing function
  ModelSpec::Port port;
  port.shape = {1, 3, image.height, image.width};
  static constexpr std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};
  static constexpr std::array<float, 3> std = {255.f, 255.f, 255.f};

  tensorValues.clear();

  return nchw_tensorFromRGBA(
      port,
      image.pixels.data(),
      image.width,
      image.height,
      image.width,
      image.height,
      tensorValues,
      mean,
      std);
}

static std::size_t tensorElementSize(ONNXTensorElementDataType t)
{
  switch (t)
  {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return 8;
    default:
      throw std::runtime_error(
          fmt::format("FastVLM: unsupported tensor element type {}", (int)t));
  }
}

// Read a float or float16 tensor into a flat fp32 vector.
static std::vector<float> tensorToFloats(const Ort::Value& v)
{
  auto info = v.GetTensorTypeAndShapeInfo();
  const std::size_t n = info.GetElementCount();
  std::vector<float> out(n);
  switch (info.GetElementType())
  {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      const auto* d = v.GetTensorData<Ort::Float16_t>();
      for (std::size_t i = 0; i < n; ++i)
        out[i] = d[i].ToFloat();
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      const auto* d = v.GetTensorData<float>();
      std::copy_n(d, n, out.begin());
      break;
    }
    default:
      throw std::runtime_error(
          fmt::format(
              "FastVLM: unsupported output element type {}",
              (int)info.GetElementType()));
  }
  return out;
}

FastVLMInference::FastVLMInference(
    std::string_view visionEncoderPath,
    std::string_view embedTokensPath,
    std::string_view decoderPath,
    std::string_view tokenizerModelPath)
    : env(Onnx::make_env("FastVLMInference"))
    , tokenizer(nullptr)
{
  Onnx::Options oopts;
  sessionOptions = Onnx::create_session_options(oopts);

  try
  {
#if defined(_WIN32)
    // ORT wants wide paths on Windows: decode the UTF-8 byte paths to wstring.
    auto model0_str
        = std::filesystem::path(std::u8string(
                                    reinterpret_cast<const char8_t*>(
                                        visionEncoderPath.data()),
                                    visionEncoderPath.size()))
              .wstring();
    auto model1_str
        = std::filesystem::path(std::u8string(
                                    reinterpret_cast<const char8_t*>(
                                        embedTokensPath.data()),
                                    embedTokensPath.size()))
              .wstring();
    auto model2_str
        = std::filesystem::path(std::u8string(
                                    reinterpret_cast<const char8_t*>(
                                        decoderPath.data()),
                                    decoderPath.size()))
              .wstring();
#else
    auto model0_str = visionEncoderPath;
    auto model1_str = embedTokensPath;
    auto model2_str = decoderPath;
#endif
    // Load separate ONNX models from onnx/ directory
    visionEncoderSession
        = create_session_with_fallback(env, model0_str, sessionOptions);
    embedTokensSession
        = create_session_with_fallback(env, model1_str, sessionOptions);
    decoderSession
        = create_session_with_fallback(env, model2_str, sessionOptions);

    if (tokenizerModelPath.ends_with("tokenizer.json"))
      tokenizerModelPath = tokenizerModelPath.substr(
          0, tokenizerModelPath.size() - strlen("tokenizer.json"));

    extError_t result = OrtxCreateTokenizer(
        &tokenizer, std::string(tokenizerModelPath).c_str());
    if (result != kOrtxOK)
    {
      throw std::runtime_error(
          fmt::format(
              "Failed to create Ortx tokenizer: {}",
              OrtxGetLastErrorMessage()));
    }

    // Cache decoder I/O names for hot path optimization
    decoderInputNames.reserve(decoderSession->GetInputCount());
    decoderOutputNames.reserve(decoderSession->GetOutputCount());
    decoderInputNamePtrs.reserve(decoderSession->GetInputCount());
    decoderOutputNamePtrs.reserve(decoderSession->GetOutputCount());

    for (size_t i = 0; i < decoderSession->GetInputCount(); ++i)
    {
      decoderInputNames.push_back(
          decoderSession->GetInputNameAllocated(i, allocator).get());
      decoderInputNamePtrs.push_back(decoderInputNames.back().c_str());
    }

    for (size_t i = 0; i < decoderSession->GetOutputCount(); ++i)
    {
      decoderOutputNames.push_back(
          decoderSession->GetOutputNameAllocated(i, allocator).get());
      decoderOutputNamePtrs.push_back(decoderOutputNames.back().c_str());
    }

    // Discover the decoder graph properties this export variant was built
    // with. Inputs are inputs_embeds, attention_mask, position_ids, then
    // (past_key_values.N.key, past_key_values.N.value) per layer.
    const std::size_t nIn = decoderSession->GetInputCount();
    numLayers = nIn >= 3 ? int((nIn - 3) / 2) : 0;
    if (numLayers <= 0)
      throw std::runtime_error(
          fmt::format(
              "Unexpected decoder input count: {} (not a merged FastVLM "
              "decoder?)",
              nIn));
    keyCache.assign(numLayers, {});
    valueCache.assign(numLayers, {});
    cacheShapes.assign(numLayers, {});

    for (std::size_t i = 0; i < nIn; ++i)
    {
      // Keep the TypeInfo alive: GetTensorTypeAndShapeInfo() is a view on it.
      const auto typeInfo = decoderSession->GetInputTypeInfo(i);
      const auto info = typeInfo.GetTensorTypeAndShapeInfo();
      if (decoderInputNames[i] == "inputs_embeds")
      {
        embedsType = info.GetElementType();
        if (const auto sh = info.GetShape(); sh.size() == 3 && sh[2] > 0)
          hiddenSize = sh[2];
      }
      else if (decoderInputNames[i] == "past_key_values.0.key")
      {
        // Shape is [batch, kv_heads, past_seq_len, head_dim]; batch and
        // past_seq_len are symbolic (-1), the others are concrete.
        kvType = info.GetElementType();
        if (const auto sh = info.GetShape(); sh.size() == 4)
        {
          if (sh[1] > 0)
            kvHeads = sh[1];
          if (sh[3] > 0)
            headDim = sh[3];
        }
      }
    }

    {
      const auto visionTypeInfo = visionEncoderSession->GetInputTypeInfo(0);
      visionInputType
          = visionTypeInfo.GetTensorTypeAndShapeInfo().GetElementType();

      const auto embedTypeInfo = embedTokensSession->GetOutputTypeInfo(0);
      if (const auto sh
          = embedTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
          sh.size() == 3 && sh[2] > 0)
        hiddenSize = sh[2];
    }
  }
  catch (const Ort::Exception& e)
  {
    throw std::runtime_error(
        fmt::format("Failed to load ONNX models: {}", e.what()));
  }
}

FastVLMInference::~FastVLMInference()
{
  if (tokenizer)
  {
    OrtxDispose((OrtxObject**)&tokenizer);
  }
}

std::string FastVLMInference::generateResponse(
    const Onnx::ImageData& image,
    const std::string& prompt,
    float temperature,
    int maxTokens)
{
  try
  {
    // Process image through vision encoder using existing Images.hpp functions
    auto processedImageTensor = preprocessImageForFastVLM(image, tensorValues);

    auto imageFeatures = runVisionEncoder(
        processedImageTensor.storage, image.width, image.height);
    std::swap(processedImageTensor.storage, tensorValues);
    if (imageFeatures.empty())
      throw std::runtime_error("No image feature");
    if (!std::isfinite(imageFeatures[0]))
      throw std::runtime_error("Anormal feature");

    // Create the prompt with proper chat template (like HuggingFace)
    std::string formattedPrompt = createPromptTemplate(prompt);

    // Tokenize the prompt with proper image token handling
    auto tokenIds = tokenizeImagePrompt(formattedPrompt);

    // Create proper multimodal embeddings by replacing IMAGE_TOKEN with vision feature
    auto multimodalEmbeddings
        = createMultimodalEmbeddings(tokenIds, imageFeatures);

    maxTokens = std::clamp(maxTokens, 1, FastVLMTokenizerConstants::MAX_LENGTH);

    // Generate tokens using our working ONNX decoder with temperature sampling
    auto generatedTokens
        = generateWithONNXDecoder(multimodalEmbeddings, maxTokens, temperature);

    // Step 6: Decode the generated tokens
    std::string response = decodeTokens(generatedTokens);

    // Extract just the assistant's response from the full sequence
    size_t assistantPos = response.find("<|im_start|>assistant\n");
    if (assistantPos != std::string::npos)
    {
      // Skip "<|im_start|>assistant\n"
      response = response.substr(assistantPos + 22);

      // Remove any trailing tokens
      size_t endPos = response.find("<|im_end|>");
      if (endPos != std::string::npos)
      {
        response = response.substr(0, endPos);
      }
    }

    return response;
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error(
        fmt::format("Multimodal inference failed: {}", e.what()));
  }
}

std::vector<float>
FastVLMInference::runVisionEncoder(std::span<float> imageData, int w, int h)
{
  try
  {
    auto memoryInfo
        = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> inputShape = {1, 3, h, w};

    Ort::Value inputTensor{nullptr};
    if (visionInputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
      reusableF16Scratch.resize(imageData.size());
      for (std::size_t i = 0; i < imageData.size(); ++i)
        reusableF16Scratch[i] = Ort::Float16_t(imageData[i]);
      inputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(
          memoryInfo,
          reusableF16Scratch.data(),
          reusableF16Scratch.size(),
          inputShape.data(),
          inputShape.size());
    }
    else
    {
      inputTensor = Ort::Value::CreateTensor<float>(
          memoryInfo,
          const_cast<float*>(imageData.data()),
          imageData.size(),
          inputShape.data(),
          inputShape.size());
    }

    auto inputName = visionEncoderSession->GetInputNameAllocated(0, allocator);
    auto outputName
        = visionEncoderSession->GetOutputNameAllocated(0, allocator);

    const char* inputNames[] = {inputName.get()};
    const char* outputNames[] = {outputName.get()};

    auto outputs = visionEncoderSession->Run(
        Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    return tensorToFloats(outputs[0]);
  }
  catch (const Ort::Exception& e)
  {
    throw std::runtime_error(
        fmt::format("Vision encoder failed: {}", e.what()));
  }
}

std::vector<float>
FastVLMInference::runEmbedTokens(std::span<int64_t> tokenIds)
{
  try
  {
    auto memoryInfo
        = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> inputShape
        = {1, static_cast<int64_t>(tokenIds.size())};
    auto inputTensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo,
        const_cast<int64_t*>(tokenIds.data()),
        tokenIds.size(),
        inputShape.data(),
        inputShape.size());

    auto inputName = embedTokensSession->GetInputNameAllocated(0, allocator);
    auto outputName = embedTokensSession->GetOutputNameAllocated(0, allocator);

    const char* inputNames[] = {inputName.get()};
    const char* outputNames[] = {outputName.get()};

    auto outputs = embedTokensSession->Run(
        Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    return tensorToFloats(outputs[0]);
  }
  catch (const Ort::Exception& e)
  {
    throw std::runtime_error(fmt::format("Embed tokens failed: {}", e.what()));
  }
}

std::string FastVLMInference::decodeTokens(std::span<int64_t> tokens) const
{
  try
  {
    std::vector<extTokenId_t> ortxTokens;
    ortxTokens.reserve(tokens.size());

    for (int64_t token : tokens)
    {
      ortxTokens.push_back(static_cast<extTokenId_t>(token));
    }

    OrtxStringArray* stringArray = nullptr;
    extError_t result = OrtxDetokenize1D(
        tokenizer, ortxTokens.data(), ortxTokens.size(), &stringArray);
    if (result != kOrtxOK)
    {
      throw std::runtime_error(
          fmt::format("Token decoding failed: {}", OrtxGetLastErrorMessage()));
    }

    // Get the decoded string from the array (should have one item)
    const char* decodedString = nullptr;
    result = OrtxStringArrayGetItem(stringArray, 0, &decodedString);
    if (result != kOrtxOK)
    {
      OrtxDispose((OrtxObject**)&stringArray);
      throw std::runtime_error(
          fmt::format(
              "Failed to get decoded string: {}", OrtxGetLastErrorMessage()));
    }

    std::string resultStr(decodedString);
    OrtxDispose((OrtxObject**)&stringArray);
    return resultStr;
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error(
        fmt::format("Token decoding failed: {}", e.what()));
  }
}

std::vector<int64_t>
FastVLMInference::tokenizeImagePrompt(const std::string& prompt) const
{
  try
  {
    // Split the prompt by <image> tokens, similar to Python's tokenizer_image_token
    std::vector<std::string> chunks;
    size_t pos = 0;

    while (pos < prompt.length())
    {
      size_t imagePos
          = prompt.find(FastVLMTokenizerConstants::DEFAULT_IMAGE_TOKEN, pos);
      if (imagePos == std::string::npos)
      {
        // No more image tokens, add the rest
        chunks.push_back(prompt.substr(pos));
        break;
      }

      // Add text before image token
      if (imagePos > pos)
      {
        chunks.push_back(prompt.substr(pos, imagePos - pos));
      }

      // Add empty string to mark where image token was
      chunks.push_back("");

      pos = imagePos + FastVLMTokenizerConstants::DEFAULT_IMAGE_TOKEN.length();
    }

    std::vector<int64_t> finalTokens;

    for (size_t i = 0; i < chunks.size(); ++i)
    {
      if (i > 0 && i % 2 == 1)
      {
        // This is where an image token was, insert IMAGE_TOKEN_INDEX
        finalTokens.push_back(FastVLMTokenizerConstants::IMAGE_TOKEN_INDEX);
      }
      else if (!chunks[i].empty())
      {
        // Tokenize text chunk using Ortx
        const char* inputTexts[] = {chunks[i].c_str()};
        OrtxTokenId2DArray* tokenArray = nullptr;

        extError_t result
            = OrtxTokenize(tokenizer, inputTexts, 1, &tokenArray);
        if (result != kOrtxOK)
        {
          throw std::runtime_error(
              fmt::format(
                  "Tokenization failed: {}", OrtxGetLastErrorMessage()));
        }

        // Get tokens from the first sequence
        const extTokenId_t* tokenData = nullptr;
        size_t tokenCount = 0;
        result = OrtxTokenId2DArrayGetItem(
            tokenArray, 0, &tokenData, &tokenCount);
        if (result != kOrtxOK)
        {
          OrtxDispose((OrtxObject**)&tokenArray);
          throw std::runtime_error(
              fmt::format(
                  "Failed to get tokens: {}", OrtxGetLastErrorMessage()));
        }

        // Handle BOS token - only add at the very beginning
        size_t startIdx = 0;
        if (i == 0 && tokenCount > 0
            && tokenData[0] == FastVLMTokenizerConstants::BOS_TOKEN_ID)
        {
          finalTokens.push_back(static_cast<int64_t>(tokenData[0]));
          startIdx = 1;
        }

        for (size_t j = startIdx; j < tokenCount; ++j)
        {
          finalTokens.push_back(static_cast<int64_t>(tokenData[j]));
        }

        OrtxDispose((OrtxObject**)&tokenArray);
      }
    }

    return finalTokens;
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error(
        fmt::format("Image tokenization failed: {}", e.what()));
  }
}

std::string
FastVLMInference::createPromptTemplate(std::string_view userPrompt) const
{
  return fmt::format(
      "<|im_start|>system\n"
      "You are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\n"
      "<image>\n{}<|im_end|>\n"
      "<|im_start|>assistant\n",
      userPrompt);
}

std::vector<float> FastVLMInference::createMultimodalEmbeddings(
    std::span<int64_t> tokenIds,
    std::span<float> imageFeatures)
{
  try
  {
    std::vector<float> multimodalEmbeddings;
    multimodalEmbeddings.reserve(
        tokenIds.size() * hiddenSize + imageFeatures.size());

    // Embed the text in contiguous segments (one embed_tokens run per
    // segment instead of one per token), splicing the image features in
    // place of each image token.
    std::vector<int64_t> segment;
    segment.reserve(tokenIds.size());
    auto flushSegment = [&]
    {
      if (segment.empty())
        return;
      auto emb = runEmbedTokens(segment);
      multimodalEmbeddings.insert(
          multimodalEmbeddings.end(), emb.begin(), emb.end());
      segment.clear();
    };

    for (int64_t id : tokenIds)
    {
      if (id == FastVLMTokenizerConstants::IMAGE_TOKEN_INDEX)
      {
        flushSegment();
        multimodalEmbeddings.insert(
            multimodalEmbeddings.end(),
            imageFeatures.begin(),
            imageFeatures.end());
      }
      else
      {
        segment.push_back(id);
      }
    }
    flushSegment();

    return multimodalEmbeddings;
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error(
        fmt::format("Failed to create multimodal embeddings: {}", e.what()));
  }
}

std::vector<int64_t> FastVLMInference::generateWithONNXDecoder(
    std::span<float> embeddings,
    int maxTokens,
    float temperature)
{
  try
  {
    const size_t seqLen = embeddings.size() / hiddenSize;

    auto memoryInfo
        = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // The decoder may want fp32 or fp16 inputs_embeds depending on the
    // export; our pipeline-internal representation is fp32.
    auto createEmbedsTensor
        = [&memoryInfo, this](std::span<float> embeds, size_t seqLen)
    {
      std::vector<int64_t> shape
          = {1, static_cast<int64_t>(seqLen), hiddenSize};
      if (embedsType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
      {
        reusableF16Scratch.resize(embeds.size());
        for (std::size_t i = 0; i < embeds.size(); ++i)
          reusableF16Scratch[i] = Ort::Float16_t(embeds[i]);
        return Ort::Value::CreateTensor<Ort::Float16_t>(
            memoryInfo,
            reusableF16Scratch.data(),
            reusableF16Scratch.size(),
            shape.data(),
            shape.size());
      }
      return Ort::Value::CreateTensor<float>(
          memoryInfo,
          const_cast<float*>(embeds.data()),
          embeds.size(),
          shape.data(),
          shape.size());
    };

    auto createAttentionTensor = [&memoryInfo, this](size_t seqLen)
    {
      reusableAttentionMask.assign(seqLen, 1);
      std::vector<int64_t> shape = {1, static_cast<int64_t>(seqLen)};
      return Ort::Value::CreateTensor<int64_t>(
          memoryInfo,
          reusableAttentionMask.data(),
          reusableAttentionMask.size(),
          shape.data(),
          shape.size());
    };

    auto createPositionTensor = [&memoryInfo, this]()
    {
      std::vector<int64_t> shape
          = {1, static_cast<int64_t>(reusablePositionIds.size())};
      return Ort::Value::CreateTensor<int64_t>(
          memoryInfo,
          reusablePositionIds.data(),
          reusablePositionIds.size(),
          shape.data(),
          shape.size());
    };

    // KV cache tensors are created in the decoder's own dtype straight over
    // the raw cache bytes.
    auto createKVCacheTensor = [&memoryInfo, this](
                                   std::vector<std::byte>& cache,
                                   const std::vector<int64_t>& shape)
    {
      return Ort::Value::CreateTensor(
          memoryInfo,
          cache.data(),
          cache.size(),
          shape.data(),
          shape.size(),
          kvType);
    };

    // Create initial tensors using helper functions
    auto embedsTensor = createEmbedsTensor(embeddings, seqLen);
    auto attentionTensor = createAttentionTensor(seqLen);

    reusablePositionIds.resize(seqLen);
    std::iota(reusablePositionIds.begin(), reusablePositionIds.end(), 0);
    auto positionTensor = createPositionTensor();

    // Prepare inputs vector with the three main inputs
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(embedsTensor));
    inputs.push_back(std::move(attentionTensor));
    inputs.push_back(std::move(positionTensor));

    // Create empty KV cache tensors for every layer
    std::vector<std::byte> emptyCache;
    std::vector<int64_t> emptyCacheShape = {1, kvHeads, 0, headDim};
    for (int layer = 0; layer < numLayers; ++layer)
    {
      inputs.push_back(createKVCacheTensor(emptyCache, emptyCacheShape));
      inputs.push_back(createKVCacheTensor(emptyCache, emptyCacheShape));
    }

    // Run the decoder
    auto outputs = decoderSession->Run(
        Ort::RunOptions{nullptr},
        decoderInputNamePtrs.data(),
        inputs.data(),
        std::min(inputs.size(), decoderInputNamePtrs.size()),
        decoderOutputNamePtrs.data(),
        decoderOutputNamePtrs.size());

    if (outputs.empty())
    {
      return {};
    }

    // Random number generator for temperature sampling
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());

    // Temperature-based token sampling function; logits may be fp32 or fp16.
    std::vector<float> logitsRow;
    auto sampleToken
        = [temperature, &logitsRow](
              const Ort::Value& logitsTensor) -> std::pair<int64_t, float>
    {
      auto info = logitsTensor.GetTensorTypeAndShapeInfo();
      auto logitsShape = info.GetShape();
      size_t vocabSize = logitsShape[2];
      size_t lastPos = (logitsShape[1] - 1) * vocabSize;

      const float* logitsData;
      if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
      {
        const auto* d = logitsTensor.GetTensorData<Ort::Float16_t>();
        logitsRow.resize(vocabSize);
        for (size_t i = 0; i < vocabSize; ++i)
          logitsRow[i] = d[lastPos + i].ToFloat();
        logitsData = logitsRow.data();
        lastPos = 0;
      }
      else
      {
        logitsData = logitsTensor.GetTensorData<float>();
      }

      if (temperature <= 0.0f)
      {
        // Greedy sampling when temperature is 0
        float maxLogit = logitsData[lastPos];
        int64_t bestToken = 0;

        for (size_t i = 1; i < vocabSize; ++i)
        {
          if (logitsData[lastPos + i] > maxLogit)
          {
            maxLogit = logitsData[lastPos + i];
            bestToken = static_cast<int64_t>(i);
          }
        }
        return {bestToken, maxLogit};
      }

      // Apply temperature scaling and convert to probabilities
      std::vector<float> probabilities(vocabSize);
      float maxLogit = *std::max_element(
          logitsData + lastPos, logitsData + lastPos + vocabSize);

      // Apply temperature and calculate softmax
      float sumExp = 0.0f;
      for (size_t i = 0; i < vocabSize; ++i)
      {
        float scaledLogit = (logitsData[lastPos + i] - maxLogit) / temperature;
        probabilities[i] = std::exp(scaledLogit);
        sumExp += probabilities[i];
      }

      // Normalize probabilities
      for (size_t i = 0; i < vocabSize; ++i)
      {
        probabilities[i] /= sumExp;
      }

      // Sample from the probability distribution
      std::discrete_distribution<size_t> dist(
          probabilities.begin(), probabilities.end());
      size_t sampledToken = dist(gen);

      return {
          static_cast<int64_t>(sampledToken),
          logitsData[lastPos + sampledToken]};
    };

    // Copy the present.* outputs into the byte-level KV cache.
    const std::size_t kvElemSize = tensorElementSize(kvType);
    auto storeKVCache = [this, kvElemSize](std::vector<Ort::Value>& outs)
    {
      for (int layer = 0; layer < numLayers; ++layer)
      {
        auto& keyTensor = outs[1 + layer * 2]; // present.{layer}.key
        const auto keyInfo = keyTensor.GetTensorTypeAndShapeInfo();
        const auto keyShape = keyInfo.GetShape();
        const std::size_t bytes = keyInfo.GetElementCount() * kvElemSize;

        const auto* keyData
            = static_cast<const std::byte*>(keyTensor.GetTensorRawData());
        keyCache[layer].assign(keyData, keyData + bytes);
        cacheShapes[layer].assign(keyShape.begin(), keyShape.end());

        auto& valueTensor = outs[1 + layer * 2 + 1]; // present.{layer}.value
        const auto* valueData
            = static_cast<const std::byte*>(valueTensor.GetTensorRawData());
        valueCache[layer].assign(valueData, valueData + bytes);
      }
    };

    auto [bestToken, maxLogit] = sampleToken(outputs[0]);

    // Store the generated tokens
    std::vector<int64_t> generatedTokens = {bestToken};

    // Check for EOS token
    if (bestToken == FastVLMTokenizerConstants::EOS_TOKEN_ID)
    {
      return generatedTokens;
    }

    // Store KV cache from first generation step
    storeKVCache(outputs);

    // Continue generation for remaining tokens
    for (int step = 1; step < maxTokens; ++step)
    {
      // Get embedding for the new token
      auto tokenEmbedding = runEmbedTokens({&bestToken, 1});

      // Create inputs for next step using helper functions
      std::vector<Ort::Value>& nextInputs = inputs;
      nextInputs.clear();

      nextInputs.push_back(createEmbedsTensor(tokenEmbedding, 1));

      size_t currentSeqLen = seqLen + step;
      nextInputs.push_back(createAttentionTensor(currentSeqLen));

      reusablePositionIds.clear();
      reusablePositionIds.push_back(static_cast<int64_t>(seqLen + step - 1));
      nextInputs.push_back(createPositionTensor());

      // Add KV cache from previous step
      for (int layer = 0; layer < numLayers; ++layer)
      {
        nextInputs.push_back(
            createKVCacheTensor(keyCache[layer], cacheShapes[layer]));
        nextInputs.push_back(
            createKVCacheTensor(valueCache[layer], cacheShapes[layer]));
      }

      // Run decoder for next token
      auto nextOutputs = decoderSession->Run(
          Ort::RunOptions{nullptr},
          decoderInputNamePtrs.data(),
          nextInputs.data(),
          std::min(nextInputs.size(), decoderInputNamePtrs.size()),
          decoderOutputNamePtrs.data(),
          decoderOutputNamePtrs.size());

      // Get next token from logits using temperature sampling
      auto [nextBestToken, nextMaxLogit] = sampleToken(nextOutputs[0]);
      bestToken = nextBestToken;

      generatedTokens.push_back(bestToken);

      // Check for EOS token
      if (bestToken == FastVLMTokenizerConstants::EOS_TOKEN_ID)
      {
        break;
      }

      // Update KV cache for next iteration
      storeKVCache(nextOutputs);
    }

    return generatedTokens;
  }
  catch (const Ort::Exception& e)
  {
    throw std::runtime_error(fmt::format("ONNX decoder error: {}", e.what()));
  }
}
}
