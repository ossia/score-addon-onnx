#include "FastVLM.hpp"

#include <QImage>
#include <QRect>

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>
#include <cmath>

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <iostream>
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
  static constexpr int IGNORE_INDEX
      = -100; // For labels to ignore during training
  static constexpr int IM_START_TOKEN_ID = 151644;
  static constexpr int IM_END_TOKEN_ID = 151645;
  static constexpr int MAX_LENGTH = 8192;
  static constexpr int VOCAB_SIZE = 151646;
  static constexpr int HIDDEN_SIZE = 896;
  static constexpr int MM_HIDDEN_SIZE = 3072;

  // String tokens
  static inline const std::string DEFAULT_IMAGE_TOKEN = "<image>";
  static inline const std::string DEFAULT_IM_START_TOKEN = "<im_start>";
  static inline const std::string DEFAULT_IM_END_TOKEN = "<im_end>";
  static inline const std::string IMAGE_PLACEHOLDER = "<image-placeholder>";
};

static Onnx::FloatTensor preprocessImageForFastVLM(
    const QImage& image,
    boost::container::vector<float>& tensorValues)
{
  // Create a minimal ModelSpec::Port for the existing function
  ModelSpec::Port port;
  port.shape = {1, 3, image.height(), image.width()};
  static constexpr std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};
  static constexpr std::array<float, 3> std = {255.f, 255.f, 255.f};

  tensorValues.clear();

  return nchw_tensorFromRGBA(
      port,
      image.constBits(),
      image.width(),
      image.height(),
      image.width(),
      image.height(),
      tensorValues,
      mean,
      std);
}

FastVLMInference::FastVLMInference(
    std::string_view visionEncoderPath,
    std::string_view embedTokensPath,
    std::string_view decoderPath,
    std::string_view tokenizerModelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "FastVLMInference")
    , tokenizer(nullptr)
{
  Onnx::Options oopts;
  sessionOptions = Onnx::create_session_options(oopts);

  try
  {
#if defined(_WIN32)
    auto model0 = QString::fromUtf8(visionEncoderPath);
    auto model1 = QString::fromUtf8(embedTokensPath);
    auto model2 = QString::fromUtf8(decoderPath);
    auto model0_str = model0.toStdWString();
    auto model1_str = model1.toStdWString();
    auto model2_str = model2.toStdWString();
#else
    auto model0_str = visionEncoderPath;
    auto model1_str = embedTokensPath;
    auto model2_str = decoderPath;
#endif
    // Load separate ONNX models from onnx/ directory
    visionEncoderSession = std::make_unique<Ort::Session>(
        env, model0_str.data(), sessionOptions);
    embedTokensSession = std::make_unique<Ort::Session>(
        env, model1_str.data(), sessionOptions);
    decoderSession = std::make_unique<Ort::Session>(
        env, model2_str.data(), sessionOptions);

    if (tokenizerModelPath.ends_with("tokenizer.json"))
      tokenizerModelPath = tokenizerModelPath.substr(
          0, tokenizerModelPath.size() - strlen("tokenizer.json"));

    extError_t result = OrtxCreateTokenizer(
        &tokenizer, std::string(tokenizerModelPath).c_str());
    if (result != kOrtxOK)
    {
      throw std::runtime_error(
          std::format(
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
  }
  catch (const Ort::Exception& e)
  {
    throw std::runtime_error(
        std::format("Failed to load ONNX models: {}", e.what()));
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
    const QImage& image,
    const std::string& prompt,
    float temperature)
{
  try
  {
    // Process image through vision encoder using existing Images.hpp functions
    auto processedImageTensor = preprocessImageForFastVLM(image, tensorValues);

    auto imageFeatures = runVisionEncoder(
        processedImageTensor.storage, image.width(), image.height());
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

    // Generate tokens using our working ONNX decoder with temperature sampling
    auto generatedTokens
        = generateWithONNXDecoder(multimodalEmbeddings, 500, temperature);

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
        std::format("Manual multimodal inference failed: {}", e.what()));
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
    auto inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        const_cast<float*>(imageData.data()),
        imageData.size(),
        inputShape.data(),
        inputShape.size());

    auto inputName = visionEncoderSession->GetInputNameAllocated(0, allocator);
    auto outputName
        = visionEncoderSession->GetOutputNameAllocated(0, allocator);

    const char* inputNames[] = {inputName.get()};
    const char* outputNames[] = {outputName.get()};

    auto outputs = visionEncoderSession->Run(
        Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    auto outputTensor = outputs[0].GetTensorMutableData<float>();
    auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t outputSize = std::accumulate(
        outputShape.begin(),
        outputShape.end(),
        1ULL,
        std::multiplies<size_t>());

    return std::vector<float>(outputTensor, outputTensor + outputSize);
  }
  catch (const Ort::Exception& e)
  {
    throw std::runtime_error(
        std::format("Vision encoder failed: {}", e.what()));
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

    auto outputTensor = outputs[0].GetTensorMutableData<float>();
    auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t outputSize = std::accumulate(
        outputShape.begin(),
        outputShape.end(),
        1ULL,
        std::multiplies<size_t>());

    return std::vector<float>(outputTensor, outputTensor + outputSize);
  }
  catch (const Ort::Exception& e)
  {
    throw std::runtime_error(std::format("Embed tokens failed: {}", e.what()));
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
          std::format("Token decoding failed: {}", OrtxGetLastErrorMessage()));
    }

    // Get the decoded string from the array (should have one item)
    const char* decodedString = nullptr;
    result = OrtxStringArrayGetItem(stringArray, 0, &decodedString);
    if (result != kOrtxOK)
    {
      OrtxDispose((OrtxObject**)&stringArray);
      throw std::runtime_error(
          std::format(
              "Failed to get decoded string: {}", OrtxGetLastErrorMessage()));
    }

    std::string resultStr(decodedString);
    OrtxDispose((OrtxObject**)&stringArray);
    return resultStr;
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error(
        std::format("Token decoding failed: {}", e.what()));
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
              std::format(
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
              std::format(
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
        std::format("Image tokenization failed: {}", e.what()));
  }
}

std::string
FastVLMInference::createPromptTemplate(std::string_view userPrompt) const
{
  return std::format(
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
    const size_t hiddenSize = FastVLMTokenizerConstants::HIDDEN_SIZE; // 896
    const size_t imagePatchCount = 256; // Vision encoder outputs [1, 256, 896]

    std::vector<float> multimodalEmbeddings;
    multimodalEmbeddings.reserve(
        (tokenIds.size() - 1 + imagePatchCount) * hiddenSize);

    for (size_t i = 0; i < tokenIds.size(); ++i)
    {
      if (tokenIds[i] == FastVLMTokenizerConstants::IMAGE_TOKEN_INDEX)
      {
        // Replace single image token with 256 patch embeddings
        multimodalEmbeddings.insert(
            multimodalEmbeddings.end(),
            imageFeatures.begin(),
            imageFeatures.end());
      }
      else
      {
        // Process individual text token through embedding
        std::vector<int64_t> singleToken = {tokenIds[i]};
        auto tokenEmbedding = runEmbedTokens(singleToken);

        // Add the token embedding (should be hiddenSize values)
        multimodalEmbeddings.insert(
            multimodalEmbeddings.end(),
            tokenEmbedding.begin(),
            tokenEmbedding.end());
      }
    }

    return multimodalEmbeddings;
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error(
        std::format("Failed to create multimodal embeddings: {}", e.what()));
  }
}

std::vector<int64_t> FastVLMInference::generateWithONNXDecoder(
    std::span<float> embeddings,
    int maxTokens,
    float temperature)
{
  try
  {
    const size_t hiddenSize = FastVLMTokenizerConstants::HIDDEN_SIZE; // 896
    const size_t seqLen = embeddings.size() / hiddenSize;

    // Use CPU memory but optimize other aspects for RTX 3090 performance
    auto memoryInfo
        = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto createEmbedsTensor
        = [&memoryInfo](
              std::span<float> embeds, size_t seqLen, size_t hiddenSize)
    {
      std::vector<int64_t> shape = {
          1, static_cast<int64_t>(seqLen), static_cast<int64_t>(hiddenSize)};
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

    auto createKVCacheTensor = [&memoryInfo](
                                   const std::vector<Ort::Float16_t>& cache,
                                   const std::vector<int64_t>& shape)
    {
      return Ort::Value::CreateTensor<Ort::Float16_t>(
          memoryInfo,
          const_cast<Ort::Float16_t*>(cache.data()),
          cache.size(),
          shape.data(),
          shape.size());
    };
    // Create initial tensors using helper functions
    auto embedsTensor = createEmbedsTensor(embeddings, seqLen, hiddenSize);
    auto attentionTensor = createAttentionTensor(seqLen);

    reusablePositionIds.resize(seqLen);
    std::iota(reusablePositionIds.begin(), reusablePositionIds.end(), 0);
    auto positionTensor = createPositionTensor();

    // Prepare inputs vector with the three main inputs
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(embedsTensor));
    inputs.push_back(std::move(attentionTensor));
    inputs.push_back(std::move(positionTensor));

    // Create empty KV cache tensors for all 24 layers (float16 as expected by decoder)
    std::vector<Ort::Float16_t> emptyCache;
    std::vector<int64_t> emptyCacheShape = {1, 2, 0, 64};
    for (int layer = 0; layer < 24; ++layer)
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

    // Temperature-based token sampling function
    auto sampleToken
        = [temperature](
              const Ort::Value& logitsTensor) -> std::pair<int64_t, float>
    {
      auto logitsShape = logitsTensor.GetTensorTypeAndShapeInfo().GetShape();
      auto* logitsData = logitsTensor.GetTensorData<float>();
      size_t vocabSize = logitsShape[2];
      size_t lastPos = (logitsShape[1] - 1) * vocabSize;

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

    auto [bestToken, maxLogit] = sampleToken(outputs[0]);

    // Store the generated tokens
    std::vector<int64_t> generatedTokens = {bestToken};

    // Check for EOS token
    if (bestToken == FastVLMTokenizerConstants::EOS_TOKEN_ID)
    {
      return generatedTokens;
    }

    // Store KV cache from first generation step
    for (int layer = 0; layer < 24; ++layer)
    {
      // Extract key cache
      auto& keyTensor = outputs[1 + layer * 2]; // present.{layer}.key
      auto keyShape = keyTensor.GetTensorTypeAndShapeInfo().GetShape();
      auto* keyData = keyTensor.GetTensorData<Ort::Float16_t>();
      size_t keySize = keyShape[0] * keyShape[1] * keyShape[2] * keyShape[3];
      keyCache[layer].assign(keyData, keyData + keySize);
      cacheShapes[layer]
          = {keyShape[0], keyShape[1], keyShape[2], keyShape[3]};

      // Extract value cache
      auto& valueTensor = outputs[1 + layer * 2 + 1]; // present.{layer}.value
      auto* valueData = valueTensor.GetTensorData<Ort::Float16_t>();
      valueCache[layer].assign(valueData, valueData + keySize);
    }

    // Continue generation for remaining tokens
    for (int step = 1; step < maxTokens; ++step)
    {
      // Get embedding for the new token
      auto tokenEmbedding = runEmbedTokens({&bestToken, 1});

      // Create inputs for next step using helper functions
      std::vector<Ort::Value>& nextInputs = inputs;
      nextInputs.clear();

      nextInputs.push_back(createEmbedsTensor(tokenEmbedding, 1, hiddenSize));

      size_t currentSeqLen = seqLen + step;
      nextInputs.push_back(createAttentionTensor(currentSeqLen));

      reusablePositionIds.clear();
      reusablePositionIds.push_back(static_cast<int64_t>(seqLen + step - 1));
      nextInputs.push_back(createPositionTensor());

      // Add KV cache from previous step
      for (int layer = 0; layer < 24; ++layer)
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
      for (int layer = 0; layer < 24; ++layer)
      {
        auto& keyTensor = nextOutputs[1 + layer * 2];
        auto keyShape = keyTensor.GetTensorTypeAndShapeInfo().GetShape();
        auto* keyData = keyTensor.GetTensorData<Ort::Float16_t>();
        size_t keySize = keyShape[0] * keyShape[1] * keyShape[2] * keyShape[3];
        keyCache[layer].assign(keyData, keyData + keySize);
        cacheShapes[layer].assign(keyShape.data(), keyShape.data() + 4);

        auto& valueTensor = nextOutputs[1 + layer * 2 + 1];
        auto* valueData = valueTensor.GetTensorData<Ort::Float16_t>();
        valueCache[layer].assign(valueData, valueData + keySize);
      }
    }

    return generatedTokens;
  }
  catch (const Ort::Exception& e)
  {
    std::cout << std::format("ONNX decoder error: {}\n", e.what());
    return {};
  }
  catch (const std::exception& e)
  {
    std::cout << std::format("Generation error: {}\n", e.what());
    return {};
  }
}
}
