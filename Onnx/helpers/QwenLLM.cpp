#include "QwenLLM.hpp"

#include <QDebug>

#include <Onnx/helpers/OnnxContext.hpp>
#include <cmath>
#include <ext_status.h>

#include <algorithm>
#include <random>
#include <stdexcept>

namespace Onnx
{

QwenLLMInference::QwenLLMInference(
    std::string_view modelPath,
    std::string_view tokenizerModelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "QwenLLM")
{
  Onnx::Options oopts;
  sessionOptions = Onnx::create_session_options(oopts);

  // Configure Qwen-specific attention attributes (following onnxruntime-genai DecoderOnly_Model pattern)
  // Set q_norm and k_norm to enable query-key normalization for better attention stability
  sessionOptions.AddConfigEntry("attention.q_norm", "1");
  sessionOptions.AddConfigEntry("attention.k_norm", "1");

  // Additional Qwen model optimizations following onnxruntime-genai DecoderOnly_Model pattern
  sessionOptions.AddConfigEntry("model.type", "decoder_only");
  sessionOptions.AddConfigEntry("model.architecture", "qwen");

  // Load the model
  modelSession = std::make_unique<Ort::Session>(
      env, modelPath.data(), sessionOptions);

  if (tokenizerModelPath.ends_with("tokenizer.json"))
    tokenizerModelPath = tokenizerModelPath.substr(
        0, tokenizerModelPath.size() - strlen("tokenizer.json"));

  extError_t result = OrtxCreateTokenizer(
      &tokenizer, std::string(tokenizerModelPath).c_str());
  if (result != kOrtxOK)
  {
    const char* msg = OrtxGetLastErrorMessage();
    throw std::runtime_error(std::string("Failed to create tokenizer: ") + msg);
  }

  // Get input/output names and inspect shapes
  size_t numInputs = modelSession->GetInputCount();
  size_t numOutputs = modelSession->GetOutputCount();

  inputNames.reserve(numInputs);
  outputNames.reserve(numOutputs);

  for (size_t i = 0; i < numInputs; ++i)
  {
    auto namePtr = modelSession->GetInputNameAllocated(i, allocator);
    inputNames.push_back(namePtr.get());
  }

  for (size_t i = 0; i < numOutputs; ++i)
  {
    auto namePtr = modelSession->GetOutputNameAllocated(i, allocator);
    outputNames.push_back(namePtr.get());
  }

  inputNamePtrs.reserve(inputNames.size());
  outputNamePtrs.reserve(outputNames.size());
  
  for (const auto& name : inputNames)
    inputNamePtrs.push_back(name.c_str());
  for (const auto& name : outputNames)
    outputNamePtrs.push_back(name.c_str());

  pastKeyValues.resize(numLayers * 2);
}

QwenLLMInference::~QwenLLMInference()
{
  if (tokenizer)
  {
    OrtxDispose((OrtxObject**)&tokenizer);
  }
}

std::vector<int64_t> QwenLLMInference::tokenize(const std::string& text) const
{
  const char* inputs[] = {text.c_str()};
  OrtxTokenId2DArray* tokenIds = nullptr;
  
  extError_t result = OrtxTokenize(tokenizer, inputs, 1, &tokenIds);
  if (result != kOrtxOK)
  {
    const char* msg = OrtxGetLastErrorMessage();
    throw std::runtime_error(std::string("Tokenization failed: ") + msg);
  }

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(tokenIds, 0, &ids, &length);

  std::vector<int64_t> tokenResult(ids, ids + length);
  OrtxDispose((OrtxObject**)&tokenIds);

  return tokenResult;
}

std::string QwenLLMInference::decodeToken(int64_t tokenId) const
{
  OrtxStringArray* texts = nullptr;
  const extTokenId_t id = static_cast<extTokenId_t>(tokenId);
  
  extError_t result = OrtxDetokenize1D(tokenizer, &id, 1, &texts);
  if (result != kOrtxOK)
    return "";

  const char* text = nullptr;
  OrtxStringArrayGetItem(texts, 0, &text);
  std::string strResult(text);
  OrtxDispose((OrtxObject**)&texts);

  return strResult;
}

std::string QwenLLMInference::decodeTokens(std::span<int64_t> tokens) const
{
  if (tokens.empty())
    return "";

  std::vector<extTokenId_t> ids(tokens.begin(), tokens.end());
  OrtxStringArray* texts = nullptr;
  
  extError_t result = OrtxDetokenize1D(tokenizer, ids.data(), ids.size(), &texts);
  if (result != kOrtxOK)
    return "";

  const char* text = nullptr;
  OrtxStringArrayGetItem(texts, 0, &text);
  std::string strResult(text);
  OrtxDispose((OrtxObject**)&texts);

  return strResult;
}

void QwenLLMInference::applyTemperature(std::span<float> logits, float temperature)
{
  if (temperature <= 0.0f)
    return;

  for (float& logit : logits)
  {
    logit /= temperature;
  }
}

void QwenLLMInference::softmax(std::span<float> logits)
{
  float maxLogit = *std::max_element(logits.begin(), logits.end());
  
  float sum = 0.0f;
  for (float& logit : logits)
  {
    logit = std::exp(logit - maxLogit);
    sum += logit;
  }

  for (float& logit : logits)
  {
    logit /= sum;
  }
}

void QwenLLMInference::applyTopK(std::span<float> logits, int k)
{
  if (k <= 0 || k >= logits.size())
    return;

  std::vector<std::pair<float, int>> indexed;
  indexed.reserve(logits.size());
  
  for (int i = 0; i < logits.size(); ++i)
  {
    indexed.emplace_back(logits[i], i);
  }

  std::partial_sort(
      indexed.begin(), indexed.begin() + k, indexed.end(),
      [](const auto& a, const auto& b) { return a.first > b.first; });

  for (int i = k; i < indexed.size(); ++i)
  {
    logits[indexed[i].second] = -INFINITY;
  }
}

void QwenLLMInference::applyTopP(std::span<float> logits, float p)
{
  if (p <= 0.0f || p >= 1.0f)
    return;

  // Convert logits to probabilities for top-p calculation
  std::vector<float> probs(logits.size());
  float maxLogit = *std::max_element(logits.begin(), logits.end());
  
  float sum = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i)
  {
    probs[i] = std::exp(logits[i] - maxLogit);
    sum += probs[i];
  }
  
  for (size_t i = 0; i < logits.size(); ++i)
  {
    probs[i] /= sum;
  }

  // Sort by probability for top-p filtering
  std::vector<std::pair<float, int>> indexed;
  indexed.reserve(logits.size());
  
  for (int i = 0; i < logits.size(); ++i)
  {
    indexed.emplace_back(probs[i], i);
  }

  std::sort(indexed.begin(), indexed.end(),
      [](const auto& a, const auto& b) { return a.first > b.first; });

  float cumSum = 0.0f;
  int cutoff = 0;
  
  for (int i = 0; i < indexed.size(); ++i)
  {
    cumSum += indexed[i].first;
    if (cumSum > p)
    {
      cutoff = i + 1;
      break;
    }
  }

  // Set logits to -infinity for tokens outside top-p
  for (int i = cutoff; i < indexed.size(); ++i)
  {
    logits[indexed[i].second] = -INFINITY;
  }
}

int64_t QwenLLMInference::sampleToken(
    std::span<float> logits,
    float temperature,
    float topP,
    int topK)
{
  // For very low temperature or garbage output debugging, use greedy sampling
  if (temperature < 0.01f)
  {
    auto maxIt = std::max_element(logits.begin(), logits.end());
    int64_t greedyToken = std::distance(logits.begin(), maxIt);
    // qDebug() << "Greedy sampling: token" << greedyToken << "with logit" << *maxIt;
    return greedyToken;
  }
  
  // Apply sampling transformations in correct order
  applyTemperature(logits, temperature);
  applyTopK(logits, topK);
  applyTopP(logits, topP);
  softmax(logits);

  static thread_local std::random_device rd;
  static thread_local std::mt19937 gen(rd());
  std::discrete_distribution<> dist(logits.begin(), logits.end());

  int64_t sampled = dist(gen);
  return sampled;
}

std::vector<float> QwenLLMInference::runModel(
    std::span<int64_t> inputIds,
    std::span<int64_t> attentionMask,
    std::span<std::vector<Ort::Float16_t>> pastKeyValues,
    std::vector<std::vector<Ort::Float16_t>>& newKeyValues)
{
  std::vector<Ort::Value> inputTensors;
  
  // Create memory info for CPU tensors
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);
  
  // Create input tensors
  std::vector<int64_t> inputShape = {1, static_cast<int64_t>(inputIds.size())};
  inputTensors.push_back(
      Ort::Value::CreateTensor<int64_t>(
          memoryInfo,
          const_cast<int64_t*>(inputIds.data()),
          inputIds.size(),
          inputShape.data(),
          inputShape.size()));

  inputTensors.push_back(
      Ort::Value::CreateTensor<int64_t>(
          memoryInfo,
          const_cast<int64_t*>(attentionMask.data()),
          attentionMask.size(),
          inputShape.data(),
          inputShape.size()));

  // Add past key values - must provide all expected KV tensors
  // For first inference, provide empty tensors with shape [1, num_heads, 0, head_dim]
  bool isFirstInference = true;
  for (const auto& kv : pastKeyValues)
  {
    if (!kv.empty())
    {
      isFirstInference = false;
      break;
    }
  }
  
  if (isFirstInference)
  {
    // For initial inference, create empty KV tensors
    std::vector<int64_t> emptyKvShape = {1, numKeyValueHeads, 0, headDim};
    for (int i = 0; i < numLayers * 2; ++i)
    {
      inputTensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
          memoryInfo, nullptr, 0,
          emptyKvShape.data(), emptyKvShape.size()));
    }
  }
  else
  {
    // Use existing KV cache
    for (size_t i = 0; i < pastKeyValues.size(); ++i)
    {
      const auto& kv = pastKeyValues[i];
      if (!kv.empty())
      {
        // Shape: [batch_size, num_kv_heads, seq_len, head_dim]
        int64_t seqLen = static_cast<int64_t>(kv.size()) / (numKeyValueHeads * headDim);
        std::vector<int64_t> kvShape = {1, numKeyValueHeads, seqLen, headDim};

        inputTensors.push_back(
            Ort::Value::CreateTensor<Ort::Float16_t>(
                memoryInfo,
                const_cast<Ort::Float16_t*>(kv.data()),
                kv.size(),
                kvShape.data(),
                kvShape.size()));
      } else {
        qDebug() << "QwenLLM: WARNING - KV cache" << i << "is empty";
      }
    }
  }

  // Run inference

  try
  {
    auto outputs = modelSession->Run(
        Ort::RunOptions{nullptr},
        inputNamePtrs.data(),
        inputTensors.data(),
        inputTensors.size(),
        outputNamePtrs.data(),
        outputNamePtrs.size());

    if (outputs.empty()) {
      throw std::runtime_error("No outputs from model");
    }

    auto& logitsTensor = outputs[0];
    if (!logitsTensor.IsTensor()) {
      throw std::runtime_error("First output is not a tensor");
    }
    
    auto typeInfo = logitsTensor.GetTensorTypeAndShapeInfo();
    auto dataType = typeInfo.GetElementType();

    if (dataType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        && dataType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
      throw std::runtime_error("Logits tensor is neither float nor float16 type");
    }

    auto logitsShape = typeInfo.GetShape();
    size_t logitsSize = typeInfo.GetElementCount();

    // Check if the logits tensor has the expected vocab size in the last dimension
    if (logitsShape.empty()) {
      throw std::runtime_error("Logits tensor has no dimensions");
    }
    
    int64_t lastDim = logitsShape.back();
    if (lastDim != vocabSize) {
      qDebug() << "WARNING: Last dimension" << lastDim << "doesn't match vocab size" << vocabSize;
    }
    
    // For safety, only take the logits for the last token (most recent position)
    // Expected shape: [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
    size_t logitsPerToken = static_cast<size_t>(lastDim);
    size_t numTokens = logitsSize / logitsPerToken;
    if (numTokens == 0 || logitsPerToken == 0)
    {
      throw std::runtime_error("Invalid logits dimensions");
    }

    // Get logits for the last token only
    size_t lastTokenOffset = (numTokens - 1) * logitsPerToken;
    std::vector<float> logits(logitsPerToken);
    
    // Handle different data types
    if (dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      float* logitsData = logitsTensor.GetTensorMutableData<float>();
      if (!logitsData || logitsSize == 0) {
        throw std::runtime_error("Invalid float logits data");
      }

      // Use manual copy with bounds checking
      for (size_t i = 0; i < logitsPerToken; ++i) {
        if (lastTokenOffset + i >= logitsSize) {
          throw std::runtime_error("Logits index out of bounds");
        }
        logits[i] = logitsData[lastTokenOffset + i];
      }
    } else if (dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      Ort::Float16_t* logitsData = logitsTensor.GetTensorMutableData<Ort::Float16_t>();
      if (!logitsData || logitsSize == 0) {
        throw std::runtime_error("Invalid float16 logits data");
      }

      for (size_t i = 0; i < logitsPerToken; ++i) {
        if (lastTokenOffset + i >= logitsSize) {
          throw std::runtime_error("Logits index out of bounds");
        }

        logits[i] = logitsData[lastTokenOffset + i].ToFloat();
      }
    }

    // Update key values cache - model outputs are already properly concatenated
    newKeyValues.clear();

    for (size_t i = 1; i < outputs.size(); ++i)
    {
      auto& output = outputs[i];
      auto typeInfo = output.GetTensorTypeAndShapeInfo();
      auto shape = typeInfo.GetShape();
      auto dataType = typeInfo.GetElementType();
      size_t kvSize = typeInfo.GetElementCount();

      if (dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
      {
        Ort::Float16_t* kvData = output.GetTensorMutableData<Ort::Float16_t>();
        newKeyValues.emplace_back(kvData, kvData + kvSize);
      }
      else
      {
        qDebug() << "QwenLLM: WARNING - KV output" << i << "is not float16, skipping";
      }
    }

    return logits;
  }
  catch (const Ort::Exception& e) {
    qDebug() << "ORT Exception in runModel:" << e.what();
    throw;
  }
}

std::string QwenLLMInference::generate(
    const std::string& prompt,
    int maxTokens,
    float temperature,
    float topP,
    int topK)
{
  // Format prompt using Qwen chat template (from jinja template)
  std::string formattedPrompt = "<|im_start|>system\nYou are a chatbot<|im_end|>\n"
                               "<|im_start|>user\n" + prompt + "<|im_end|>\n"
                               "<|im_start|>assistant\n";

  auto inputIds = tokenize(formattedPrompt);

  std::vector<int64_t> generatedIds = inputIds;
  std::vector<int64_t> currentInputIds = inputIds;
  std::vector<int64_t> attentionMask(inputIds.size(), 1);
  
  pastKeyValues.clear();
  pastKeyValues.resize(numLayers * 2);

  for (int i = 0; i < maxTokens; ++i)
  {
    // Fallback to non-cached generation: always use the full sequence
    // This is slower but works correctly with models that don't support KV caching
    std::vector<int64_t> inferenceInputIds = currentInputIds;
    std::vector<int64_t> inferenceAttentionMask = attentionMask;
    // Run without KV cache - pass empty cache to force full recomputation
    std::vector<std::vector<Ort::Float16_t>> emptyKeyValues(numLayers * 2);
    std::vector<std::vector<Ort::Float16_t>> newKeyValues;
    auto logits = runModel(
        inferenceInputIds,
        inferenceAttentionMask,
        emptyKeyValues,
        newKeyValues);
    // Don't store KV cache in non-cached mode

    // The logits are already from the last token only
    std::span<float> lastLogits(logits.data(), logits.size());

    int64_t nextToken = sampleToken(lastLogits, temperature, topP, topK);

    // Check for EOS tokens: <|endoftext|> (151643) or <|im_end|> (151645)
    if (nextToken == eosTokenId || nextToken == 151645)
      break;

    generatedIds.push_back(nextToken);
    currentInputIds.push_back(nextToken);
    attentionMask.push_back(1);
  }

  size_t originalPromptSize = inputIds.size();
  if (generatedIds.size() <= originalPromptSize) {
    return "";
  }

  std::span<int64_t> generated(
      generatedIds.data() + originalPromptSize,
      generatedIds.size() - originalPromptSize);

  return decodeTokens(generated);
}

void QwenLLMInference::generateStreaming(
    const std::string& prompt,
    std::function<bool(const std::string&)> tokenCallback,
    int maxTokens,
    float temperature,
    float topP,
    int topK)
{
  // Format prompt using Qwen chat template (from jinja template)
  std::string formattedPrompt = "<|im_start|>system\nYou are a poet<|im_end|>\n"
                               "<|im_start|>user\n" + prompt + "<|im_end|>\n"
                               "<|im_start|>assistant\n";

  auto inputIds = tokenize(formattedPrompt);

  std::vector<int64_t> currentInputIds = inputIds;
  std::vector<int64_t> attentionMask(inputIds.size(), 1);
  
  pastKeyValues.clear();
  pastKeyValues.resize(numLayers * 2);

  bool shouldContinue = true;
  int step = 0;
  
  while (shouldContinue && step < maxTokens)
  {
    // Fallback to non-cached generation: always use the full sequence
    std::vector<int64_t> inferenceInputIds = currentInputIds;
    std::vector<int64_t> inferenceAttentionMask = attentionMask;
    // Run without KV cache - pass empty cache to force full recomputation
    std::vector<std::vector<Ort::Float16_t>> emptyKeyValues(numLayers * 2);
    std::vector<std::vector<Ort::Float16_t>> newKeyValues;
    auto logits = runModel(inferenceInputIds, inferenceAttentionMask, emptyKeyValues, newKeyValues);
    // Don't store KV cache in non-cached mode

    // The logits are already from the last token only
    std::span<float> lastLogits(logits.data(), logits.size());

    // Debug: show top 5 tokens before sampling
    std::vector<std::pair<float, int64_t>> topTokens;
    topTokens.reserve(lastLogits.size());
    for (size_t i = 0; i < lastLogits.size(); ++i) {
      topTokens.emplace_back(lastLogits[i], static_cast<int64_t>(i));
    }

    int64_t nextToken = sampleToken(lastLogits, temperature, topP, topK);

    // Check for EOS tokens: <|endoftext|> (151643) or <|im_end|> (151645)
    if (nextToken == eosTokenId || nextToken == 151645)
      break;

    std::string tokenText = decodeToken(nextToken);
    shouldContinue = tokenCallback(tokenText);

    if (!shouldContinue)
      break;

    currentInputIds.push_back(nextToken);
    attentionMask.push_back(1);
    step++;
  }
}

}
