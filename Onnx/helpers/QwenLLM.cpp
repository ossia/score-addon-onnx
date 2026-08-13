#include "QwenLLM.hpp"

#include <Onnx/helpers/OnnxContext.hpp>
#include <cmath>
#include <cstdio>
#include <ext_status.h>
#include <ortx_utils.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace Onnx
{

// Reads eos_token_id (a number or a list of numbers) from a HF-style JSON
// config next to the tokenizer; returns an empty vector when absent.
static std::vector<int64_t> readStopTokens(const std::filesystem::path& file)
{
  std::ifstream in(file);
  if (!in)
    return {};
  std::stringstream buf;
  buf << in.rdbuf();

  const auto json
      = nlohmann::json::parse(buf.str(), nullptr, /*allow_exceptions=*/false);
  if (json.is_discarded() || !json.is_object())
    return {};

  const auto it = json.find("eos_token_id");
  if (it == json.end())
    return {};

  std::vector<int64_t> ids;
  if (it->is_number_integer())
    ids.push_back(it->get<int64_t>());
  else if (it->is_array())
    for (const auto& v : *it)
      if (v.is_number_integer())
        ids.push_back(v.get<int64_t>());
  return ids;
}

static std::size_t qwenKvElementSize(ONNXTensorElementDataType t)
{
  switch (t)
  {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return 4;
    default:
      throw std::runtime_error(
          "QwenLLM: unsupported KV cache element type "
          + std::to_string((int)t));
  }
}

QwenLLMInference::QwenLLMInference(
    std::string_view modelPath,
    std::string_view tokenizerModelPath)
    : env(Onnx::make_env("QwenLLM"))
{
  Onnx::Options oopts;
  sessionOptions = Onnx::create_session_options(oopts);

  // Load the model. ORT's path-based Session ctor takes const ORTCHAR_T*, which
  // is wchar_t on Windows -- pass a const char* there and it fails to compile.
  // Decode the UTF-8 byte path to a wstring on Windows (same as FastVLM.cpp).
#if defined(_WIN32)
  auto modelPath_str
      = std::filesystem::path(
            std::u8string(
                reinterpret_cast<const char8_t*>(modelPath.data()),
                modelPath.size()))
            .wstring();
#else
  auto modelPath_str = modelPath;
#endif
  modelSession = create_session_with_fallback(env, modelPath_str, sessionOptions);

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

  // The reply's stop tokens live next to the tokenizer, in
  // generation_config.json (preferred; may list several ids) or config.json.
  // Without either, keep the Qwen ChatML defaults the member initializes to.
  {
    const std::filesystem::path tokDir{std::string(tokenizerModelPath)};
    auto ids = readStopTokens(tokDir / "generation_config.json");
    if (ids.empty())
      ids = readStopTokens(tokDir / "config.json");
    if (!ids.empty())
      stopTokenIds = std::move(ids);
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

  // Discover the graph geometry of this export: layer count from the number
  // of past_key_values inputs, KV dtype / heads / head-dim from the first
  // one. The dtype really does vary per model *and* per variant (the Qwen3
  // fp16 export uses an fp16 cache, the Qwen2.5 fp16 export an fp32 one), so
  // it cannot be assumed.
  int pastInputs = 0;
  for (size_t i = 0; i < inputNames.size(); ++i)
  {
    if (inputNames[i].starts_with("past_key_values."))
    {
      ++pastInputs;
      if (inputNames[i] == "past_key_values.0.key")
      {
        // Keep the TypeInfo alive: GetTensorTypeAndShapeInfo() is a view.
        const auto typeInfo = modelSession->GetInputTypeInfo(i);
        const auto info = typeInfo.GetTensorTypeAndShapeInfo();
        kvType = info.GetElementType();
        // Shape is [batch, kv_heads, past_seq_len, head_dim]
        if (const auto sh = info.GetShape(); sh.size() == 4)
        {
          if (sh[1] > 0)
            kvHeads = sh[1];
          if (sh[3] > 0)
            headDim = sh[3];
        }
      }
    }
    else if (inputNames[i] == "position_ids")
    {
      hasPositionIds = true;
    }
  }
  numLayers = pastInputs / 2;
  if (numLayers <= 0 || kvHeads <= 0 || headDim <= 0)
    throw std::runtime_error(
        "QwenLLM: could not derive the KV cache geometry from the model "
        "(not a transformers.js-style decoder export?)");

  keyCache.assign(numLayers, {});
  valueCache.assign(numLayers, {});
  cacheShapes.assign(numLayers, {});
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
  // For very low temperature, use greedy sampling
  if (temperature < 0.01f)
  {
    auto maxIt = std::max_element(logits.begin(), logits.end());
    int64_t greedyToken = std::distance(logits.begin(), maxIt);
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

std::string
QwenLLMInference::applyChatTemplate(const std::string& userPrompt) const
{
  // Only a user message: templates insert their model's own default system
  // prompt when they want one, and some (e.g. Gemma) reject a system role.
  const std::string messages
      = nlohmann::json::array({{{"role", "user"}, {"content", userPrompt}}})
            .dump();

  OrtxTensorResult* result{};
  if (OrtxApplyChatTemplate(
          tokenizer, nullptr, messages.c_str(), nullptr, &result,
          /*add_generation_prompt=*/true, /*tokenize=*/false)
      == kOrtxOK)
  {
    std::string text;
    OrtxTensor* tensor{};
    if (OrtxTensorResultGetAt(result, 0, &tensor) == kOrtxOK)
    {
      const char* data{};
      if (OrtxGetTensorData(
              tensor, reinterpret_cast<const void**>(&data), nullptr, nullptr)
              == kOrtxOK
          && data)
        text = data;
    }
    OrtxDispose((OrtxObject**)&result);
    if (!text.empty())
      return text;
  }

  // No usable template shipped with the model: Qwen-style ChatML.
  return "<|im_start|>system\nYou are a chatbot<|im_end|>\n"
         "<|im_start|>user\n" + userPrompt + "<|im_end|>\n"
         "<|im_start|>assistant\n";
}

void QwenLLMInference::generateLoop(
    const std::string& prompt,
    int maxTokens,
    float temperature,
    float topP,
    int topK,
    std::function<bool(int64_t)> onToken)
{
  auto inputIds = tokenize(applyChatTemplate(prompt));
  if (inputIds.empty())
    return;
  const size_t promptLen = inputIds.size();

  Ort::MemoryInfo memoryInfo
      = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const std::size_t kvElemSize = qwenKvElementSize(kvType);

  auto createIdsTensor = [&memoryInfo](std::span<int64_t> ids)
  {
    std::vector<int64_t> shape = {1, static_cast<int64_t>(ids.size())};
    return Ort::Value::CreateTensor<int64_t>(
        memoryInfo, ids.data(), ids.size(), shape.data(), shape.size());
  };

  // KV cache tensors are created in the model's own dtype straight over the
  // raw cache bytes.
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

  // Extract the last token's logits as fp32.
  std::vector<float> logits;
  auto readLastLogits = [&logits](const Ort::Value& logitsTensor)
  {
    const auto info = logitsTensor.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    const size_t vocab = shape.back();
    const size_t lastPos = info.GetElementCount() - vocab;
    logits.resize(vocab);

    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
      const auto* d = logitsTensor.GetTensorData<Ort::Float16_t>();
      for (size_t i = 0; i < vocab; ++i)
        logits[i] = d[lastPos + i].ToFloat();
    }
    else
    {
      const auto* d = logitsTensor.GetTensorData<float>();
      std::copy_n(d + lastPos, vocab, logits.begin());
    }
  };

  // Reset the cache from any previous generation.
  for (auto& c : keyCache)
    c.clear();
  for (auto& c : valueCache)
    c.clear();
  std::vector<int64_t> emptyCacheShape = {1, kvHeads, 0, headDim};
  for (auto& s : cacheShapes)
    s = emptyCacheShape;

  int64_t nextToken = -1;
  std::vector<int64_t> stepIds;

  for (int step = 0; step < maxTokens; ++step)
  {
    std::vector<Ort::Value> inputs;
    inputs.reserve(3 + 2 * numLayers);

    const size_t totalLen = promptLen + step;
    if (step == 0)
    {
      // Prefill: the whole prompt in one pass.
      inputs.push_back(createIdsTensor(inputIds));
      reusableAttentionMask.assign(promptLen, 1);
      inputs.push_back(createIdsTensor(reusableAttentionMask));
      if (hasPositionIds)
      {
        reusablePositionIds.resize(promptLen);
        std::iota(reusablePositionIds.begin(), reusablePositionIds.end(), 0);
        inputs.push_back(createIdsTensor(reusablePositionIds));
      }
    }
    else
    {
      // One new token against the cached past.
      stepIds.assign(1, nextToken);
      inputs.push_back(createIdsTensor(stepIds));
      reusableAttentionMask.assign(totalLen, 1);
      inputs.push_back(createIdsTensor(reusableAttentionMask));
      if (hasPositionIds)
      {
        reusablePositionIds.assign(1, static_cast<int64_t>(totalLen - 1));
        inputs.push_back(createIdsTensor(reusablePositionIds));
      }
    }

    for (int layer = 0; layer < numLayers; ++layer)
    {
      inputs.push_back(createKVCacheTensor(keyCache[layer], cacheShapes[layer]));
      inputs.push_back(
          createKVCacheTensor(valueCache[layer], cacheShapes[layer]));
    }

    auto outputs = modelSession->Run(
        Ort::RunOptions{nullptr},
        inputNamePtrs.data(),
        inputs.data(),
        std::min(inputs.size(), inputNamePtrs.size()),
        outputNamePtrs.data(),
        outputNamePtrs.size());

    if (outputs.empty())
      throw std::runtime_error("No outputs from model");

    readLastLogits(outputs[0]);
    storeKVCache(outputs);

    nextToken = sampleToken(logits, temperature, topP, topK);

    if (std::find(stopTokenIds.begin(), stopTokenIds.end(), nextToken)
        != stopTokenIds.end())
      break;

    if (!onToken(nextToken))
      break;
  }
}

std::string QwenLLMInference::generate(
    const std::string& prompt,
    int maxTokens,
    float temperature,
    float topP,
    int topK)
{
  std::vector<int64_t> generated;
  generateLoop(
      prompt, maxTokens, temperature, topP, topK,
      [&generated](int64_t token)
      {
        generated.push_back(token);
        return true;
      });

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
  // Decoding tokens one at a time splits multi-byte UTF-8 sequences (byte
  // level BPE): re-decode the whole reply each step and emit the increment.
  std::vector<int64_t> ids;
  std::string lastText;
  generateLoop(
      prompt, maxTokens, temperature, topP, topK,
      [&, this](int64_t token)
      {
        ids.push_back(token);
        std::string full = decodeTokens(ids);
        std::string delta
            = full.starts_with(lastText) ? full.substr(lastText.size()) : full;
        lastText = std::move(full);
        if (delta.empty())
          return true; // wait for the rest of a multi-byte sequence
        return tokenCallback(delta);
      });
}

}
