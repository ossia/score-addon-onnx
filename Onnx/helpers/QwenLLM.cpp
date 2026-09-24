#include "QwenLLM.hpp"

#include <Onnx/helpers/HfConfig.hpp>
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
  // The prompt is the rendered chat template, which already holds the
  // model's special tokens (gemma's and Llama's <bos>): tokenizing it must
  // not add them again, as HF's add_special_tokens=False after
  // apply_chat_template. Without this gemma got "<bos><bos>" and Llama
  // "<|begin_of_text|>" twice.
  {
    const char* keys[] = {"add_special_tokens"};
    const char* values[] = {"false"};
    OrtxUpdateTokenizerOptions(tokenizer, keys, values, 1);
  }

  // The reply's stop tokens live next to the tokenizer, in
  // generation_config.json (preferred; may list several ids) or config.json.
  // Without either, keep the Qwen ChatML defaults the member initializes to.
  {
    const std::filesystem::path tokDir{std::string(tokenizerModelPath)};
    if (auto ids = HfConfig::stopTokens(tokDir); !ids.empty())
      stopTokenIds = std::move(ids);
    // Reasoning models declare <think> and </think> as added tokens.
    thinkingModel
        = HfConfig::addsTokens(tokDir / "tokenizer.json", "<think>", "</think>");
    spaceMarker = HfConfig::decoderReplacesSpaceMarker(tokDir / "tokenizer.json");
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
  for (const auto& name : inputNames)
    inputNamePtrs.push_back(name.c_str());

  const auto hasOutput = [this](const std::string& name) {
    return std::find(outputNames.begin(), outputNames.end(), name)
           != outputNames.end();
  };
  if (!hasOutput("logits"))
    throw std::runtime_error("QwenLLM: the model has no 'logits' output");

  // Classify every input by name; refuse the ones we cannot feed rather than
  // guessing.
  std::string unknown;
  bool hasIds = false;
  for (size_t i = 0; i < inputNames.size(); ++i)
  {
    const std::string& name = inputNames[i];
    InputSlot slot;
    std::string output;
    if (name == "input_ids")
    {
      slot.role = InputRole::Ids;
      hasIds = true;
    }
    else if (name == "attention_mask")
      slot.role = InputRole::Mask;
    else if (name == "position_ids")
      slot.role = InputRole::Positions;
    else if (name == "num_logits_to_keep")
      slot.role = InputRole::LogitsToKeep;
    else if (name.starts_with("past_key_values."))
      output = "present." + name.substr(std::string_view("past_key_values.").size());
    else if (name.starts_with("past_conv."))
      output = "present_conv." + name.substr(std::string_view("past_conv.").size());
    else
    {
      unknown += (unknown.empty() ? "" : ", ") + name;
      continue;
    }

    if (!output.empty())
    {
      if (!hasOutput(output))
        throw std::runtime_error(
            "QwenLLM: state input '" + name + "' has no '" + output + "' output");
      // Keep the TypeInfo alive: GetTensorTypeAndShapeInfo() is a view.
      const auto typeInfo = modelSession->GetInputTypeInfo(i);
      const auto info = typeInfo.GetTensorTypeAndShapeInfo();
      StateSlot st;
      st.output = std::move(output);
      st.type = info.GetElementType();
      (void)qwenKvElementSize(st.type); // throws on a dtype we cannot carry
      st.initShape = info.GetShape();
      // Batch 1; a dynamic axis past it is the past length, empty at first.
      for (std::size_t d = 0; d < st.initShape.size(); ++d)
        if (st.initShape[d] <= 0)
          st.initShape[d] = (d == 0) ? 1 : 0;
      slot.role = InputRole::State;
      slot.state = (int)states.size();
      states.push_back(std::move(st));
    }
    inputSlots.push_back(slot);
  }
  if (!unknown.empty())
    throw std::runtime_error("QwenLLM: unsupported decoder inputs: " + unknown);
  if (!hasIds || states.empty())
    throw std::runtime_error(
        "QwenLLM: not a decoder with a KV cache (no input_ids or no "
        "past_key_values.* inputs)");

  runOutputNames.push_back("logits");
  for (const auto& st : states)
    runOutputNames.push_back(st.output.c_str());
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
  if (spaceMarker)
    HfConfig::replaceSpaceMarkers(strResult);

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
  if (spaceMarker)
    HfConfig::replaceSpaceMarkers(strResult);

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
QwenLLMInference::applyChatTemplate(const std::string& userPrompt, bool thinking) const
{
  auto text = applyChatTemplate(userPrompt);
  if (thinking || !thinkingModel)
    return text;
  // What Qwen3's template adds for enable_thinking=false, which the Jinja
  // engine has no way to pass: an empty, closed think block. DeepSeek-R1's
  // newer templates already open the block.
  if (text.ends_with("<think>\n"))
    return text + "\n</think>\n\n";
  return text + "<think>\n\n</think>\n\n";
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
    bool thinking,
    std::function<bool(int64_t)> onToken)
{
  auto inputIds = tokenize(applyChatTemplate(prompt, thinking));
  if (inputIds.empty())
    return;
  const size_t promptLen = inputIds.size();

  Ort::MemoryInfo memoryInfo
      = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  auto createIdsTensor = [&memoryInfo](std::span<int64_t> ids)
  {
    std::vector<int64_t> shape = {1, static_cast<int64_t>(ids.size())};
    return Ort::Value::CreateTensor<int64_t>(
        memoryInfo, ids.data(), ids.size(), shape.data(), shape.size());
  };

  // State tensors are created in their own dtype straight over the raw bytes.
  auto createStateTensor = [&memoryInfo](StateSlot& st)
  {
    return Ort::Value::CreateTensor(
        memoryInfo, st.data.data(), st.data.size(), st.shape.data(),
        st.shape.size(), st.type);
  };

  auto storeStates = [this](std::vector<Ort::Value>& outs)
  {
    for (std::size_t k = 0; k < states.size(); ++k)
    {
      auto& t = outs[1 + k]; // runOutputNames[1 + k] == states[k].output
      const auto info = t.GetTensorTypeAndShapeInfo();
      const std::size_t bytes
          = info.GetElementCount() * qwenKvElementSize(states[k].type);
      const auto* d = static_cast<const std::byte*>(t.GetTensorRawData());
      states[k].data.assign(d, d + bytes);
      states[k].shape = info.GetShape();
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

  // Reset the states from any previous generation: an empty KV cache, a
  // zeroed convolution state.
  for (auto& st : states)
  {
    st.shape = st.initShape;
    std::size_t n = 1;
    for (auto d : st.shape)
      n *= (std::size_t)d;
    st.data.assign(n * qwenKvElementSize(st.type), std::byte{0});
  }

  int64_t nextToken = -1;
  std::vector<int64_t> stepIds;

  for (int step = 0; step < maxTokens; ++step)
  {
    const size_t totalLen = promptLen + step;
    if (step == 0)
    {
      // Prefill: the whole prompt in one pass.
      stepIds = inputIds;
      reusablePositionIds.resize(promptLen);
      std::iota(reusablePositionIds.begin(), reusablePositionIds.end(), 0);
    }
    else
    {
      // One new token against the cached past.
      stepIds.assign(1, nextToken);
      reusablePositionIds.assign(1, static_cast<int64_t>(totalLen - 1));
    }
    reusableAttentionMask.assign(totalLen, 1);

    // In the session's input order.
    std::vector<Ort::Value> inputs;
    inputs.reserve(inputSlots.size());
    for (const auto& slot : inputSlots)
    {
      switch (slot.role)
      {
        case InputRole::Ids:
          inputs.push_back(createIdsTensor(stepIds));
          break;
        case InputRole::Mask:
          inputs.push_back(createIdsTensor(reusableAttentionMask));
          break;
        case InputRole::Positions:
          inputs.push_back(createIdsTensor(reusablePositionIds));
          break;
        case InputRole::LogitsToKeep:
          inputs.push_back(Ort::Value::CreateTensor<int64_t>(
              memoryInfo, &logitsToKeep, 1, nullptr, 0));
          break;
        case InputRole::State:
          inputs.push_back(createStateTensor(states[slot.state]));
          break;
      }
    }

    auto outputs = modelSession->Run(
        Ort::RunOptions{nullptr},
        inputNamePtrs.data(),
        inputs.data(),
        inputs.size(),
        runOutputNames.data(),
        runOutputNames.size());

    if (outputs.empty())
      throw std::runtime_error("No outputs from model");

    readLastLogits(outputs[0]);
    storeStates(outputs);

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
    int topK,
    bool thinking)
{
  std::vector<int64_t> generated;
  generateLoop(
      prompt, maxTokens, temperature, topP, topK, thinking,
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
    int topK,
    bool thinking)
{
  // Decoding tokens one at a time splits multi-byte UTF-8 sequences (byte
  // level BPE): re-decode the whole reply each step and emit the increment.
  std::vector<int64_t> ids;
  std::string lastText;
  generateLoop(
      prompt, maxTokens, temperature, topP, topK, thinking,
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
