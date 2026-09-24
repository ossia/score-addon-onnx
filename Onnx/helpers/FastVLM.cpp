#include "FastVLM.hpp"

// fmt is header-only and available in both the score and standalone builds;
// include it directly so this file stays free of any ossia/ include (it used to
// pull ossia/detail/fmt.hpp, a thin wrapper over <fmt/format.h>).
#if defined(check) // Thanks Unreal...
#undef check
#endif
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <Onnx/helpers/HfConfig.hpp>
#include <Onnx/helpers/ImageOps.hpp>
#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>
#include <cmath>

#include <nlohmann/json.hpp>
#include <ortx_utils.h>

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
static constexpr int MAX_LENGTH = 8192;

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

    // The image placeholder, its id and the stop tokens are the model's own;
    // without the files, FastVLM's values stay.
    {
      const std::filesystem::path tokDir{std::string(tokenizerModelPath)};
      if (auto id = HfConfig::imageTokenId(tokDir))
        imageTokenId = *id;
      if (auto t = HfConfig::imageToken(tokDir))
        imageToken = std::move(*t);
      if (auto ids = HfConfig::stopTokens(tokDir); !ids.empty())
        stopTokenIds = std::move(ids);
      spaceMarker
          = HfConfig::decoderReplacesSpaceMarker(tokDir / "tokenizer.json");

      // The family's image preprocessing and placeholder expansion.
      const auto config = HfConfig::read(tokDir / "config.json");
      const auto pre = HfConfig::read(tokDir / "preprocessor_config.json");
      const auto proc = HfConfig::read(tokDir / "processor_config.json");
      const auto tokcfg = HfConfig::read(tokDir / "tokenizer_config.json");
      const auto type = HfConfig::text(config, "model_type").value_or("");
      family = type == "idefics3" ? Family::Idefics3
               : type == "gemma3" ? Family::Gemma3
                                  : Family::Llava;
      auto triple = [&](const char* key, float* out) {
        if (pre.is_object() && pre.contains(key) && pre[key].is_array()
            && pre[key].size() == 3)
          for (int c = 0; c < 3; ++c)
            if (pre[key][c].is_number())
              out[c] = pre[key][c].get<float>();
      };
      auto repeat = [](const std::string& t, int n) {
        std::string r;
        r.reserve(t.size() * n);
        for (int i = 0; i < n; ++i)
          r += t;
        return r;
      };
      placeholder = imageToken;
      placeholderExpansion.clear();
      if (family == Family::Idefics3)
      {
        triple("image_mean", imageMean);
        triple("image_std", imageStd);
        imageSize = 512;
        if (pre.is_object() && pre.contains("max_image_size"))
          if (auto v = HfConfig::number(pre["max_image_size"], "longest_edge"))
            imageSize = (int)*v;
        const int seq
            = (int)HfConfig::number(proc, "image_seq_len").value_or(64.);
        placeholderExpansion = "<fake_token_around_image><global-img>"
                               + repeat(imageToken, seq)
                               + "<fake_token_around_image>";
      }
      else if (family == Family::Gemma3)
      {
        triple("image_mean", imageMean);
        triple("image_std", imageStd);
        imageSize = 896;
        if (pre.is_object() && pre.contains("size"))
          if (auto v = HfConfig::number(pre["size"], "height"))
            imageSize = (int)*v;
        const int seq
            = (int)HfConfig::number(config, "mm_tokens_per_image").value_or(256.);
        const auto boi
            = HfConfig::text(tokcfg, "boi_token").value_or("<start_of_image>");
        const auto eoi
            = HfConfig::text(tokcfg, "eoi_token").value_or("<end_of_image>");
        placeholder = boi;
        placeholderExpansion
            = "\n\n" + boi + repeat(imageToken, seq) + eoi + "\n\n";
      }
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

    // Bind every decoder input by name; the layer count is the number of
    // past_key_values.N.key inputs.
    const std::size_t nIn = decoderSession->GetInputCount();
    auto layerOf = [](std::string_view name, std::string_view prefix,
                      std::string_view suffix) -> int
    {
      if (!name.starts_with(prefix) || !name.ends_with(suffix)
          || name.size() <= prefix.size() + suffix.size())
        return -1;
      const auto digits = name.substr(
          prefix.size(), name.size() - prefix.size() - suffix.size());
      if (!std::all_of(digits.begin(), digits.end(), [](char c) {
            return c >= '0' && c <= '9';
          }))
        return -1;
      return std::stoi(std::string(digits));
    };
    decoderSlots.clear();
    numLayers = 0;
    std::string unknown;
    for (const auto& name : decoderInputNames)
    {
      DecoderSlot slot;
      if (name == "inputs_embeds")
        slot.role = DecoderInput::Embeds;
      else if (name == "attention_mask")
        slot.role = DecoderInput::Mask;
      else if (name == "position_ids")
        slot.role = DecoderInput::Positions;
      else if (name == "num_logits_to_keep" || name == "logits_to_keep")
        slot.role = DecoderInput::LogitsToKeep;
      else if (int l = layerOf(name, "past_key_values.", ".key"); l >= 0)
        slot = {DecoderInput::Key, l};
      else if (int l = layerOf(name, "past_key_values.", ".value"); l >= 0)
        slot = {DecoderInput::Value, l};
      else
      {
        unknown += (unknown.empty() ? "" : ", ") + name;
        continue;
      }
      numLayers = std::max(numLayers, slot.layer + 1);
      decoderSlots.push_back(slot);
    }
    if (!unknown.empty())
      throw std::runtime_error(
          fmt::format("Unsupported decoder inputs: {}", unknown));
    if (numLayers <= 0)
      throw std::runtime_error(
          fmt::format(
              "Unexpected decoder inputs ({}): no past_key_values (not a "
              "merged decoder?)",
              nIn));

    presentKeyOutput.assign(numLayers, -1);
    presentValueOutput.assign(numLayers, -1);
    logitsOutput = -1;
    for (int i = 0; i < (int)decoderOutputNames.size(); ++i)
    {
      const auto& name = decoderOutputNames[i];
      if (name == "logits")
        logitsOutput = i;
      else if (int l = layerOf(name, "present.", ".key"); l >= 0 && l < numLayers)
        presentKeyOutput[l] = i;
      else if (int l = layerOf(name, "present.", ".value"); l >= 0 && l < numLayers)
        presentValueOutput[l] = i;
    }
    if (logitsOutput < 0)
      throw std::runtime_error("The decoder has no logits output");
    for (int l = 0; l < numLayers; ++l)
      if (presentKeyOutput[l] < 0 || presentValueOutput[l] < 0)
        throw std::runtime_error(
            fmt::format("The decoder has no present.{} key/value output", l));

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
    int w = image.width, h = image.height;
    auto pixels = preprocess(image, w, h);
    auto imageFeatures = runVisionEncoder(pixels, w, h);
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

    maxTokens = std::clamp(maxTokens, 1, MAX_LENGTH);

    // Generate tokens using our working ONNX decoder with temperature sampling
    auto generatedTokens
        = generateWithONNXDecoder(multimodalEmbeddings, maxTokens, temperature);

    // Only the generated tokens are decoded, without the stop token.
    return decodeTokens(generatedTokens);
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error(
        fmt::format("Multimodal inference failed: {}", e.what()));
  }
}

std::vector<float>
FastVLMInference::preprocess(const Onnx::ImageData& image, int& w, int& h) const
{
  if (family == Family::Llava)
  {
    // FastVLM: the image's own size, 0..1 (unchanged from the original
    // pipeline; the HF processor would resize to 1024²).
    boost::container::vector<float> values;
    auto t = preprocessImageForFastVLM(image, values);
    w = image.width;
    h = image.height;
    return {t.storage.begin(), t.storage.end()};
  }

  // Idefics3 / gemma3: the whole image stretched to the model's square size,
  // (x / 255 - mean) / std, planar RGB.
  const int side = imageSize > 0 ? imageSize : 512;
  std::vector<uint8_t> rgba((std::size_t)side * side * 4);
  Onnx::resize(
      {.data = image.pixels.data(), .w = image.width, .h = image.height},
      {.data = rgba.data(), .w = side, .h = side});
  std::vector<float> out((std::size_t)3 * side * side);
  const std::size_t plane = (std::size_t)side * side;
  for (std::size_t i = 0; i < plane; ++i)
    for (int c = 0; c < 3; ++c)
      out[c * plane + i]
          = (rgba[i * 4 + c] / 255.f - imageMean[c]) / imageStd[c];
  w = h = side;
  return out;
}

std::vector<float>
FastVLMInference::runVisionEncoder(std::span<float> imageData, int w, int h)
{
  try
  {
    auto memoryInfo
        = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Bound by name: pixel_values ([1,3,H,W], or [1,1,3,H,W] for one image
    // of idefics3) and idefics3's pixel_attention_mask (all pixels valid).
    const std::size_t nIn = visionEncoderSession->GetInputCount();
    std::vector<Ort::AllocatedStringPtr> names;
    std::vector<const char*> namePtrs;
    std::vector<Ort::Value> inputs;
    std::vector<std::vector<int64_t>> shapes(nIn);
    std::unique_ptr<bool[]> mask;
    for (std::size_t i = 0; i < nIn; ++i)
    {
      names.push_back(visionEncoderSession->GetInputNameAllocated(i, allocator));
      namePtrs.push_back(names.back().get());
      const std::string name = names.back().get();
      const auto typeInfo = visionEncoderSession->GetInputTypeInfo(i);
      const auto info = typeInfo.GetTensorTypeAndShapeInfo();
      const std::size_t rank = info.GetShape().size();
      auto& shape = shapes[i];
      if (name == "pixel_attention_mask")
      {
        shape = rank == 4 ? std::vector<int64_t>{1, 1, h, w}
                          : std::vector<int64_t>{1, h, w};
        mask.reset(new bool[(std::size_t)w * h]);
        std::fill_n(mask.get(), (std::size_t)w * h, true);
        inputs.push_back(Ort::Value::CreateTensor<bool>(
            memoryInfo, mask.get(), (std::size_t)w * h, shape.data(),
            shape.size()));
        continue;
      }
      if (i > 0 && name != "pixel_values")
        throw std::runtime_error(
            fmt::format("unsupported vision encoder input '{}'", name));
      shape = rank == 5 ? std::vector<int64_t>{1, 1, 3, h, w}
                        : std::vector<int64_t>{1, 3, h, w};
      if (visionInputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
      {
        reusableF16Scratch.resize(imageData.size());
        for (std::size_t k = 0; k < imageData.size(); ++k)
          reusableF16Scratch[k] = Ort::Float16_t(imageData[k]);
        inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
            memoryInfo, reusableF16Scratch.data(), reusableF16Scratch.size(),
            shape.data(), shape.size()));
      }
      else
      {
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, imageData.data(), imageData.size(), shape.data(),
            shape.size()));
      }
    }

    auto outputName
        = visionEncoderSession->GetOutputNameAllocated(0, allocator);
    const char* outputNames[] = {outputName.get()};

    auto outputs = visionEncoderSession->Run(
        Ort::RunOptions{nullptr}, namePtrs.data(), inputs.data(),
        inputs.size(), outputNames, 1);

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
    if (spaceMarker)
      HfConfig::replaceSpaceMarkers(resultStr);
    return resultStr;
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error(
        fmt::format("Token decoding failed: {}", e.what()));
  }
}

std::vector<int64_t>
FastVLMInference::tokenizeText(const std::string& text) const
{
  const char* inputTexts[] = {text.c_str()};
  OrtxTokenId2DArray* tokenArray = nullptr;
  if (OrtxTokenize(tokenizer, inputTexts, 1, &tokenArray) != kOrtxOK)
    throw std::runtime_error(
        fmt::format("Tokenization failed: {}", OrtxGetLastErrorMessage()));

  const extTokenId_t* tokenData = nullptr;
  size_t tokenCount = 0;
  if (OrtxTokenId2DArrayGetItem(tokenArray, 0, &tokenData, &tokenCount)
      != kOrtxOK)
  {
    OrtxDispose((OrtxObject**)&tokenArray);
    throw std::runtime_error(
        fmt::format("Failed to get tokens: {}", OrtxGetLastErrorMessage()));
  }
  std::vector<int64_t> out(tokenData, tokenData + tokenCount);
  OrtxDispose((OrtxObject**)&tokenArray);
  return out;
}

// The text between image placeholders is tokenized on its own, and each
// placeholder becomes the image token id, which is not in the vocabulary
// (FastVLM) or not tokenized as a single piece by every tokenizer.
std::vector<int64_t>
FastVLMInference::tokenizeImagePrompt(const std::string& prompt) const
{
  try
  {
    std::vector<int64_t> finalTokens;
    size_t pos = 0;
    while (pos < prompt.size())
    {
      const size_t imagePos = prompt.find(imageToken, pos);
      const size_t end
          = imagePos == std::string::npos ? prompt.size() : imagePos;
      if (end > pos)
      {
        auto ids = tokenizeText(prompt.substr(pos, end - pos));
        finalTokens.insert(finalTokens.end(), ids.begin(), ids.end());
      }
      if (imagePos == std::string::npos)
        break;
      finalTokens.push_back(imageTokenId);
      pos = imagePos + imageToken.size();
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
  // LLaVA's template concatenates a string content; the others iterate over
  // typed items and emit their own image placeholder.
  const std::string content = imageToken + "\n" + std::string(userPrompt);
  nlohmann::json message{{"role", "user"}};
  if (family == Family::Llava)
    message["content"] = content;
  else
    message["content"] = nlohmann::json::array(
        {{{"type", "image"}},
         {{"type", "text"}, {"text", std::string(userPrompt)}}});
  const std::string messages = nlohmann::json::array({message}).dump();
  auto expand = [this](std::string text) {
    if (placeholderExpansion.empty())
      return text;
    if (const auto pos = text.find(placeholder); pos != std::string::npos)
      text.replace(pos, placeholder.size(), placeholderExpansion);
    return text;
  };

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
      return expand(std::move(text));
  }

  return fmt::format(
      "<|im_start|>system\n"
      "You are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\n"
      "{}<|im_end|>\n"
      "<|im_start|>assistant\n",
      content);
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
    // One image token takes all the features (FastVLM); several take one
    // row each (SmolVLM, gemma3).
    const auto nImageTokens
        = std::count(tokenIds.begin(), tokenIds.end(), imageTokenId);
    std::size_t rowSize = imageFeatures.size();
    if (nImageTokens > 1)
    {
      if (imageFeatures.size() != (std::size_t)nImageTokens * hiddenSize)
        throw std::runtime_error(fmt::format(
            "{} image tokens in the prompt but {} feature values ({} per "
            "token expected)",
            nImageTokens, imageFeatures.size(), hiddenSize));
      rowSize = (std::size_t)hiddenSize;
    }
    std::size_t nextRow = 0;

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
      if (id == imageTokenId)
      {
        flushSegment();
        multimodalEmbeddings.insert(
            multimodalEmbeddings.end(),
            imageFeatures.begin() + nextRow,
            imageFeatures.begin() + std::min(nextRow + rowSize, imageFeatures.size()));
        nextRow += rowSize;
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

    // One decoder step: `embeds` holds `n` new positions starting at
    // `firstPos`, after `firstPos` cached ones (none on the first step).
    std::vector<std::byte> emptyCache;
    const std::vector<int64_t> emptyCacheShape = {1, kvHeads, 0, headDim};
    int64_t logitsToKeep = 1;
    std::vector<Ort::Value> inputs;
    auto runStep = [&](std::span<float> embeds, size_t n, int64_t firstPos)
    {
      const bool first = firstPos == 0;
      inputs.clear();
      for (const auto& slot : decoderSlots)
      {
        switch (slot.role)
        {
          case DecoderInput::Embeds:
            inputs.push_back(createEmbedsTensor(embeds, n));
            break;
          case DecoderInput::Mask:
            inputs.push_back(createAttentionTensor(firstPos + n));
            break;
          case DecoderInput::Positions:
            reusablePositionIds.resize(n);
            std::iota(
                reusablePositionIds.begin(), reusablePositionIds.end(),
                firstPos);
            inputs.push_back(createPositionTensor());
            break;
          case DecoderInput::LogitsToKeep:
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, &logitsToKeep, 1, nullptr, 0));
            break;
          case DecoderInput::Key:
            inputs.push_back(
                first ? createKVCacheTensor(emptyCache, emptyCacheShape)
                      : createKVCacheTensor(
                          keyCache[slot.layer], cacheShapes[slot.layer]));
            break;
          case DecoderInput::Value:
            inputs.push_back(
                first ? createKVCacheTensor(emptyCache, emptyCacheShape)
                      : createKVCacheTensor(
                          valueCache[slot.layer], cacheShapes[slot.layer]));
            break;
        }
      }
      return decoderSession->Run(
          Ort::RunOptions{nullptr},
          decoderInputNamePtrs.data(),
          inputs.data(),
          inputs.size(),
          decoderOutputNamePtrs.data(),
          decoderOutputNamePtrs.size());
    };

    auto outputs = runStep(embeddings, seqLen, 0);
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
        auto& keyTensor = outs[presentKeyOutput[layer]];
        const auto keyInfo = keyTensor.GetTensorTypeAndShapeInfo();
        const auto keyShape = keyInfo.GetShape();
        const std::size_t bytes = keyInfo.GetElementCount() * kvElemSize;

        const auto* keyData
            = static_cast<const std::byte*>(keyTensor.GetTensorRawData());
        keyCache[layer].assign(keyData, keyData + bytes);
        cacheShapes[layer].assign(keyShape.begin(), keyShape.end());

        auto& valueTensor = outs[presentValueOutput[layer]];
        const std::size_t valueBytes
            = valueTensor.GetTensorTypeAndShapeInfo().GetElementCount()
              * kvElemSize;
        const auto* valueData
            = static_cast<const std::byte*>(valueTensor.GetTensorRawData());
        valueCache[layer].assign(valueData, valueData + valueBytes);
      }
    };
    auto isStop = [this](int64_t token)
    {
      return std::find(stopTokenIds.begin(), stopTokenIds.end(), token)
             != stopTokenIds.end();
    };

    auto [bestToken, maxLogit] = sampleToken(outputs[logitsOutput]);

    std::vector<int64_t> generatedTokens;
    if (isStop(bestToken))
      return generatedTokens;
    generatedTokens.push_back(bestToken);

    // Store KV cache from first generation step
    storeKVCache(outputs);

    // Continue generation for remaining tokens
    for (int step = 1; step < maxTokens; ++step)
    {
      // Get embedding for the new token
      auto tokenEmbedding = runEmbedTokens({&bestToken, 1});

      auto nextOutputs = runStep(
          tokenEmbedding, 1, static_cast<int64_t>(seqLen + step - 1));

      // Get next token from logits using temperature sampling
      auto [nextBestToken, nextMaxLogit]
          = sampleToken(nextOutputs[logitsOutput]);
      bestToken = nextBestToken;
      if (isStop(bestToken))
        break;
      generatedTokens.push_back(bestToken);

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
