// Vision Language Model families (BUG-LEDGER F3): the helper only ran FastVLM
// (LLaVA-Qwen2). SmolVLM (idefics3) failed in its vision encoder (5-D pixel
// values plus a pixel mask) and gemma3 in its embeddings (896² input), and
// neither expanded its image placeholder into the tokens the features fill.
#include <Onnx/helpers/FastVLM.hpp>
#include <Onnx/helpers/HfConfig.hpp>
#include <OnnxModels/Utils.hpp>

#include <QImage>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace
{
std::string models()
{
  const char* env = std::getenv("ONNX_TEST_VLM_MODELS");
  return env ? env : "/mnt/win2/models";
}

Onnx::ImageData photo()
{
  const QImage img
      = QImage("/mnt/win2/models/libreonnx/ossia-detection-model-pack/test_images/body.jpg")
            .convertToFormat(QImage::Format_RGBA8888);
  Onnx::ImageData d;
  d.width = img.width();
  d.height = img.height();
  d.pixels.assign(img.constBits(), img.constBits() + (std::size_t)d.width * d.height * 4);
  return d;
}

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Returns the answer, or an empty string when the model is not on disk.
std::string ask(
    const std::string& dir, const char* variant, const Onnx::ImageData& image,
    const std::string& prompt, int max_tokens, std::string& family)
{
  const std::string v = variant;
  const auto onnx = dir + "/onnx/";
  if(!std::filesystem::exists(onnx + "decoder_model_merged" + v + ".onnx"))
    return {};
  Onnx::FastVLMInference vlm(
      onnx + "vision_encoder" + v + ".onnx", onnx + "embed_tokens" + v + ".onnx",
      onnx + "decoder_model_merged" + v + ".onnx", dir + "/tokenizer.json");
  family = vlm.familyName();
  return vlm.generateResponse(image, prompt, 0.f, max_tokens);
}
}

TEST_CASE("VLM: FastVLM and SmolVLM see the person", "[onnx][vlm]")
{
  const auto image = photo();
  if(image.empty())
    SKIP("test photo not found");
  REQUIRE(OnnxModels::initOnnxRuntime());
  struct Case
  {
    const char* dir;
    const char* variant;
    const char* family;
  };
  int ran = 0;
  for(auto c : {Case{"FastVLM-0.5B-ONNX", "_q4", "llava"},
                Case{"SmolVLM-256M-Instruct", "_q4", "idefics3"}})
  {
    std::string family;
    const auto answer = ask(
        models() + "/" + c.dir, c.variant, image,
        "Is there a person in this image? Answer yes or no.", 8, family);
    if(answer.empty() && family.empty())
      continue;
    INFO(c.dir << ": " << answer);
    ran++;
    CHECK(family == c.family);
    CHECK(lower(answer).find("yes") != std::string::npos);
  }
  if(!ran)
    SKIP("no VLM found under " << models());
}

// 4B parameters: about 30 s and 3 GB of memory on a CPU.
TEST_CASE("VLM: gemma3 describes the image in clean text", "[onnx][vlm]")
{
  const auto image = photo();
  if(image.empty())
    SKIP("test photo not found");
  REQUIRE(OnnxModels::initOnnxRuntime());
  std::string family;
  const auto answer = ask(
      models() + "/gemma-3-4b-it", "_q4", image,
      "List three things you see in this image, as a markdown bullet list.", 64,
      family);
  if(family.empty())
    SKIP("gemma-3-4b-it not found");
  {
    // One <bos> (id 2), first: the tokenizer added one to the template's own
    // and one after the image block, where each text segment was tokenized.
    const auto dir = models() + "/gemma-3-4b-it";
    Onnx::FastVLMInference vlm(
        dir + "/onnx/vision_encoder_q4.onnx", dir + "/onnx/embed_tokens_q4.onnx",
        dir + "/onnx/decoder_model_merged_q4.onnx", dir + "/tokenizer.json");
    const auto ids = vlm.promptTokens("What is this?");
    REQUIRE(!ids.empty());
    CHECK(ids.front() == 2);
    CHECK(std::count(ids.begin(), ids.end(), 2) == 1);
    CHECK(std::count(ids.begin(), ids.end(), vlm.imagePlaceholderId()) == 256);
  }
  INFO(answer);
  CHECK(family == "gemma3");
  // The photo: football players on a pitch.
  const auto a = lower(answer);
  bool seen = false;
  for(const char* w : {"player", "person", "people", "man", "soccer", "football"})
    seen |= a.find(w) != std::string::npos;
  CHECK(seen);
  // SentencePiece's space marker must not leak (HfConfig::replaceSpaceMarkers).
  CHECK(answer.find("▁") == std::string::npos);
}

TEST_CASE("HfConfig: SentencePiece space markers", "[onnx][vlm]")
{
  std::string s = "*▁▁▁**Top:**▁purple";
  Onnx::HfConfig::replaceSpaceMarkers(s);
  CHECK(s == "*   **Top:** purple");
  const auto gemma = models() + "/gemma-3-4b-it/tokenizer.json";
  const auto smol = models() + "/SmolVLM-256M-Instruct/tokenizer.json";
  if(std::filesystem::exists(gemma))
    CHECK(Onnx::HfConfig::decoderReplacesSpaceMarker(gemma));
  if(std::filesystem::exists(smol))
    CHECK_FALSE(Onnx::HfConfig::decoderReplacesSpaceMarker(smol));
}
