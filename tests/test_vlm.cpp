// Vision Language Model families besides FastVLM (LLaVA-Qwen2): SmolVLM
// (idefics3, 5-D pixel values plus a pixel mask) and gemma3 (896² input), each
// expanding its image placeholder into the tokens the features fill.
#include <tests/TestPaths.hpp>
#include <Onnx/helpers/FastVLM.hpp>
#include <Onnx/helpers/HfConfig.hpp>
#include <OnnxModels/Utils.hpp>

#include <QImage>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace
{
std::string models()
{
  return TestPaths::vlm();
}

Onnx::ImageData photo()
{
  const QImage img
      = QImage(QString::fromStdString(TestPaths::images() + "/body.jpg"))
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

// The snapshot's <variant> exports, or null when they are not on disk.
std::unique_ptr<Onnx::FastVLMInference> load(const std::string& dir, const std::string& variant)
{
  const auto onnx = dir + "/onnx/";
  if(!std::filesystem::exists(onnx + "decoder_model_merged" + variant + ".onnx"))
    return {};
  return std::make_unique<Onnx::FastVLMInference>(
      onnx + "vision_encoder" + variant + ".onnx", onnx + "embed_tokens" + variant + ".onnx",
      onnx + "decoder_model_merged" + variant + ".onnx", dir + "/tokenizer.json");
}
}

TEST_CASE("VLM: FastVLM and SmolVLM see the person", "[onnx][vlm][model]")
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
    const auto vlm = load(models() + "/" + c.dir, c.variant);
    if(!vlm)
      continue;
    ran++;
    const auto answer = vlm->generateResponse(
        image, "Is there a person in this image? Answer yes or no.", 0.f, 8);
    INFO(c.dir << ": " << answer);
    CHECK(std::string_view{vlm->familyName()} == c.family);
    CHECK(lower(answer).find("yes") != std::string::npos);
  }
  if(!ran)
    SKIP("no VLM found under " << models());
}

// 4B parameters: slow and memory-hungry on a CPU.
TEST_CASE("VLM: gemma3 describes the image in clean text", "[onnx][vlm][model]")
{
  const auto image = photo();
  if(image.empty())
    SKIP("test photo not found");
  REQUIRE(OnnxModels::initOnnxRuntime());
  const auto vlm = load(models() + "/gemma-3-4b-it", "_q4");
  if(!vlm)
    SKIP("gemma-3-4b-it not found");
  CHECK(std::string_view{vlm->familyName()} == "gemma3");

  // One <bos> (id 2), first: the tokenizer must not add one to the template's
  // own, nor after the image block where each text segment is tokenized.
  const auto ids = vlm->promptTokens("What is this?");
  REQUIRE(!ids.empty());
  CHECK(ids.front() == 2);
  CHECK(std::count(ids.begin(), ids.end(), 2) == 1);
  CHECK(std::count(ids.begin(), ids.end(), vlm->imagePlaceholderId()) == 256);

  const auto answer = vlm->generateResponse(
      image, "List three things you see in this image, as a markdown bullet list.", 0.f, 64);
  INFO(answer);
  // The photo: football players on a pitch.
  const auto a = lower(answer);
  bool seen = false;
  for(const char* w : {"player", "person", "people", "man", "soccer", "football"})
    seen |= a.find(w) != std::string::npos;
  CHECK(seen);
  // SentencePiece's space marker must not leak (HfConfig::replaceSpaceMarkers).
  CHECK(answer.find("▁") == std::string::npos);
}

TEST_CASE("HfConfig: SentencePiece space markers become spaces", "[onnx][vlm]")
{
  std::string s = "*▁▁▁**Top:**▁purple";
  Onnx::HfConfig::replaceSpaceMarkers(s);
  CHECK(s == "*   **Top:** purple");
}

TEST_CASE("HfConfig: which tokenizers decode the space marker", "[onnx][vlm][model]")
{
  const auto gemma = models() + "/gemma-3-4b-it/tokenizer.json";
  const auto smol = models() + "/SmolVLM-256M-Instruct/tokenizer.json";
  if(!std::filesystem::exists(gemma) && !std::filesystem::exists(smol))
    SKIP("no gemma-3 or SmolVLM tokenizer under " << models());
  if(std::filesystem::exists(gemma))
    CHECK(Onnx::HfConfig::decoderReplacesSpaceMarker(gemma));
  if(std::filesystem::exists(smol))
    CHECK_FALSE(Onnx::HfConfig::decoderReplacesSpaceMarker(smol));
}
