// Smoke-test harness for Onnx::FastVLMInference against the
// onnx-community/FastVLM-*-ONNX export variants.
//
//   fastvlm_smoke <model-dir> [variant...]
//
// <model-dir> is a HuggingFace snapshot layout: tokenizer.json at the root,
// onnx/{vision_encoder,embed_tokens,decoder_model_merged}[_variant].onnx.
// With no variant arguments, every variant present on disk is tried.
// Set SCORE_ONNX_FORCE_PROVIDER=cpu (or cuda...) to pick the execution
// provider, as in score.
#include <Onnx/helpers/FastVLM.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::fprintf(stderr, "usage: %s <model-dir> [variant...]\n", argv[0]);
    return 2;
  }

  Ort::InitApi();

  const std::filesystem::path root = argv[1];
  const auto onnxDir = root / "onnx";
  const auto tokenizer = root / "tokenizer.json";

  std::vector<std::string> variants;
  for (int i = 2; i < argc; ++i)
    variants.push_back(argv[i][0] == '-' ? "" : std::string("_") + argv[i]);
  if (variants.empty())
  {
    for (const char* v :
         {"", "_fp16", "_int8", "_uint8", "_quantized", "_q4", "_q4f16",
          "_bnb4"})
    {
      if (std::filesystem::exists(
              onnxDir / (std::string("decoder_model_merged") + v + ".onnx")))
        variants.push_back(v);
    }
  }

  // A small synthetic image: red-to-blue horizontal gradient over a green
  // band, 256x256 so the vision encoder sees a few patches.
  Onnx::ImageData image;
  image.width = 256;
  image.height = 256;
  image.pixels.resize(image.width * image.height * 4);
  for (int y = 0; y < image.height; ++y)
  {
    for (int x = 0; x < image.width; ++x)
    {
      auto* px = &image.pixels[(y * image.width + x) * 4];
      px[0] = (unsigned char)(255 * x / image.width);
      px[1] = (y > image.height / 3 && y < 2 * image.height / 3) ? 200 : 0;
      px[2] = (unsigned char)(255 - 255 * x / image.width);
      px[3] = 255;
    }
  }

  int failures = 0;
  for (const auto& v : variants)
  {
    const auto vision
        = (onnxDir / (std::string("vision_encoder") + v + ".onnx")).string();
    const auto embed
        = (onnxDir / (std::string("embed_tokens") + v + ".onnx")).string();
    const auto decoder
        = (onnxDir / (std::string("decoder_model_merged") + v + ".onnx"))
              .string();

    std::printf(
        "=== variant '%s'\n", v.empty() ? "(fp32)" : v.c_str() + 1);
    std::fflush(stdout);

    try
    {
      using clk = std::chrono::steady_clock;
      auto t0 = clk::now();
      Onnx::FastVLMInference vlm(vision, embed, decoder, tokenizer.string());
      auto t1 = clk::now();

      // BUG-LEDGER F1: the ids and the template come from the model's files.
      // For FastVLM they must equal the values that used to be hard-coded.
      if (root.filename().string().starts_with("FastVLM"))
      {
        const bool ids = vlm.imagePlaceholder() == "<image>"
                         && vlm.imagePlaceholderId() == 151646
                         && vlm.stopTokens().size() == 1
                         && vlm.stopTokens()[0] == 151645;
        const bool tmpl
            = vlm.promptFor("Q?")
              == "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                 "<|im_start|>user\n<image>\nQ?<|im_end|>\n"
                 "<|im_start|>assistant\n";
        std::printf(
            "    config ids %s, chat template %s\n", ids ? "ok" : "DIFFER",
            tmpl ? "ok" : "DIFFERS");
        if (!ids || !tmpl)
          ++failures;
      }

      std::string response
          = vlm.generateResponse(image, "What colors do you see?", 0.f);
      auto t2 = clk::now();

      auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a)
            .count();
      };
      std::printf(
          "    OK  load %lldms  infer %lldms\n    response: %s\n",
          (long long)ms(t0, t1), (long long)ms(t1, t2),
          response.empty() ? "(EMPTY!)" : response.c_str());
      if (response.empty())
        ++failures;
    }
    catch (const std::exception& e)
    {
      std::printf("    FAIL: %s\n", e.what());
      ++failures;
    }
    std::fflush(stdout);
  }

  std::printf("=== %d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
