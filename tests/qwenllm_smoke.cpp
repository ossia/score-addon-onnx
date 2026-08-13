// Smoke-test harness for Onnx::QwenLLMInference against transformers.js-style
// Qwen exports (onnx-community/Qwen*-ONNX layouts).
//
//   qwenllm_smoke <model-dir> [variant...]
//
// <model-dir> is a HuggingFace snapshot: tokenizer.json at the root,
// onnx/model[_variant].onnx. With no variant arguments, every variant present
// on disk is tried. Set SCORE_ONNX_FORCE_PROVIDER=cpu (or cuda...) to pick
// the execution provider, as in score.
#include <Onnx/helpers/QwenLLM.hpp>

#include <chrono>
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

  // HF snapshots keep the models in onnx/; onnxruntime-genai exports put
  // model.onnx next to the tokenizer at the root.
  const std::filesystem::path root = argv[1];
  const auto onnxDir
      = std::filesystem::exists(root / "onnx") ? root / "onnx" : root;
  const auto tokenizer = root / "tokenizer.json";

  std::vector<std::string> variants;
  for (int i = 2; i < argc; ++i)
    variants.push_back(std::string("_") + argv[i]);
  if (variants.empty())
  {
    for (const char* v :
         {"", "_fp16", "_int8", "_uint8", "_quantized", "_q4", "_q4f16",
          "_bnb4"})
    {
      if (std::filesystem::exists(
              onnxDir / (std::string("model") + v + ".onnx")))
        variants.push_back(v);
    }
  }

  int failures = 0;
  for (const auto& v : variants)
  {
    const auto model = (onnxDir / (std::string("model") + v + ".onnx")).string();

    std::printf("=== variant '%s'\n", v.empty() ? "(fp32)" : v.c_str() + 1);
    std::fflush(stdout);

    try
    {
      using clk = std::chrono::steady_clock;
      auto t0 = clk::now();
      Onnx::QwenLLMInference llm(model, tokenizer.string());
      auto t1 = clk::now();

      // Greedy sampling for reproducibility; short reply.
      std::string response
          = llm.generate("What is the capital of France? Answer briefly.", 32, 0.f, 0.9f, 40);
      auto t2 = clk::now();

      // Streaming path: same request, incremental delivery.
      int chunks = 0;
      std::string streamed;
      llm.generateStreaming(
          "What is the capital of France? Answer briefly.",
          [&](const std::string& delta) {
            ++chunks;
            streamed += delta;
            return true;
          },
          32, 0.f, 0.9f, 40);
      std::printf("    streaming: %d chunks -> %s\n", chunks,
                  streamed.empty() ? "(EMPTY!)" : streamed.c_str());
      if (chunks == 0 || streamed.empty())
        ++failures;

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
