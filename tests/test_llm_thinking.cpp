// Language Model Thinking port (BUG-LEDGER L2): a reasoning model's <think>
// block was counted against Max tokens and sent as-is. Hide removes it from
// Partial and Response; Off asks the model not to think at all.
#include <OnnxModels/QwenLLM.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace
{
std::string llmDir(const char* name)
{
  const char* env = std::getenv("ONNX_TEST_MODELS");
  return std::string(env ? env : "/mnt/win2/models/models-presets/models")
         + "/language-model/" + name;
}

struct Harness
{
  OnnxModels::QwenLLMNode node;
  std::string model, tokenizer;
  std::string partials;

  Harness(const std::string& dir, const char* variant)
      : model{dir + "/onnx/" + variant}
      , tokenizer{dir + "/tokenizer.json"}
  {
    node.inputs.model.file.filename = model;
    node.inputs.tokenizer.file.filename = tokenizer;
    node.inputs.temperature.value = 0.f; // greedy
    node.inputs.partialMode.value = OnnxModels::QwenLLMNode::Token;
    node.worker.request = [this](auto... args) {
      if(auto done = OnnxModels::QwenLLMNode::worker::work(std::move(args)...))
        done(node);
    };
  }

  std::string ask(const std::string& prompt, int max_tokens)
  {
    node.inputs.prompt.value = prompt;
    node.inputs.maxTokens.value = max_tokens;
    node.inputs.prompt.update(node);
    partials.clear();
    for(int i = 0; i < 10000; i++)
    {
      node();
      if(node.outputs.partial.value)
        partials += *node.outputs.partial.value;
      else if(i > 0 && !node.outputs.isGenerating.value)
        break;
    }
    return node.outputs.response.value;
  }
};
}

TEST_CASE("Language Model: Thinking Show / Hide / Off", "[onnx][llm]")
{
  const auto dir = llmDir("Qwen3-0.6B");
  if(!std::filesystem::exists(dir + "/onnx/model_q4.onnx"))
    SKIP("Qwen3-0.6B not found");
  REQUIRE(OnnxModels::initOnnxRuntime());

  const std::string q = "What is the capital of France? Answer in one word.";
  using M = OnnxModels::QwenLLMNode::ThinkingMode;

  Harness h{dir, "model_q4.onnx"};

  // Show (the default): the block is part of the reply, as before.
  const auto shown = h.ask(q, 48);
  CHECK(shown.find("<think>") != std::string::npos);
  CHECK(h.partials == shown);

  // Hide: same generation, nothing of the block is sent.
  h.node.inputs.thinking.value = M::Hide;
  const auto hidden = h.ask(q, 400);
  CHECK(hidden.find("think>") == std::string::npos);
  CHECK(hidden.find("Paris") != std::string::npos);
  CHECK(h.partials == hidden);

  // Off: the model answers directly, within a few tokens.
  h.node.inputs.thinking.value = M::Off;
  const auto direct = h.ask(q, 16);
  CHECK(direct.find("think>") == std::string::npos);
  CHECK(direct.find("Paris") != std::string::npos);
}

TEST_CASE("Language Model: Thinking Off leaves other models alone", "[onnx][llm]")
{
  const auto dir = llmDir("Qwen2.5-Coder-0.5B-Instruct");
  if(!std::filesystem::exists(dir + "/onnx/model_q4.onnx"))
    SKIP("Qwen2.5-Coder-0.5B-Instruct not found");
  REQUIRE(OnnxModels::initOnnxRuntime());

  const std::string q = "What is the capital of France? Answer in one word.";
  Harness show{dir, "model_q4.onnx"};
  Harness off{dir, "model_q4.onnx"};
  off.node.inputs.thinking.value = OnnxModels::QwenLLMNode::Off;
  CHECK(show.ask(q, 16) == off.ask(q, 16));
}
