#pragma once
#include <Onnx/helpers/QwenLLM.hpp>
#include <OnnxModels/Utils.hpp>
#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <functional>
#include <memory>
#include <queue>

namespace OnnxModels
{

struct QwenLLMNode : OnnxObject
{
public:
  halp_meta(name, "Qwen LLM");
  halp_meta(c_name, "qwen_llm");
  halp_meta(category, "AI/Language Model");
  halp_meta(author, "Qwen Team, Onnxruntime");
  halp_meta(
      description,
      "Real-time inference of Qwen language models for text generation.");
  halp_meta(uuid, "f8d7e6c5-4b3a-2c1e-9f8d-7e6c5b4a3f2e");

  struct
  {
    struct : halp::lineedit<"Prompt", "">
    {
      void update(QwenLLMNode& g) { g.must_infer = true; }
    } prompt;

    struct : halp::file_port<"Model", halp::mmap_file_view>
    {
      halp_meta(extensions, "*.onnx");
      void update(QwenLLMNode& g) { g.must_infer = true; }
    } model;

    struct : halp::file_port<"Tokenizer", halp::mmap_file_view>
    {
      halp_meta(extensions, "*.json");
      void update(QwenLLMNode& g) { g.must_infer = true; }
    } tokenizer;

    struct : halp::knob_f32<"Temperature", halp::range{0.f, 2.f, 0.7f}>
    {
      void update(QwenLLMNode& g) { g.must_infer = true; }
    } temperature;
    struct : halp::knob_f32<"Top P", halp::range{0.f, 1.f, 0.9f}>
    {
      void update(QwenLLMNode& g) { g.must_infer = true; }
    } topP;
    struct : halp::spinbox_i32<"Max tokens", halp::range{1, 4096, 512}>
    {
      void update(QwenLLMNode& g) { g.must_infer = true; }
    } maxTokens;
    struct : halp::spinbox_i32<"Top K", halp::range{0, 100, 40}>
    {
      void update(QwenLLMNode& g) { g.must_infer = true; }
    } topK;
  } inputs;

  struct
  {
    halp::val_port<"Response", std::string> response;
    halp::val_port<"Tokens/sec", float> tokensPerSecond;
    halp::toggle<"Generating"> isGenerating;
  } outputs;

  QwenLLMNode() noexcept;
  ~QwenLLMNode();

  void operator()();

  struct worker
  {
    std::function<void(
        std::string,
        float,
        float,
        int,
        int,
        bool,
        std::shared_ptr<Onnx::QwenLLMInference>)>
        request;

    static std::function<void(QwenLLMNode&)> work(
        std::string prompt,
        float temperature,
        float topP,
        int topK,
        int maxTokens,
        bool stream,
        std::shared_ptr<Onnx::QwenLLMInference> llm);
  } worker;

private:
  void initialize_model();
  bool needs_reinit() const noexcept;
  void request_inference();

  std::shared_ptr<Onnx::QwenLLMInference> llm;
  std::string last_model_path;
  std::string last_tokenizer_path;
  std::string last_processed_prompt;

  std::queue<std::string> token_queue;
  std::chrono::steady_clock::time_point generation_start_time;
  int total_tokens_generated = 0;
  bool must_infer = false;
  bool inference_in_progress = false;
};

}
