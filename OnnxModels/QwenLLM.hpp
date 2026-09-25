#pragma once
#include <Onnx/helpers/QwenLLM.hpp>
#include <OnnxModels/ThinkFilter.hpp>
#include <OnnxModels/Utils.hpp>
#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace OnnxModels
{

struct QwenLLMNode : OnnxObject
{
public:
  halp_meta(name, "Language Model");
  halp_meta(c_name, "qwen_llm");
  halp_meta(category, "AI/Language Model");
  halp_meta(author, "Qwen Team, Onnxruntime");
  halp_meta(
      description,
      "Real-time inference of local chat language models (Qwen, Llama, "
      "Gemma, Phi, DeepSeek, ... in transformers.js / onnxruntime-genai "
      "ONNX exports) for text generation.");
  halp_meta(uuid, "f8d7e6c5-4b3a-2c1e-9f8d-7e6c5b4a3f2e");

  // Tokens streamed from the worker thread's generation loop, drained on the
  // processing thread every tick.
  struct TokenStream
  {
    std::mutex mutex;
    std::vector<std::string> pending;
  };

  enum PartialMode
  {
    Sentence,
    Word,
    Token
  };

  // What a reasoning model (Qwen3, DeepSeek-R1) does with its <think> block:
  // shown as generated, removed from Partial and Response (it still counts
  // against Max tokens), or not generated at all. Other models ignore it.
  enum ThinkingMode
  {
    Show,
    Hide,
    Off
  };

  struct
  {
    struct : halp::lineedit<"Prompt", "">
    {
      void update(QwenLLMNode& g) { g.must_infer = true; }
    } prompt;

    struct : ModelPort<"Model">
    {
      void update(QwenLLMNode& g)
      {
        ModelPort::update(g);
        g.must_infer = true;
      }
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

    // Appended after the original ports so existing presets keep their
    // inlet ids. In manual mode nothing runs until Trigger is banged.
    halp::toggle<"Manual mode"> manual;
    halp::val_port<"Trigger", std::optional<halp::impulse>> trigger;

    // Granularity of the Partial output during generation.
    struct : halp::enum_t<PartialMode, "Partial mode">
    {
      enum widget
      {
        combobox
      };
    } partialMode;

    struct : halp::enum_t<ThinkingMode, "Thinking">
    {
      enum widget
      {
        combobox
      };
      void update(QwenLLMNode& g) { g.must_infer = true; }
    } thinking;
  } inputs;

  struct
  {
    // The full reply, sent once when the generation finishes.
    halp::val_port<"Response", std::string> response;
    halp::val_port<"Tokens/sec", float> tokensPerSecond;
    halp::toggle<"Generating"> isGenerating;
    // Streamed increments during generation, segmented according to the
    // "Partial mode" input: finished sentences, words, or raw per-tick
    // token text. One message per tick at most.
    halp::val_port<"Partial", std::optional<std::string>> partial;
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
        std::shared_ptr<Onnx::QwenLLMInference>,
        std::shared_ptr<TokenStream>)>
        request;

    static std::function<void(QwenLLMNode&)> work(
        std::string prompt,
        float temperature,
        float topP,
        int topK,
        int maxTokens,
        bool thinking,
        std::shared_ptr<Onnx::QwenLLMInference> llm,
        std::shared_ptr<TokenStream> stream);
  } worker;

private:
  void initialize_model();
  bool needs_reinit() const noexcept;
  void request_inference();

  std::shared_ptr<Onnx::QwenLLMInference> llm;
  std::string last_model_path;
  std::string last_tokenizer_path;
  std::string last_processed_prompt;

  void segmentPartials(std::string_view delta);
  // Adds generated text to the response and the partials, through the
  // think filter when Thinking is Hide or Off.
  void appendGenerated(std::string_view delta);

  ThinkFilter think_filter;
  bool filter_thinking = false;

  std::shared_ptr<TokenStream> token_stream;
  std::string accumulated_response;
  std::string partial_buffer;
  std::deque<std::string> ready_partials;
  std::chrono::steady_clock::time_point generation_start_time;
  int total_tokens_generated = 0;
  bool must_infer = false;
  bool inference_in_progress = false;
};

}
