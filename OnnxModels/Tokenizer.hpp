#pragma once
// Text -> token ids, with a Hugging Face tokenizer.json. Feeds the Text Token
// Processor (CLIP's text encoder) or anything else that takes token ids.
#include <Onnx/helpers/TextTokenizer.hpp>
#include <OnnxModels/ModelLoader.hpp>
#include <OnnxModels/Utils.hpp>

#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace OnnxModels
{
// The tokenizer loads on the worker (`load`), and encodes `text` there.
struct TokenizeJob
{
  ModelJob<const Onnx::TextTokenizer> load;
  std::shared_ptr<const Onnx::TextTokenizer> tokenizer;
  std::string text;
  bool special = true;
};

struct TokenizerNode
{
public:
  halp_meta(name, "Tokenizer");
  halp_meta(c_name, "onnx_tokenizer");
  halp_meta(category, "AI/Text");
  halp_meta(author, "ossia team, onnxruntime-extensions");
  halp_meta(
      description,
      "Turns text into the token ids of a Hugging Face tokenizer.json (CLIP, "
      "GPT-2, Llama, Qwen, Gemma, T5, ...), for the Text Token Processor and "
      "other models that take token ids.");
  halp_meta(uuid, "0b6f3c1e-5a7d-4e2b-9c84-3d1f6a2e7b95");

  struct
  {
    halp::lineedit<"Text", ""> text;
    struct : halp::file_port<"Tokenizer", halp::mmap_file_view>
    {
      halp_meta(extensions, "*.json");
    } tokenizer;
    struct : halp::toggle<"Special Tokens", halp::toggle_setup{.init = true}>
    {
      halp_meta(
          description,
          "Add the model's start and end tokens, as the tokenizer's "
          "post-processor declares them (CLIP: 49406 ... 49407)");
    } special;
  } inputs;

  struct
  {
    halp::val_port<"Tokens", std::vector<int>> tokens;
  } outputs;

  struct worker
  {
    std::function<void(std::unique_ptr<TokenizeJob>)> request;
    static std::function<void(TokenizerNode&)> work(std::unique_ptr<TokenizeJob> job);
  } worker;

  void operator()();

private:
  ModelLoader<const Onnx::TextTokenizer, TokenizeJob> m_tokenizers;
  // What the last ids were asked for, whatever came of it.
  uint32_t m_tokenizer_gen = 0, m_encoded_gen = 0;
  std::string m_text;
  bool m_special = true;
  bool m_busy = false;
  FailureLog m_failures;
};
}
