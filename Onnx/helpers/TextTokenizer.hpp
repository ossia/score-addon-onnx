#pragma once
// Text -> token ids with a Hugging Face tokenizer.json, through
// onnxruntime-extensions (BPE: CLIP, GPT-2, Llama, Qwen, Gemma, Whisper, ...;
// Unigram: T5, XLM-R, ...). WordPiece (BERT) is not supported by the library.
//
// The library wants the tokenizer_config.json that sits next to tokenizer.json
// in a Hugging Face snapshot. A lone tokenizer.json (the CLIP export of the
// preset pack) gets a config whose tokenizer_class is guessed from it.
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace Onnx
{
class TextTokenizer
{
public:
  // Throws std::runtime_error with the reason when the file cannot be used.
  explicit TextTokenizer(const std::string& tokenizerJson);
  ~TextTokenizer();
  TextTokenizer(const TextTokenizer&) = delete;
  TextTokenizer& operator=(const TextTokenizer&) = delete;

  std::vector<int> encode(std::string_view text, bool specialTokens) const;

  const std::string& path() const noexcept { return m_path; }
  // The tokenizer_class used: from tokenizer_config.json, or guessed.
  const std::string& tokenizerClass() const noexcept { return m_class; }

  // The class a lone tokenizer.json most likely belongs to.
  static std::string guessClass(std::string_view tokenizerJson);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
  std::string m_path;
  std::string m_class;
};
}
