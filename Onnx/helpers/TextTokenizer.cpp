#include "TextTokenizer.hpp"

#include <ext_status.h>
#include <ortx_tokenizer.h>
#include <ortx_utils.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace Onnx
{
namespace
{
std::string slurp(const std::filesystem::path& p)
{
  std::ifstream in(p, std::ios::binary);
  if(!in)
    return {};
  std::stringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

std::runtime_error ortxError(const char* what)
{
  return std::runtime_error(std::string(what) + ": " + OrtxGetLastErrorMessage());
}
}

struct TextTokenizer::Impl
{
  OrtxTokenizer* tok{};
  // The blob points into these: they live as long as the tokenizer.
  std::string config, vocab;
  ~Impl()
  {
    if(tok)
      OrtxDispose((OrtxObject**)&tok);
  }
};

std::string TextTokenizer::guessClass(std::string_view tokenizerJson)
{
  const auto json = nlohmann::json::parse(tokenizerJson, nullptr, false);
  if(json.is_discarded() || !json.is_object())
    return "GPT2Tokenizer";
  if(json.contains("added_tokens") && json["added_tokens"].is_array())
    for(const auto& t : json["added_tokens"])
      if(t.is_object() && t.value("content", "") == "<|startoftext|>")
        return "CLIPTokenizer";
  if(json.contains("model") && json["model"].is_object()
     && json["model"].value("type", "") == "Unigram")
    return "T5Tokenizer";
  return "GPT2Tokenizer";
}

TextTokenizer::TextTokenizer(const std::string& tokenizerJson)
    : m_impl{std::make_unique<Impl>()}
    , m_path{tokenizerJson}
{
  const std::filesystem::path file{tokenizerJson};
  m_impl->vocab = slurp(file);
  if(m_impl->vocab.empty())
    throw std::runtime_error("cannot read " + tokenizerJson);

  m_impl->config = slurp(file.parent_path() / "tokenizer_config.json");
  auto config = nlohmann::json::parse(m_impl->config, nullptr, false);
  if(config.is_discarded() || !config.is_object())
    config = nlohmann::json::object();
  if(!config.contains("tokenizer_class") || !config["tokenizer_class"].is_string())
  {
    const auto cls = guessClass(m_impl->vocab);
    config["tokenizer_class"] = cls;
    // The special tokens of openai/clip-vit-base-patch32's config: the library
    // adds the start and end tokens from these.
    if(cls == "CLIPTokenizer")
      for(auto [k, v] :
          {std::pair{"bos_token", "<|startoftext|>"}, {"eos_token", "<|endoftext|>"},
           {"pad_token", "<|endoftext|>"}, {"unk_token", "<|endoftext|>"}})
        if(!config.contains(k))
          config[k] = v;
    m_impl->config = config.dump();
  }
  m_class = config["tokenizer_class"].get<std::string>();

  const OrtxTokenizerBlob blob{m_impl->config, m_impl->vocab};
  if(OrtxCreateTokenizerFromBlob(&m_impl->tok, &blob) != kOrtxOK)
    throw ortxError("cannot load the tokenizer");
}

TextTokenizer::~TextTokenizer() = default;

std::vector<int> TextTokenizer::encode(std::string_view text, bool specialTokens) const
{
  const char* keys[]{"add_special_tokens"};
  const char* values[]{specialTokens ? "true" : "false"};
  OrtxUpdateTokenizerOptions(m_impl->tok, keys, values, 1);

  const std::string s{text};
  const char* inputs[]{s.c_str()};
  OrtxTokenId2DArray* ids{};
  if(OrtxTokenize(m_impl->tok, inputs, 1, &ids) != kOrtxOK)
    throw ortxError("tokenization failed");
  const extTokenId_t* p{};
  size_t n{};
  OrtxTokenId2DArrayGetItem(ids, 0, &p, &n);
  std::vector<int> out(p, p + n);
  OrtxDispose((OrtxObject**)&ids);
  return out;
}
}
