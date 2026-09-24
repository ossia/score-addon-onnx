#pragma once
// Reads the Hugging Face configuration files that sit next to a tokenizer
// (config.json, generation_config.json, processor_config.json,
// tokenizer.json) for the language model helpers.
#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace Onnx::HfConfig
{
// The file's JSON object, or a discarded / null value when it is missing or
// not an object.
inline nlohmann::json read(const std::filesystem::path& file)
{
  std::ifstream in(file);
  if(!in)
    return {};
  std::stringstream buf;
  buf << in.rdbuf();
  auto json = nlohmann::json::parse(buf.str(), nullptr, /*allow_exceptions=*/false);
  if(json.is_discarded() || !json.is_object())
    return {};
  return json;
}

// A number or a list of numbers.
inline std::vector<int64_t> ids(const nlohmann::json& obj, const char* key)
{
  std::vector<int64_t> out;
  if(!obj.is_object())
    return out;
  const auto it = obj.find(key);
  if(it == obj.end())
    return out;
  if(it->is_number_integer())
    out.push_back(it->get<int64_t>());
  else if(it->is_array())
    for(const auto& v : *it)
      if(v.is_number_integer())
        out.push_back(v.get<int64_t>());
  return out;
}

// The ids that end a reply: generation_config.json first (it may list
// several, and for idefics3 it is the only correct one), then config.json,
// then its text_config. Empty when none declares eos_token_id.
inline std::vector<int64_t> stopTokens(const std::filesystem::path& dir)
{
  if(auto v = ids(read(dir / "generation_config.json"), "eos_token_id"); !v.empty())
    return v;
  const auto config = read(dir / "config.json");
  if(auto v = ids(config, "eos_token_id"); !v.empty())
    return v;
  if(config.is_object() && config.contains("text_config"))
    return ids(config["text_config"], "eos_token_id");
  return {};
}

// The id a VLM's image placeholder stands for: config.json's
// image_token_index (LLaVA, gemma3) or image_token_id (idefics3).
inline std::optional<int64_t> imageTokenId(const std::filesystem::path& dir)
{
  const auto config = read(dir / "config.json");
  for(const char* key : {"image_token_index", "image_token_id"})
    if(auto v = ids(config, key); v.size() == 1)
      return v[0];
  return std::nullopt;
}

// The placeholder text in the prompt: processor_config.json's image_token.
inline std::optional<std::string> imageToken(const std::filesystem::path& dir)
{
  const auto p = read(dir / "processor_config.json");
  if(p.is_object())
    if(const auto it = p.find("image_token"); it != p.end() && it->is_string())
      return it->get<std::string>();
  return std::nullopt;
}

// Whether tokenizer.json declares both tokens among its added tokens.
inline bool addsTokens(
    const std::filesystem::path& tokenizerJson, std::string_view a,
    std::string_view b)
{
  const auto json = read(tokenizerJson);
  if(!json.is_object())
    return false;
  const auto it = json.find("added_tokens");
  if(it == json.end() || !it->is_array())
    return false;
  bool hasA = false, hasB = false;
  for(const auto& t : *it)
  {
    if(!t.is_object() || !t.contains("content") || !t["content"].is_string())
      continue;
    const auto& c = t["content"].get_ref<const std::string&>();
    hasA |= c == a;
    hasB |= c == b;
  }
  return hasA && hasB;
}
}
