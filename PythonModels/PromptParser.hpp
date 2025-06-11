#pragma once
#include <optional>
#include <string>
#include <vector>

#include <string_view>

namespace PythonModels
{
struct WeightedPromptElement
{
  std::string text;
  double value;
};

/**
 * Small parser for the following prompt language:
 * 
 * (some text, 0.1), (other text, 0.5), (blablabla, 1)
 */
std::optional<std::vector<WeightedPromptElement>>
parse_input_string(std::string_view str);

/**
 * Quotes an arbitrary C++ string to be a valid Python string literal
 * delimited by single quotes.
 */
std::string quote_string_for_python(std::string_view input);

}
