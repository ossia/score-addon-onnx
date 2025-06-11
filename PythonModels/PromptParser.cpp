#include "PromptParser.hpp"

#include <ossia/detail/fmt.hpp>

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3.hpp>

#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <string_view>

BOOST_FUSION_ADAPT_STRUCT(
    PythonModels::WeightedPromptElement,
    (std::string, text)(double, value))

namespace PythonModels
{
namespace x3 = boost::spirit::x3;

struct TextContentTag;
struct NumberTag;
struct WeightedPromptElementTag;
struct DataListTag;

const x3::rule<TextContentTag, std::string> text_content = "text_content";
const x3::rule<NumberTag, double> number = "number";
const x3::rule<WeightedPromptElementTag, WeightedPromptElement> data_item
    = "data_item";
const x3::rule<DataListTag, std::vector<WeightedPromptElement>> data_list
    = "data_list";

auto const text_content_def = x3::lexeme[*(x3::char_ - ':')];
auto const number_def = x3::double_;
auto const data_item_def = '(' >> text_content >> ':' >> number >> ')';
auto const data_list_def = data_item % ',';

BOOST_SPIRIT_DEFINE(text_content, number, data_item, data_list);

std::optional<std::vector<WeightedPromptElement>>
parse_input_string(std::string_view str)
{
  std::vector<WeightedPromptElement> result_data;
  auto iterator = str.begin();
  auto const end_iterator = str.end();

  const auto success = x3::phrase_parse(
      iterator, end_iterator, data_list, x3::ascii::space, result_data);

  if (success && iterator == end_iterator)
    return result_data;

  return std::nullopt;
}

std::string quote_string_for_python(std::string_view input)
{
  std::ostringstream oss;
  oss << '\'';

  for (unsigned char c : input)
  {
    switch (c)
    {
      case '\'':
        oss << "\\'";
        break;
      case '\\':
        oss << "\\\\";
        break;
      case '\a':
        oss << "\\a";
        break;
      case '\b':
        oss << "\\b";
        break;
      case '\f':
        oss << "\\f";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      case '\v':
        oss << "\\v";
        break;
      default:
        if (c >= 32 && c <= 126)
          oss << (char)c;
        else
          oss << "\\x" << fmt::format("{:02x}", c);
        break;
    }
  }

  oss << '\'';
  return oss.str();
}
}
