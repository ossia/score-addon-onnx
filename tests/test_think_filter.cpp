// Tests for the Language Model node's Thinking = Hide filter (BUG-LEDGER L2):
// the leading <think> block is removed over arbitrary token boundaries.
#include <catch2/catch_test_macros.hpp>

#include <OnnxModels/ThinkFilter.hpp>

#include <string>

using OnnxModels::ThinkFilter;

namespace
{
std::string run(const std::string& text, std::size_t chunk)
{
  ThinkFilter f;
  std::string out;
  for (std::size_t i = 0; i < text.size(); i += chunk)
    out += f.feed(std::string_view(text).substr(i, chunk));
  return out + f.finish();
}
}

TEST_CASE("Think filter: the block and the blank lines after it are hidden")
{
  for (std::size_t chunk : {1u, 2u, 3u, 5u, 7u, 100u})
  {
    CHECK(run("<think>\nLet me see. 2+2 is 4.\n</think>\n\nIt is 4.", chunk)
          == "It is 4.");
    CHECK(run("\n<think></think>Paris.", chunk) == "Paris.");
  }
}

TEST_CASE("Think filter: a reply without a block passes through")
{
  for (std::size_t chunk : {1u, 3u, 100u})
  {
    CHECK(run("Paris is the capital.", chunk) == "Paris is the capital.");
    CHECK(run("<b>bold</b> text", chunk) == "<b>bold</b> text");
    CHECK(run("<thin", chunk) == "<thin");
    // Only a leading block is a think block.
    CHECK(run("A <think>x</think> B", chunk) == "A <think>x</think> B");
  }
}

TEST_CASE("Think filter: a block cut by Max tokens shows nothing")
{
  CHECK(run("<think>\nStill reasoning about </thi", 4).empty());
}

TEST_CASE("Think filter: nothing is released while the block is open")
{
  ThinkFilter f;
  CHECK(f.feed("<thi").empty());
  CHECK(f.feed("nk>abc</th").empty());
  CHECK(f.thinking());
  CHECK(f.feed("ink>\n").empty());
  CHECK(f.feed("\nHi").empty() == false);
  CHECK(f.finish().empty());
}
