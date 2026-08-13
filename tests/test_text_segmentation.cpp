// Tests for the Language Model node's Partial-output segmentation:
// streaming word / sentence cutting over arbitrary token boundaries.
#include <catch2/catch_test_macros.hpp>

#include <OnnxModels/TextSegmentation.hpp>

#include <string>
#include <vector>

using namespace OnnxModels::TextSegmentation;

namespace
{
// Feed a text in fixed-size chunks, as a token stream would.
template <auto F>
std::pair<std::vector<std::string>, std::string>
feed(const std::string& text, std::size_t chunkSize)
{
  std::string buffer;
  std::vector<std::string> out;
  for (std::size_t i = 0; i < text.size(); i += chunkSize)
    F(buffer, std::string_view(text).substr(i, chunkSize), out);
  return {out, buffer};
}
}

TEST_CASE("Words: split on whitespace, tail stays buffered")
{
  for (std::size_t chunk : {1u, 2u, 3u, 100u})
  {
    auto [words, rest] = feed<cut_words>("Hello brave  new\nworld", chunk);
    REQUIRE(words == std::vector<std::string>{"Hello", "brave", "new"});
    REQUIRE(rest == "world");
  }
}

TEST_CASE("Words: trailing whitespace completes the last word")
{
  auto [words, rest] = feed<cut_words>("one two ", 4);
  REQUIRE(words == std::vector<std::string>{"one", "two"});
  REQUIRE(rest.empty());
}

TEST_CASE("Sentences: cut after terminator + space")
{
  for (std::size_t chunk : {1u, 5u, 100u})
  {
    auto [sents, rest]
        = feed<cut_sentences>("First one. Second one! Third", chunk);
    REQUIRE(sents == std::vector<std::string>{"First one.", "Second one!"});
    REQUIRE(rest == "Third");
  }
}

TEST_CASE("Sentences: decimals and abbreviation-less text survive")
{
  auto [sents, rest] = feed<cut_sentences>("Pi is 3.14159 exactly. Yes", 1);
  REQUIRE(sents == std::vector<std::string>{"Pi is 3.14159 exactly."});
  REQUIRE(rest == "Yes");
}

TEST_CASE("Sentences: newline ends a sentence even without punctuation")
{
  auto [sents, rest] = feed<cut_sentences>("A haiku line\nsecond line", 3);
  REQUIRE(sents == std::vector<std::string>{"A haiku line"});
  REQUIRE(rest == "second line");
}

TEST_CASE("Sentences: trailing quotes and ellipses belong to the sentence")
{
  auto [sents, rest]
      = feed<cut_sentences>("He said \"stop.\" Then... nothing. End", 2);
  REQUIRE(
      sents
      == std::vector<std::string>{"He said \"stop.\"", "Then... nothing."});
  REQUIRE(rest == "End");
}

TEST_CASE("Sentences: terminator at end of buffer waits for more text")
{
  std::string buffer;
  std::vector<std::string> out;
  cut_sentences(buffer, "Wait.", out);
  REQUIRE(out.empty());
  REQUIRE(buffer == "Wait.");
  // The next chunk starts with whitespace: now it is a finished sentence.
  cut_sentences(buffer, " More", out);
  REQUIRE(out == std::vector<std::string>{"Wait."});
  REQUIRE(buffer == "More");
}

TEST_CASE("Question marks and exclamations")
{
  auto [sents, rest] = feed<cut_sentences>("Really?! Yes! Sure", 1);
  REQUIRE(sents == std::vector<std::string>{"Really?!", "Yes!"});
  REQUIRE(rest == "Sure");
}
