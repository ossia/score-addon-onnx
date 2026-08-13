#pragma once
// Streaming text segmentation for the Language Model node's Partial output:
// cuts a growing reply into words or sentences as tokens arrive. Kept free of
// any dependency so the unit tests can exercise it without onnxruntime.
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace OnnxModels::TextSegmentation
{
inline bool is_space(char c)
{
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Appends `delta` to `buffer`, moves every *finished* word (a word followed
// by whitespace) into `out`, and leaves the unfinished tail in `buffer`.
inline void cut_words(
    std::string& buffer, std::string_view delta, std::vector<std::string>& out)
{
  buffer += delta;
  std::size_t pos = 0;
  for (;;)
  {
    std::size_t start = pos;
    while (start < buffer.size() && is_space(buffer[start]))
      ++start;
    std::size_t end = start;
    while (end < buffer.size() && !is_space(buffer[end]))
      ++end;
    if (end == buffer.size())
    {
      pos = start; // incomplete word (or nothing) stays buffered
      break;
    }
    out.emplace_back(buffer.substr(start, end - start));
    pos = end;
  }
  buffer.erase(0, pos);
}

// Appends `delta` to `buffer` and moves every finished sentence into `out`.
// A sentence ends on .!? followed by whitespace (decimals like 3.14 stay
// intact) or on a newline; trailing quotes/brackets belong to the sentence.
// The possibly-unterminated last sentence stays in `buffer`.
inline void cut_sentences(
    std::string& buffer, std::string_view delta, std::vector<std::string>& out)
{
  buffer += delta;

  const auto is_terminator
      = [](char c) { return c == '.' || c == '!' || c == '?'; };
  const auto is_closing = [](char c)
  { return c == '"' || c == '\'' || c == ')' || c == ']' || c == '}'; };

  for (;;)
  {
    std::size_t cut = std::string::npos;
    for (std::size_t i = 0; i < buffer.size(); ++i)
    {
      if (buffer[i] == '\n')
      {
        cut = i + 1;
        break;
      }
      if (is_terminator(buffer[i]))
      {
        // Group the whole terminator run: "?!" ends a sentence, but a run
        // of dots ("..." or "..") is an ellipsis that continues it.
        std::size_t termEnd = i + 1;
        bool onlyDots = buffer[i] == '.';
        while (termEnd < buffer.size() && is_terminator(buffer[termEnd]))
        {
          onlyDots = onlyDots && buffer[termEnd] == '.';
          ++termEnd;
        }
        const bool ellipsis = onlyDots && termEnd - i >= 2;

        std::size_t j = termEnd;
        while (j < buffer.size() && is_closing(buffer[j]))
          ++j;
        if (!ellipsis && j < buffer.size() && is_space(buffer[j]))
        {
          cut = j;
          break;
        }
        // Terminator at the end of the buffer: more text may still arrive,
        // keep waiting (the end-of-generation flush handles the rest).
        i = j - 1;
      }
    }
    if (cut == std::string::npos)
      break;

    // Absorb the whitespace run after the cut so it does not leak into the
    // next sentence (or into the end-of-generation flush).
    while (cut < buffer.size() && is_space(buffer[cut]))
      ++cut;

    std::string sentence = buffer.substr(0, cut);
    buffer.erase(0, cut);

    std::size_t b = 0, e = sentence.size();
    while (b < e && is_space(sentence[b]))
      ++b;
    while (e > b && is_space(sentence[e - 1]))
      --e;
    if (e > b)
      out.emplace_back(sentence.substr(b, e - b));
  }
}
}
