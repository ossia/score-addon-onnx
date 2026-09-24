#pragma once
// Streaming removal of a reasoning model's leading <think>...</think> block
// (Qwen3, DeepSeek-R1) for the Language Model node. Tags may be split across
// any number of deltas. Kept free of any dependency so the unit tests can
// exercise it without onnxruntime.
#include <cstddef>
#include <string>
#include <string_view>

namespace OnnxModels
{
class ThinkFilter
{
public:
  static constexpr std::string_view open_tag = "<think>";
  static constexpr std::string_view close_tag = "</think>";

  // Returns the part of `delta` that is visible: nothing before the reply's
  // first non-blank text is known not to start a think block, nothing inside
  // the block, and the blank lines that follow it are dropped too.
  std::string feed(std::string_view delta)
  {
    m_buffer += delta;
    std::string out;
    for (;;)
    {
      switch (m_state)
      {
        case Start: {
          const auto first = m_buffer.find_first_not_of(" \t\r\n");
          if (first == std::string::npos)
            return out;
          const std::string_view rest = std::string_view(m_buffer).substr(first);
          if (rest.starts_with(open_tag))
          {
            m_buffer.erase(0, first + open_tag.size());
            m_state = Inside;
            continue;
          }
          if (open_tag.starts_with(rest))
            return out; // maybe the start of the tag: wait
          m_state = Visible;
          continue;
        }
        case Inside: {
          const auto end = m_buffer.find(close_tag);
          if (end == std::string::npos)
          {
            // Keep what could be the start of a split closing tag.
            const std::size_t keep = close_tag.size() - 1;
            if (m_buffer.size() > keep)
              m_buffer.erase(0, m_buffer.size() - keep);
            return out;
          }
          m_buffer.erase(0, end + close_tag.size());
          m_state = AfterBlock;
          continue;
        }
        case AfterBlock: {
          const auto first = m_buffer.find_first_not_of(" \t\r\n");
          if (first == std::string::npos)
          {
            m_buffer.clear();
            return out;
          }
          m_buffer.erase(0, first);
          m_state = Visible;
          continue;
        }
        case Visible:
          out += m_buffer;
          m_buffer.clear();
          return out;
      }
    }
  }

  // End of the reply: text held back while waiting for a possible tag is
  // released. An unterminated block (the reply ran out of tokens while
  // thinking) stays hidden.
  std::string finish()
  {
    std::string out;
    if (m_state == Start || m_state == Visible)
      out = std::move(m_buffer);
    reset();
    return out;
  }

  void reset()
  {
    m_buffer.clear();
    m_state = Start;
  }

  bool thinking() const noexcept { return m_state == Inside; }

private:
  enum State
  {
    Start,
    Inside,
    AfterBlock,
    Visible
  } m_state{Start};
  std::string m_buffer;
};
}
