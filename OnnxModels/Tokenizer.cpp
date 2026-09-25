#include "Tokenizer.hpp"

namespace OnnxModels
{
void TokenizerNode::operator()()
{
  const std::string_view path = inputs.tokenizer.file.filename;
  const bool changed = !m_requested || path != m_path || inputs.text.value != m_text
                       || inputs.special.value != m_special;
  if(!changed || m_busy || path.empty() || !worker.request)
    return;

  m_path.assign(path);
  m_text = inputs.text.value;
  m_special = inputs.special.value;
  m_requested = true;
  m_busy = true;
  worker.request(std::make_unique<TokenizeJob>(TokenizeJob{
      .tokenizer = m_tokenizer, .path = m_path, .text = m_text, .special = m_special}));
}

std::function<void(TokenizerNode&)> TokenizerNode::worker::work(std::unique_ptr<TokenizeJob> job)
{
  if(!job || job->path.empty())
    return {}; // a Dispose job: the tokenizer it held is freed here

  auto tok = job->tokenizer;
  try
  {
    if(!tok || tok->path() != job->path)
      tok = std::make_shared<Onnx::TextTokenizer>(job->path);
    auto ids = tok->encode(job->text, job->special);
    return [tok = std::move(tok), ids = std::move(ids)](TokenizerNode& self) mutable {
      self.m_busy = false;
      self.m_failures.succeeded();
      if(tok != self.m_tokenizer)
      {
        // The old tokenizer's last owner is a job: it is freed on the worker.
        auto old = std::exchange(self.m_tokenizer, std::move(tok));
        if(old && self.worker.request)
          self.worker.request(
              std::make_unique<TokenizeJob>(TokenizeJob{.dispose = std::move(old)}));
      }
      std::swap(self.outputs.tokens.value, ids);
      // `ids` holds the previous output now; it is small, freed here.
    };
  }
  catch(const std::exception& e)
  {
    return [path = std::move(job->path), what = std::string(e.what())](TokenizerNode& self) {
      self.m_busy = false;
      self.m_failures.failed("Tokenizer", path, what);
    };
  }
}
}
