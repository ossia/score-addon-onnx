#include "Tokenizer.hpp"

#include <OnnxModels/JobPool.hpp>

namespace OnnxModels
{
void TokenizerNode::operator()()
{
  if(inputs.tokenizer.file.filename.empty() || !worker.request)
    return;
  if(!m_tokenizers.requested().is(inputs.tokenizer))
    m_tokenizers.request(worker, ModelFile::of(inputs.tokenizer));

  const auto& tok = m_tokenizers.model();
  const bool changed = m_encoded_gen != m_tokenizer_gen || inputs.text.value != m_text
                       || inputs.special.value != m_special;
  if(!tok || !changed || m_busy)
    return;

  m_encoded_gen = m_tokenizer_gen;
  m_text = inputs.text.value;
  m_special = inputs.special.value;
  m_busy = true;
  auto job = std::make_unique<TokenizeJob>();
  job->tokenizer = tok;
  job->text = m_text;
  job->special = m_special;
  worker.request(std::move(job));
}

std::function<void(TokenizerNode&)> TokenizerNode::worker::work(std::unique_ptr<TokenizeJob> job)
{
  if(!job)
    return {};
  if(job->load.active())
    return ModelLoader<const Onnx::TextTokenizer, TokenizeJob>::work<
        &TokenizerNode::m_tokenizers, TokenizerNode>(
        job->load,
        [](const ModelFile& f) { return std::make_shared<const Onnx::TextTokenizer>(f.path); },
        [](TokenizerNode& self) { ++self.m_tokenizer_gen; },
        [](TokenizerNode& self, std::string_view what) {
      self.m_failures.failed("Tokenizer", self.m_tokenizers.requested().path, what);
    });
  if(!job->tokenizer)
    return {};

  try
  {
    auto ids = job->tokenizer->encode(job->text, job->special);
    return [ids = std::move(ids)](TokenizerNode& self) mutable {
      self.m_busy = false;
      self.m_failures.succeeded();
      std::swap(self.outputs.tokens.value, ids);
      // `ids` holds the previous output now; it is small, freed here.
    };
  }
  catch(const std::exception& e)
  {
    return [path = job->tokenizer->path(), what = std::string(e.what())](TokenizerNode& self) {
      self.m_busy = false;
      self.m_failures.failed("Tokenizer", path, what);
    };
  }
}
}
