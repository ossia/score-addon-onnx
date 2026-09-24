#include "QwenLLM.hpp"

#include <OnnxModels/TextSegmentation.hpp>

#include <chrono>
#include <cstdio>

namespace OnnxModels
{

QwenLLMNode::QwenLLMNode() noexcept = default;

QwenLLMNode::~QwenLLMNode() = default;

bool QwenLLMNode::needs_reinit() const noexcept
{
  return !llm || last_model_path != inputs.model.file.filename
         || last_tokenizer_path != inputs.tokenizer.file.filename;
}

void QwenLLMNode::initialize_model()
try
{
  if (inputs.model.file.filename.empty()
      || inputs.tokenizer.file.filename.empty())
  {
    return;
  }

  llm = std::make_shared<Onnx::QwenLLMInference>(
      inputs.model.file.filename, inputs.tokenizer.file.filename);

  last_model_path = inputs.model.file.filename;
  last_tokenizer_path = inputs.tokenizer.file.filename;
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "LLM initialization error: %s\n", e.what());
  llm.reset();
}

void QwenLLMNode::request_inference()
{
  if (inference_in_progress)
    return;

  const std::string& prompt = inputs.prompt.value;
  if (prompt.empty())
    return;

  inference_in_progress = true;
  outputs.isGenerating.value = true;
  // The previous response stays visible until the new one is finished.
  accumulated_response.clear();
  partial_buffer.clear();
  ready_partials.clear();
  think_filter.reset();
  filter_thinking = inputs.thinking.value != ThinkingMode::Show;
  total_tokens_generated = 0;
  generation_start_time = std::chrono::steady_clock::now();

  token_stream = std::make_shared<TokenStream>();
  worker.request(
      prompt,
      inputs.temperature.value,
      inputs.topP.value,
      inputs.topK.value,
      inputs.maxTokens.value,
      inputs.thinking.value != ThinkingMode::Off,
      llm,
      token_stream);
}

// Cuts the growing reply into segments according to the Partial mode:
// finished words, finished sentences, or the raw increment. Segments queue
// up in ready_partials and are sent one per tick.
void QwenLLMNode::segmentPartials(std::string_view delta)
{
  std::vector<std::string> segments;
  switch (inputs.partialMode.value)
  {
    case PartialMode::Token:
      if (!delta.empty())
        segments.emplace_back(delta);
      break;
    case PartialMode::Word:
      TextSegmentation::cut_words(partial_buffer, delta, segments);
      break;
    case PartialMode::Sentence:
      TextSegmentation::cut_sentences(partial_buffer, delta, segments);
      break;
  }
  for (auto& s : segments)
    ready_partials.push_back(std::move(s));
}

void QwenLLMNode::appendGenerated(std::string_view delta)
{
  if (filter_thinking)
  {
    const std::string visible = think_filter.feed(delta);
    accumulated_response += visible;
    segmentPartials(visible);
  }
  else
  {
    accumulated_response += delta;
    segmentPartials(delta);
  }
}

void QwenLLMNode::operator()()
try
{
  outputs.partial.value.reset();

  if (!available)
  {
    outputs.response.value = "ONNX Runtime not available";
    return;
  }

  if (needs_reinit())
  {
    initialize_model();
  }

  if (!llm)
  {
    outputs.response.value
        = "Model not initialized. Please provide model and tokenizer files.";
    return;
  }

  // Trigger always (re)starts a generation with the current prompt; prompt
  // and parameter changes only start one on their own outside manual mode.
  if (inputs.trigger.value && !inference_in_progress)
  {
    must_infer = false;
    request_inference();
  }
  else if (
      must_infer && !inputs.manual && !inputs.prompt.value.empty()
      && !inference_in_progress)
  {
    last_processed_prompt = inputs.prompt.value;
    must_infer = false;
    request_inference();
  }

  // Drain the tokens the worker generated since the last tick and cut them
  // into partial segments; the full response is only sent at the end.
  if (token_stream)
  {
    std::vector<std::string> chunk;
    {
      std::lock_guard lock{token_stream->mutex};
      chunk.swap(token_stream->pending);
    }
    if (!chunk.empty())
    {
      std::string delta;
      for (auto& c : chunk)
        delta += c;
      total_tokens_generated += chunk.size();
      appendGenerated(delta);

      using namespace std::chrono;
      const auto elapsed = duration_cast<duration<float>>(
                               steady_clock::now() - generation_start_time)
                               .count();
      if (elapsed > 0.f)
        outputs.tokensPerSecond.value = total_tokens_generated / elapsed;
    }
  }

  if (!ready_partials.empty())
  {
    outputs.partial.value = std::move(ready_partials.front());
    ready_partials.pop_front();
  }
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "LLM processing error: %s\n", e.what());
  outputs.response.value = std::string("Error: ") + e.what();
  outputs.isGenerating.value = false;
  inference_in_progress = false;
}
catch (...)
{
  std::fprintf(stderr, "LLM unknown error\n");
  outputs.response.value = "Unknown error occurred";
  outputs.isGenerating.value = false;
  inference_in_progress = false;
}

std::function<void(QwenLLMNode&)> QwenLLMNode::worker::work(
    std::string prompt,
    float temperature,
    float topP,
    int topK,
    int maxTokens,
    bool thinking,
    std::shared_ptr<Onnx::QwenLLMInference> llm,
    std::shared_ptr<TokenStream> stream)
{
  if (!llm || prompt.empty() || !stream)
  {
    return [](QwenLLMNode& node)
    {
      node.inference_in_progress = false;
      node.outputs.isGenerating.value = false;
      node.outputs.response.value = "Invalid input for inference";
    };
  }

  try
  {
    // Generate on the worker thread, pushing each text increment into the
    // shared stream; the processing thread drains it every tick.
    llm->generateStreaming(
        prompt,
        [&stream](const std::string& delta)
        {
          std::lock_guard lock{stream->mutex};
          stream->pending.push_back(delta);
          return true;
        },
        maxTokens, temperature, topP, topK, thinking);

    return [](QwenLLMNode& node)
    {
      // Applied on the processing thread after generation ended: drain
      // whatever the last tick left in the stream, flush the unterminated
      // tail as a final partial, and emit the finished response.
      if (node.token_stream)
      {
        std::vector<std::string> chunk;
        {
          std::lock_guard lock{node.token_stream->mutex};
          chunk.swap(node.token_stream->pending);
        }
        for (auto& c : chunk)
          node.appendGenerated(c);
        node.total_tokens_generated += chunk.size();
      }
      if (node.filter_thinking)
      {
        const std::string rest = node.think_filter.finish();
        node.accumulated_response += rest;
        node.segmentPartials(rest);
      }
      if (!node.partial_buffer.empty())
      {
        node.ready_partials.push_back(std::move(node.partial_buffer));
        node.partial_buffer.clear();
      }
      node.token_stream.reset();
      node.outputs.response.value = node.accumulated_response;
      node.inference_in_progress = false;
      node.outputs.isGenerating.value = false;
    };
  }
  catch (const std::exception& e)
  {
    std::string errorMsg = std::string("Inference error: ") + e.what();

    return [errorMsg = std::move(errorMsg)](QwenLLMNode& node) mutable
    {
      node.inference_in_progress = false;
      node.outputs.isGenerating.value = false;
      node.outputs.response.value = std::move(errorMsg);
    };
  }
  catch (...)
  {
    std::fprintf(stderr, "Unknown inference error\n");
    return [](QwenLLMNode& node)
    {
      node.inference_in_progress = false;
      node.outputs.isGenerating.value = false;
      node.outputs.response.value = "Unknown inference error";
    };
  }
}

}
