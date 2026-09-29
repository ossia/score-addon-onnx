#include "QwenLLM.hpp"

#include <OnnxModels/TextSegmentation.hpp>

#include <chrono>
#include <cstdio>

namespace OnnxModels
{

QwenLLMNode::QwenLLMNode() noexcept = default;

// A generation still running for this node stops after its current token.
QwenLLMNode::~QwenLLMNode()
{
  cancel_generation();
}

void QwenLLMNode::cancel_generation() noexcept
{
  if (token_stream)
    token_stream->cancelled = true;
}

void QwenLLMNode::request_inference()
{
  if (inference_in_progress || !llm)
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

  token_stream = std::make_shared<LlmTokenStream>();
  auto job = std::make_unique<LlmJob>();
  job->prompt = prompt;
  job->temperature = inputs.temperature.value;
  job->topP = inputs.topP.value;
  job->topK = inputs.topK.value;
  job->maxTokens = inputs.maxTokens.value;
  job->thinking = inputs.thinking.value != ThinkingMode::Off;
  job->llm = llm;
  job->stream = token_stream;
  worker.request(std::move(job));
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

// The tokens the worker generated since the last tick. The worker only holds
// the lock to append one token: when it does, they are taken next tick
// rather than waited for.
void QwenLLMNode::drain_stream()
{
  if (!token_stream)
    return;
  std::vector<std::string> chunk;
  {
    std::unique_lock lock{token_stream->mutex, std::try_to_lock};
    if (!lock.owns_lock())
      return;
    chunk.swap(token_stream->pending);
  }
  if (chunk.empty())
    return;

  std::string delta;
  for (auto& c : chunk)
    delta += c;
  total_tokens_generated += chunk.size();
  appendGenerated(delta);

  using namespace std::chrono;
  const auto elapsed
      = duration_cast<duration<float>>(steady_clock::now() - generation_start_time)
            .count();
  if (elapsed > 0.f)
    outputs.tokensPerSecond.value = total_tokens_generated / elapsed;
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

  // New files: the model loads on the worker; the running one (if any) keeps
  // answering until it arrives, with the generation it was running stopped.
  // Files that fail are not loaded again until one of them changes.
  const auto& model = inputs.model.file.filename;
  const auto& tokenizer = inputs.tokenizer.file.filename;
  if (const auto& r = models.requested();
      !model.empty() && !tokenizer.empty()
      && !(r.model.is(inputs.model) && r.tokenizer.is(inputs.tokenizer)))
  {
    cancel_generation();
    models.request(
        worker, LlmFiles{ModelFile::of(inputs.model), ModelFile::of(inputs.tokenizer)});
  }

  if (!llm)
  {
    if (model.empty() || tokenizer.empty())
      outputs.response.value
          = "Model not initialized. Please provide model and tokenizer files.";
    return;
  }

  // Trigger always (re)starts a generation with the current prompt; prompt
  // and parameter changes only start one on their own outside manual mode.
  // A generation of an outdated prompt stops, the new one starts once it has.
  // While new files load, such a change waits for them.
  const bool start
      = inputs.trigger.value.has_value() || restart
        || (must_infer && !models.loading() && !inputs.manual && !inputs.prompt.value.empty());
  if (start)
  {
    if (inference_in_progress)
    {
      cancel_generation();
      restart = true;
    }
    else
    {
      must_infer = false;
      restart = false;
      request_inference();
    }
  }

  drain_stream();

  if (!ready_partials.empty())
  {
    outputs.partial.value = std::move(ready_partials.front());
    ready_partials.pop_front();
  }
}
catch (const std::exception& e)
{
  outputs.response.value = std::string("Error: ") + e.what();
}
catch (...)
{
  outputs.response.value = "Unknown error occurred";
}

std::function<void(QwenLLMNode&)> QwenLLMNode::worker::work(std::unique_ptr<LlmJob> job)
{
  if (!job)
    return {};

  if (job->load.active())
    return ModelLoader<Onnx::QwenLLMInference, LlmJob, LlmFiles>::work<
        &QwenLLMNode::models, QwenLLMNode>(
        job->load,
        [](const LlmFiles& f) {
      return std::make_shared<Onnx::QwenLLMInference>(f.model.path, f.tokenizer.path);
    },
        [](QwenLLMNode& node) { node.llm = node.models.model(); },
        [](QwenLLMNode& node, std::string_view what) {
      std::string msg = std::string("Cannot load the model: ").append(what);
      std::fprintf(stderr, "Language Model: %s\n", msg.c_str());
      node.outputs.response.value = std::move(msg);
    });

  auto& stream = job->stream;
  if (!job->llm || job->prompt.empty() || !stream)
  {
    return [](QwenLLMNode& node) {
      node.inference_in_progress = false;
      node.outputs.isGenerating.value = false;
      node.outputs.response.value = "Invalid input for inference";
    };
  }

  try
  {
    // Each text increment goes to the shared stream, which the processing
    // thread drains every tick.
    job->llm->generateStreaming(
        job->prompt,
        [&stream](const std::string& delta) {
          if (stream->cancelled.load(std::memory_order_relaxed))
            return false;
          std::lock_guard lock{stream->mutex};
          stream->pending.push_back(delta);
          return true;
        },
        job->maxTokens, job->temperature, job->topP, job->topK, job->thinking);

    const bool cancelled = stream->cancelled.load(std::memory_order_relaxed);
    return [cancelled](QwenLLMNode& node) {
      node.inference_in_progress = false;
      node.outputs.isGenerating.value = false;
      if (cancelled)
      {
        // Stopped for a newer prompt or a new model: nothing to show.
        node.token_stream.reset();
        return;
      }
      // Drain whatever the last tick left in the stream (the generation has
      // ended: nothing holds the lock), flush the unterminated tail as a
      // final partial, and emit the finished response.
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
    };
  }
  catch (const std::exception& e)
  {
    std::string errorMsg = std::string("Inference error: ") + e.what();
    return [errorMsg = std::move(errorMsg)](QwenLLMNode& node) mutable {
      node.inference_in_progress = false;
      node.outputs.isGenerating.value = false;
      node.token_stream.reset();
      node.outputs.response.value = std::move(errorMsg);
    };
  }
  catch (...)
  {
    return [](QwenLLMNode& node) {
      node.inference_in_progress = false;
      node.outputs.isGenerating.value = false;
      node.token_stream.reset();
      node.outputs.response.value = "Unknown inference error";
    };
  }
}

}
