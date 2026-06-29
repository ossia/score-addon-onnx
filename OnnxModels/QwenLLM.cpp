#include "QwenLLM.hpp"

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
    std::fprintf(stderr, "QwenLLM: Model or tokenizer path is empty\n");
    return;
  }

  llm = std::make_shared<Onnx::QwenLLMInference>(
      inputs.model.file.filename, inputs.tokenizer.file.filename);

  last_model_path = inputs.model.file.filename;
  last_tokenizer_path = inputs.tokenizer.file.filename;

  std::fprintf(stderr, "QwenLLM: Model initialized successfully\n");
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "QwenLLM initialization error: %s\n", e.what());
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
  outputs.response.value = "";
  total_tokens_generated = 0;
  generation_start_time = std::chrono::steady_clock::now();

  worker.request(
      prompt,
      inputs.temperature.value,
      inputs.topP.value,
      inputs.topK.value,
      inputs.maxTokens.value,
      false,
      llm);
}

void QwenLLMNode::operator()()
try
{
  if (!available)
  {
    outputs.response.value = "ONNX Runtime not available";
    std::fprintf(stderr, "QwenLLM: ONNX Runtime not available\n");
    return;
  }

  if (needs_reinit())
  {
    std::fprintf(stderr, "QwenLLM: Need to reinitialize model\n");
    initialize_model();
  }

  if (!llm)
  {
    outputs.response.value
        = "Model not initialized. Please provide model and tokenizer files.";
    return;
  }

  // Only trigger inference when prompt changes
  if (must_infer && !inputs.prompt.value.empty() && !inference_in_progress)
  {
    last_processed_prompt = inputs.prompt.value;
    must_infer = false;
    request_inference();
  }
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "QwenLLM processing error: %s\n", e.what());
  outputs.response.value = std::string("Error: ") + e.what();
  outputs.isGenerating.value = false;
  inference_in_progress = false;
}
catch (...)
{
  std::fprintf(stderr, "QwenLLM unknown error\n");
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
    bool stream,
    std::shared_ptr<Onnx::QwenLLMInference> llm)
{
  if (!llm || prompt.empty())
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
    std::string response
        = llm->generate(prompt, maxTokens, temperature, topP, topK);
    return [response = std::move(response)](QwenLLMNode& node) mutable
    {
      node.outputs.response.value = std::move(response);
      node.total_tokens_generated = node.outputs.response.value.size() / 4;
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
