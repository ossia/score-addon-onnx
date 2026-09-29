#include "FastVLM.hpp"

#include <algorithm>
#include <cstdio>

namespace OnnxModels
{

FastVLMNode::FastVLMNode() noexcept = default;

FastVLMNode::~FastVLMNode() = default;

std::array<std::string_view, 4> FastVLMNode::files() const noexcept
{
  return {
      inputs.visionEncoder.file.filename, inputs.embedTokens.file.filename,
      inputs.decoder.file.filename, inputs.tokenizer.file.filename};
}

void FastVLMNode::requestLoad()
{
  const auto f = files();
  loading = true;
  auto job = std::make_unique<VlmJob>();
  job->kind = VlmJob::Kind::Load;
  for(std::size_t i = 0; i < f.size(); ++i)
  {
    requested[i] = f[i];
    job->files[i] = f[i];
  }
  worker.request(std::move(job));
}

void FastVLMNode::dispose(std::shared_ptr<Onnx::FastVLMInference> old)
{
  if(!old)
    return;
  auto job = std::make_unique<VlmJob>();
  job->kind = VlmJob::Kind::Dispose;
  job->vlm = std::move(old);
  worker.request(std::move(job));
}

void FastVLMNode::requestInference()
{
  if(inferenceInProgress || !vlm)
    return;

  auto& in_tex = inputs.image.texture;
  if(!in_tex.bytes || in_tex.width <= 0 || in_tex.height <= 0)
    return;

  // The worker owns a copy of the pixels: the texture is only valid now.
  auto job = std::make_unique<VlmJob>();
  job->kind = VlmJob::Kind::Infer;
  job->image.width = in_tex.width;
  job->image.height = in_tex.height;
  job->image.pixels.assign(
      reinterpret_cast<const unsigned char*>(in_tex.bytes),
      reinterpret_cast<const unsigned char*>(in_tex.bytes)
          + static_cast<std::size_t>(in_tex.width) * in_tex.height * 4);
  job->prompt = inputs.prompt.value;
  job->temperature = inputs.temperature.value;
  job->maxTokens = inputs.maxTokens.value;
  job->vlm = vlm;

  inferenceInProgress = true;
  worker.request(std::move(job));
}

void FastVLMNode::operator()()
try
{
  if (!available)
  {
    outputs.response.value = "ONNX Runtime not available";
    return;
  }
  // A file that is set but could not be read: say which one instead of
  // staying silent.
  auto missing = [this](const auto& port, const char* name) {
    if (!port.current_model_invalid || port.file.filename.empty())
      return false;
    outputs.response.value = std::string("Cannot read the ") + name + " file: "
                             + std::string(port.file.filename);
    return true;
  };
  if (missing(inputs.visionEncoder, "Vision Encoder")
      || missing(inputs.embedTokens, "Embed Tokens")
      || missing(inputs.decoder, "Decoder") || missing(inputs.tokenizer, "Tokenizer"))
    return;

  // New files: the models load on the worker; the running ones (if any) keep
  // answering until they arrive.
  const auto f = files();
  const bool complete
      = std::none_of(f.begin(), f.end(), [](std::string_view s) { return s.empty(); });
  if (complete && !loading && !std::equal(f.begin(), f.end(), requested.begin()))
    requestLoad();

  if (!vlm)
  {
    if (!complete)
      outputs.response.value
          = "Model not initialized. Please provide all required model files.";
    return;
  }

  // Outside manual mode the node re-runs continuously as soon as it is idle;
  // in manual mode only a bang on Trigger starts an inference.
  if (!inputs.manual || inputs.trigger.value)
    requestInference();
}
catch (const std::exception& e)
{
  outputs.response.value = std::string("Error: ") + e.what();
}
catch (...)
{
  outputs.response.value = "Unknown error occurred";
}

std::function<void(FastVLMNode&)> FastVLMNode::worker::work(std::unique_ptr<VlmJob> job)
{
  if (!job)
    return {};

  switch (job->kind)
  {
    case VlmJob::Kind::Dispose:
      job->vlm.reset();
      return {};

    case VlmJob::Kind::Load:
      try
      {
        const auto& f = job->files;
        auto vlm = std::make_shared<Onnx::FastVLMInference>(f[0], f[1], f[2], f[3]);
        return [vlm = std::move(vlm), files = std::move(job->files)](
                   FastVLMNode& node) mutable {
          node.loading = false;
          if (!std::equal(files.begin(), files.end(), node.files().begin()))
          {
            node.dispose(std::move(vlm)); // other files were picked meanwhile
            return;
          }
          std::swap(node.vlm, vlm);
          node.dispose(std::move(vlm));
        };
      }
      catch (const std::exception& e)
      {
        std::string what = std::string("Cannot load the model: ") + e.what();
        std::fprintf(stderr, "Vision Language Model: %s\n", what.c_str());
        return [what = std::move(what)](FastVLMNode& node) mutable {
          node.loading = false;
          node.outputs.response.value = std::move(what);
        };
      }

    case VlmJob::Kind::Infer:
      break;
  }

  if (!job->vlm || job->image.empty() || job->prompt.empty())
  {
    return [](FastVLMNode& node) {
      node.inferenceInProgress = false;
      node.outputs.response.value = "Invalid input for inference";
    };
  }

  try
  {
    std::string response = job->vlm->generateResponse(
        job->image, job->prompt, job->temperature, job->maxTokens);
    return [response = std::move(response)](FastVLMNode& node) mutable {
      node.inferenceInProgress = false;
      node.outputs.response.value = std::move(response);
    };
  }
  catch (const std::exception& e)
  {
    std::string errorMsg = std::string("Inference error: ") + e.what();
    return [errorMsg = std::move(errorMsg)](FastVLMNode& node) mutable {
      node.inferenceInProgress = false;
      node.outputs.response.value = std::move(errorMsg);
    };
  }
  catch (...)
  {
    return [](FastVLMNode& node) {
      node.inferenceInProgress = false;
      node.outputs.response.value = "Unknown inference error";
    };
  }
}

}
