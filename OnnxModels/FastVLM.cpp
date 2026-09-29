#include "FastVLM.hpp"

#include <algorithm>
#include <cstdio>

namespace OnnxModels
{

FastVLMNode::FastVLMNode() noexcept = default;

FastVLMNode::~FastVLMNode() = default;

void FastVLMNode::requestInference()
{
  if(inferenceInProgress || !vlm)
    return;

  auto& in_tex = inputs.image.texture;
  if(!in_tex.bytes || in_tex.width <= 0 || in_tex.height <= 0)
    return;

  // The worker owns a copy of the pixels: the texture is only valid now.
  auto job = std::make_unique<VlmJob>();
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
  // answering until they arrive. Files that fail are not loaded again until
  // one of them changes.
  const auto& i = inputs;
  const bool complete = !i.visionEncoder.file.filename.empty()
                        && !i.embedTokens.file.filename.empty()
                        && !i.decoder.file.filename.empty()
                        && !i.tokenizer.file.filename.empty();
  const auto& r = models.requested();
  if (complete
      && !(r[0].is(i.visionEncoder) && r[1].is(i.embedTokens) && r[2].is(i.decoder)
           && r[3].is(i.tokenizer)))
    models.request(
        worker, VlmFiles{
                    ModelFile::of(i.visionEncoder), ModelFile::of(i.embedTokens),
                    ModelFile::of(i.decoder), ModelFile::of(i.tokenizer)});

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

  if (job->load.active())
    return ModelLoader<Onnx::FastVLMInference, VlmJob, VlmFiles>::work<
        &FastVLMNode::models, FastVLMNode>(
        job->load,
        [](const VlmFiles& f) {
      return std::make_shared<Onnx::FastVLMInference>(
          f[0].path, f[1].path, f[2].path, f[3].path);
    },
        [](FastVLMNode& node) { node.vlm = node.models.model(); },
        [](FastVLMNode& node, std::string_view what) {
      std::string msg = std::string("Cannot load the model: ").append(what);
      std::fprintf(stderr, "Vision Language Model: %s\n", msg.c_str());
      node.outputs.response.value = std::move(msg);
    });

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
