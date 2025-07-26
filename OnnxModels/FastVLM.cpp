#include "FastVLM.hpp"

#include <QDebug>
#include <QImage>

namespace OnnxModels
{

FastVLMNode::FastVLMNode() noexcept = default;

FastVLMNode::~FastVLMNode() = default;

bool FastVLMNode::needsReinitialization() const
{
  return !vlm || lastVisionEncoderPath != inputs.visionEncoder.file.filename
         || lastEmbedTokensPath != inputs.embedTokens.file.filename
         || lastDecoderPath != inputs.decoder.file.filename
         || lastTokenizerPath != inputs.tokenizer.file.filename;
}

void FastVLMNode::initializeModel()
{
  try
  {
    if (inputs.visionEncoder.file.filename.empty()
        || inputs.embedTokens.file.filename.empty()
        || inputs.decoder.file.filename.empty()
        || inputs.tokenizer.file.filename.empty())
    {
      return;
    }

    vlm = std::make_shared<Onnx::FastVLMInference>(
        inputs.visionEncoder.file.filename,
        inputs.embedTokens.file.filename,
        inputs.decoder.file.filename,
        inputs.tokenizer.file.filename);

    lastVisionEncoderPath = inputs.visionEncoder.file.filename;
    lastEmbedTokensPath = inputs.embedTokens.file.filename;
    lastDecoderPath = inputs.decoder.file.filename;
    lastTokenizerPath = inputs.tokenizer.file.filename;
  }
  catch (const std::exception& e)
  {
    qDebug() << "FastVLM initialization error:" << e.what();
    vlm.reset();
  }
}

void FastVLMNode::requestInference()
{
  // Don't start new inference if one is already in progress
  if (inferenceInProgress)
    return;

  auto& in_tex = inputs.image.texture;
  if (!in_tex.bytes || in_tex.width <= 0 || in_tex.height <= 0)
    return;

  // Convert texture to QImage
  QImage image(
      reinterpret_cast<const uchar*>(in_tex.bytes),
      in_tex.width,
      in_tex.height,
      QImage::Format_RGBA8888);

  if (image.isNull())
    return;

  // Mark inference as in progress
  inferenceInProgress = true;

  // Start worker thread computation
  worker.request(image, inputs.prompt.value, inputs.temperature.value, vlm);
}

void FastVLMNode::operator()()
try
{
  if (!available)
  {
    outputs.response.value = "ONNX Runtime not available";
    return;
  }

  if (needsReinitialization())
  {
    initializeModel();
  }

  if (!vlm)
  {
    outputs.response.value
        = "Model not initialized. Please provide all required model files.";
    return;
  }

  // The actual processing happens in the worker thread
  // This operator() just handles initialization and model management
  if (!inferenceInProgress)
  {
    requestInference();
  }
}
catch (const std::exception& e)
{
  qDebug() << "FastVLM processing error:" << e.what();
  outputs.response.value = std::string("Error: ") + e.what();
}
catch (...)
{
  qDebug() << "FastVLM unknown error";
  outputs.response.value = "Unknown error occurred";
}

// Worker thread implementation
std::function<void(FastVLMNode&)> FastVLMNode::worker::work(
    QImage image,
    std::string prompt,
    float temperature,
    std::shared_ptr<Onnx::FastVLMInference> vlm)
{
  if (!vlm || image.isNull() || prompt.empty())
  {
    return [](FastVLMNode& node)
    {
      node.inferenceInProgress = false;
      node.outputs.response.value = "Invalid input for inference";
    };
  }

  try
  {
    // Perform the inference in the worker thread
    std::string response = vlm->generateResponse(image, prompt, temperature);

    // Return a function that will be executed in the main thread
    return [response = std::move(response)](FastVLMNode& node) mutable
    {
      node.inferenceInProgress = false;
      node.outputs.response.value = std::move(response);
    };
  }
  catch (const std::exception& e)
  {
    std::string errorMsg = std::string("Inference error: ") + e.what();
    return [errorMsg = std::move(errorMsg)](FastVLMNode& node) mutable
    {
      node.inferenceInProgress = false;
      node.outputs.response.value = std::move(errorMsg);
    };
  }
  catch (...)
  {
    return [](FastVLMNode& node)
    {
      node.inferenceInProgress = false;
      node.outputs.response.value = "Unknown inference error";
    };
  }
}

}
