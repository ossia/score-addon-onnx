#include "FastVLM.hpp"

#include <QDebug>
#include <QImage>

namespace OnnxModels
{

FastVLMNode::FastVLMNode() noexcept = default;

FastVLMNode::~FastVLMNode() = default;

bool FastVLMNode::needsReinitialization() const
{
  return !vlm ||
         lastVisionEncoderPath != inputs.visionEncoder.file.filename ||
         lastEmbedTokensPath != inputs.embedTokens.file.filename ||
         lastDecoderPath != inputs.decoder.file.filename ||
         lastTokenizerPath != inputs.tokenizer.file.filename;
}

void FastVLMNode::initializeModel()
{
  try
  {
    if (inputs.visionEncoder.file.filename.empty() ||
        inputs.embedTokens.file.filename.empty() ||
        inputs.decoder.file.filename.empty() ||
        inputs.tokenizer.file.filename.empty())
    {
      return;
    }

    vlm = std::make_unique<Onnx::FastVLMInference>(
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

void FastVLMNode::operator()()
try
{
  if (!available)
  {
    outputs.response.value = "ONNX Runtime not available";
    return;
  }

  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
  {
    return;
  }

  if (needsReinitialization())
  {
    initializeModel();
  }

  if (!vlm)
  {
    outputs.response.value = "Model not initialized. Please provide all required model files.";
    return;
  }

  if (!in_tex.bytes || in_tex.width <= 0 || in_tex.height <= 0)
  {
    outputs.response.value = "No input image provided";
    return;
  }

  // Convert texture to QImage
  QImage image(
      reinterpret_cast<const uchar*>(in_tex.bytes),
      in_tex.width,
      in_tex.height,
      QImage::Format_RGBA8888);

  if (image.isNull())
  {
    outputs.response.value = "Invalid input image";
    return;
  }

  std::string response = vlm->generateResponse(
      image,
      inputs.prompt.value,
      inputs.temperature.value);

  outputs.response.value = std::move(response);
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

}
