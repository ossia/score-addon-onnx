#pragma once
#include <QImage>

#include <Onnx/helpers/FastVLM.hpp>
#include <OnnxModels/Utils.hpp>
#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <functional>
#include <memory>

namespace OnnxModels
{

struct FastVLMNode : OnnxObject
{
public:
  halp_meta(name, "Fast VLM");
  halp_meta(c_name, "fastvlm");
  halp_meta(category, "AI/Vision Language Model");
  halp_meta(author, "Fast VLM authors, Onnxruntime");
  halp_meta(
      description,
      "Vision Language Model for image captioning and visual question "
      "answering.");
  halp_meta(uuid, "3a3b4824-2b39-4cc0-9b6c-6c030de40dc4");

  struct
  {
    struct : halp::texture_input<"Image">
    {
      // Request computation when image changes
      void update(FastVLMNode& g)
      {
        if (g.available && g.vlm)
        {
          g.requestInference();
        }
      }
    } image;

    ModelPort<"Vision Encoder"> visionEncoder;
    ModelPort<"Embed Tokens"> embedTokens;
    ModelPort<"Decoder"> decoder;

    struct : halp::file_port<"Tokenizer", halp::mmap_file_view>
    {
      halp_meta(extensions, "*.json");
      void update(FastVLMNode& self) { self.current_model_invalid = false; }
    } tokenizer;

    struct : halp::lineedit<"Prompt", "What do you see in this image?">
    {
      // Request computation when prompt changes
      void update(FastVLMNode& g)
      {
        if (g.available && g.vlm)
        {
          g.requestInference();
        }
      }
    } prompt;

    halp::knob_f32<"Temperature", halp::range{0.f, 2.f, 1.f}> temperature;
    halp::spinbox_i32<"Max tokens", halp::range{1, 2048, 500}> maxTokens;
  } inputs;

  struct
  {
    halp::val_port<"Response", std::string> response;
  } outputs;

  FastVLMNode() noexcept;
  ~FastVLMNode();

  void operator()();

  // Worker thread infrastructure
  struct worker
  {
    std::function<void(
        QImage,
        std::string,
        float,
        std::shared_ptr<Onnx::FastVLMInference>)>
        request;

    // Called back in a worker thread
    // The returned function will be later applied in this object's processing thread
    static std::function<void(FastVLMNode&)> work(
        QImage image,
        std::string prompt,
        float temperature,
        std::shared_ptr<Onnx::FastVLMInference> vlm);
  } worker;

private:
  std::shared_ptr<Onnx::FastVLMInference> vlm;
  std::string lastVisionEncoderPath;
  std::string lastEmbedTokensPath;
  std::string lastDecoderPath;
  std::string lastTokenizerPath;

  bool inferenceInProgress = false;

  void initializeModel();
  bool needsReinitialization() const;
  void requestInference();
};

}
