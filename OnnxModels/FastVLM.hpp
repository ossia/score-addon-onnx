#pragma once
#include <Onnx/helpers/ImageBuffer.hpp>

#include <Onnx/helpers/FastVLM.hpp>
#include <OnnxModels/Utils.hpp>
#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace OnnxModels
{

struct VlmJob
{
  enum class Kind : uint8_t
  {
    Load,    // the models at `files`
    Infer,   // a response for `image` and `prompt`
    Dispose, // frees `vlm` off the processing thread
  } kind = Kind::Infer;
  std::array<std::string, 4> files; // vision encoder, embeddings, decoder, tokenizer
  Onnx::ImageData image;
  std::string prompt;
  float temperature = 1.f;
  int maxTokens = 500;
  std::shared_ptr<Onnx::FastVLMInference> vlm;
};

struct FastVLMNode : OnnxObject
{
public:
  halp_meta(name, "Vision Language Model");
  halp_meta(c_name, "fastvlm");
  halp_meta(category, "AI/Vision Language Model");
  halp_meta(author, "Fast VLM authors, Onnxruntime");
  halp_meta(
      description,
      "Vision Language Model for image captioning and visual question "
      "answering: FastVLM, SmolVLM and Gemma 3 onnx-community exports "
      "(vision encoder, token embeddings, merged decoder).");
  halp_meta(uuid, "3a3b4824-2b39-4cc0-9b6c-6c030de40dc4");

  struct
  {
    struct : halp::texture_input<"Image">
    {
      // Request computation when image changes
      void update(FastVLMNode& g)
      {
        if (g.available && g.vlm && !g.inputs.manual)
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
      void update(FastVLMNode& self)
      {
        current_model_invalid = this->file.bytes.size() < 32;
      }
      bool current_model_invalid{};
    } tokenizer;

    struct : halp::lineedit<"Prompt", "What do you see in this image?">
    {
      // Request computation when prompt changes
      void update(FastVLMNode& g)
      {
        if (g.available && g.vlm && !g.inputs.manual)
        {
          g.requestInference();
        }
      }
    } prompt;

    halp::knob_f32<"Temperature", halp::range{0.f, 2.f, 1.f}> temperature;
    halp::spinbox_i32<"Max tokens", halp::range{1, 2048, 500}> maxTokens;

    // Appended after the original ports so existing presets keep their
    // inlet ids. Outside manual mode the node re-runs continuously; in
    // manual mode inference only happens when Trigger is banged.
    halp::toggle<"Manual mode"> manual;
    halp::val_port<"Trigger", std::optional<halp::impulse>> trigger;
  } inputs;

  struct
  {
    halp::val_port<"Response", std::string> response;
  } outputs;

  FastVLMNode() noexcept;
  ~FastVLMNode();

  void operator()();

  // Loading the four models and running them both happen on the worker.
  struct worker
  {
    std::function<void(std::unique_ptr<VlmJob>)> request;
    static std::function<void(FastVLMNode&)> work(std::unique_ptr<VlmJob> job);
  } worker;

private:
  std::shared_ptr<Onnx::FastVLMInference> vlm;
  // The files of the last load, the running one or one in progress: a load
  // that fails is not retried until a file changes.
  std::array<std::string, 4> requested;
  bool loading = false;
  bool inferenceInProgress = false;

  std::array<std::string_view, 4> files() const noexcept;
  void requestLoad();
  void dispose(std::shared_ptr<Onnx::FastVLMInference> old);
  void requestInference();
};

}
