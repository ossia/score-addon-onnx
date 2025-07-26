#pragma once
#include <OnnxModels/Utils.hpp>
#include <Onnx/helpers/FastVLM.hpp>
#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

namespace OnnxModels
{

struct FastVLMNode : OnnxObject
{
public:
  halp_meta(name, "Fast VLM");
  halp_meta(c_name, "fastvlm");
  halp_meta(category, "AI/Vision Language Model");
  halp_meta(author, "Fast VLM authors, Onnxruntime");
  halp_meta(description, "Vision Language Model for image captioning and visual question answering.");
  halp_meta(uuid, "a1b2c3d4-e5f6-7890-abcd-ef1234567890");

  struct
  {
    halp::texture_input<"Image"> image;

    struct : halp::file_port<"Vision Encoder", halp::mmap_file_view> {
      halp_meta(extensions, "*.onnx");
    } visionEncoder;
    
    struct : halp::file_port<"Embed Tokens", halp::mmap_file_view> {
      halp_meta(extensions, "*.onnx");
    } embedTokens;
    
    struct : halp::file_port<"Decoder", halp::mmap_file_view> {
      halp_meta(extensions, "*.onnx");
    } decoder;
    
    struct : halp::file_port<"Tokenizer", halp::mmap_file_view> {
      halp_meta(extensions, "*.json");
    } tokenizer;

    halp::lineedit<"Prompt", "What do you see in this image?"> prompt;
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

private:
  std::unique_ptr<Onnx::FastVLMInference> vlm;
  std::string lastVisionEncoderPath;
  std::string lastEmbedTokensPath; 
  std::string lastDecoderPath;
  std::string lastTokenizerPath;
  
  void initializeModel();
  bool needsReinitialization() const;
};

}
