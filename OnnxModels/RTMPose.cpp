//Copied from BlazePose.cpp

#include "RTMPose.hpp"

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/RTMPose.hpp>

namespace OnnxModels
{

// set input image size for RTMPose
RTMPoseDetector::RTMPoseDetector() noexcept
{
  inputs.image.request_height = 256;
  inputs.image.request_width = 192;
}
RTMPoseDetector::~RTMPoseDetector() = default;

void RTMPoseDetector::operator()()
{
  //create onnx context, copy from Blazepose
  auto& in_tex = inputs.image.texture;

  if (!in_tex.changed)
    return;

  if (!this->ctx)
  {
    this->ctx = std::make_unique<Onnx::OnnxRunContext>(
        this->inputs.model.file.bytes);
  }
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();

  // create input tensor from image

  auto t = Onnx::tensorFromARGB(
      spec.inputs[0], in_tex.bytes, in_tex.width, in_tex.height, 256, 192);
  Ort::Value tensor_inputs[1] = {std::move(t.value)};

  Ort::Value tensor_outputs[2]{Ort::Value{nullptr}, Ort::Value{nullptr}};

  // run inference
  ctx.infer(spec, tensor_inputs, tensor_outputs);

  // process output to obtain keypoints
  outputs.image.create(in_tex.width, in_tex.height);
  outputs.image.texture.changed = true;

  //get_simcc_maximum(simcc_x, simcc_y)

  //draw on img
  for (auto b = outputs.image.texture.bytes;
       b != outputs.image.texture.bytes + outputs.image.texture.bytesize();
       b++)
    *b = rand();
}
}
