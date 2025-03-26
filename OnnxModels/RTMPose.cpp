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
  auto t = tensorFromARGB(
      spec.inputs[0], in_tex.bytes, in_tex.width, in_tex.height, 192, 256);

  Ort::Value tensor_inputs[1] = {std::move(t.value)};

  // Prepare two output tensors (simcc_x and simcc_y).
  Ort::Value tensor_outputs[2]{Ort::Value{nullptr}, Ort::Value{nullptr}};

  // run inference
  ctx.infer(spec, tensor_inputs, tensor_outputs);

  static const RTMPose::RTMPose_fullbody pose;

  // process output to obtain keypoints
  outputs.image.create(in_tex.width, in_tex.height);
  outputs.image.texture.changed = true;

  std::optional<RTMPose::RTMPose_fullbody::pose_data> out;

  //call decode() and pass it the tensor_output[]
  RTMPose::RTMPose_fullbody::decode(
      std::span<Ort::Value>(tensor_outputs, 2), out);

  if (out)
  {
    // static_assert(sizeof(*out) == sizeof(*outputs.detection.value));
    outputs.detection.value.emplace();
    memcpy((void*)&*outputs.detection.value, (void*)&*out, sizeof(*out));

    auto img = Onnx::drawKeypoints(
        in_tex.bytes, in_tex.width, in_tex.height, 0.5, out->keypoints);

    outputs.image.create(in_tex.width, in_tex.height);
    memcpy(
        outputs.image.texture.bytes,
        img.constBits(),
        in_tex.width * in_tex.height * 4);

    outputs.image.texture.changed = true;
  }
}

}
