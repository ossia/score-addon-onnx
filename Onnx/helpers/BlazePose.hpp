#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>

namespace OnnxModels::Blazepose
{

struct BlazePose_fullbody
{
  static constexpr int NUM_KPS = 39;

  struct keypoint
  {
    float x, y, z;
    float visibility;
    float presence;
    float confidence() const noexcept { return visibility; }
  };

  struct pose_data
  {
    keypoint keypoints[NUM_KPS]{};
  };

  static bool processOutput(
      const Onnx::ModelSpec& spec,
      std::span<Ort::Value> outputTensor,
      std::optional<pose_data>& out)
  {
    out.reset();
    if (outputTensor.size() == 0)
      return false;

    const int Nfloats
        = outputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount();
    if (Nfloats != 195)
      return false;

    const float* data = outputTensor.front().GetTensorData<float>();
    out = pose_data{};
    std::copy_n(
        data, NUM_KPS * 5, reinterpret_cast<float*>(&out->keypoints[0]));

    // Apply sigmoid to convert logits to per-keypoint probabilities
    auto& kps = out->keypoints;
    for (int i = 0; i < NUM_KPS; i++)
    {
      kps[i].visibility = Onnx::sigmoid(kps[i].visibility);
      kps[i].presence = Onnx::sigmoid(kps[i].presence);
    }

    return true;
  }
};
}
