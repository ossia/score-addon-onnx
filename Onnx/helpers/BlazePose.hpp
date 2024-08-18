#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>

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

    float probas_visibility[NUM_KPS];
    float probas_presence[NUM_KPS];
    static thread_local std::vector<float> probas_out_presence;
    static thread_local std::vector<float> probas_out_visibility;
    for (int i = 0; i < NUM_KPS; i++)
    {
      probas_visibility[i] = (*out).keypoints[i].visibility;
      probas_presence[i] = (*out).keypoints[i].presence;
    }
    Onnx::softmax(probas_visibility, probas_out_visibility);
    Onnx::softmax(probas_presence, probas_out_presence);
    auto& kps = out->keypoints;
    for (int i = 0; i < NUM_KPS; i++)
    {
      kps[i].visibility = probas_out_visibility[i];
      kps[i].presence = probas_out_presence[i];
    }

    return true;
  }
};
}
