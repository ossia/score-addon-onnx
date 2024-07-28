#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>

namespace OnnxModels
{

struct BlazePose_fullbody
{
  static constexpr int NUM_KPS = 39;

  struct keypoint
  {
    float x, y, z;
    float visibility;
    float presence;
  };

  struct pose_data
  {
    keypoint keypoints[NUM_KPS]{};
  };

  static void processOutput(
      const Onnx::ModelSpec& spec,
      std::span<Ort::Value> outputTensor,
      std::optional<pose_data>& out,
      int max_detect = 100,
      float min_confidence = 0.75,
      int image_x = 0,
      int image_y = 0,
      int image_w = 640,
      int image_h = 640,
      int model_w = 640,
      int model_h = 640)
  {
    const int src_cols = image_w;
    const int src_rows = image_h;
    out.reset();
    if (outputTensor.size() > 0)
    {
      const int Nfloats
          = outputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount();
      if (Nfloats == 195)
      {
        const float* data = outputTensor.front().GetTensorData<float>();
        // data layout xxxx... yyyy... zzzz...
        auto kps = reinterpret_cast<const keypoint*>(data);

        out = pose_data{};
        std::copy_n(data, 195, reinterpret_cast<float*>(&out->keypoints[0]));
      }
    }
  }
};
}
