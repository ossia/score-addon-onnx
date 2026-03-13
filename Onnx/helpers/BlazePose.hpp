#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>

namespace OnnxModels::Blazepose
{

struct Anchor {
    float x;
    float y;
};

struct BlazePose_alignment
{

    // Total anchors: (28*28*2) + (14*14*2) + (7*7*2 * 3) = 2254
  static constexpr int kNumAnchors = 2254;

  // This computes the anchors at compile-time
  static constexpr std::array<Anchor, kNumAnchors> anchors = []() constexpr {
      std::array<Anchor, kNumAnchors> result{};
      int strides[] = {8, 16, 32, 32, 32};
      int idx = 0;

      for (int stride : strides) {
          int grid_size = 224 / stride; // Integer division matches Python //
          
          for (int r = 0; r < grid_size; ++r) {
              for (int c = 0; c < grid_size; ++c) {
                  // Python loop: for _ in range(2)
                  for (int i = 0; i < 2; ++i) {
                      result[idx].x = (static_cast<float>(c) + 0.5f) / static_cast<float>(grid_size);
                      result[idx].y = (static_cast<float>(r) + 0.5f) / static_cast<float>(grid_size);
                      idx++;
                  }
              }
          }
      }
      return result;
  }();

  static constexpr float scalef = 224.0;

  struct pose_align
  {
    Detection detections[kNumAnchors]{};
    
  };

  struct Detection {
    float xmin, ymin, width, height;
    float score;
    std::vector<Point> keypoints;
  };

  static bool processOutput(
      const Onnx::ModelSpec& spec,
      std::span<Ort::Value> outputTensor,
      std::optional<pose_align>& out)
      { 
        out.reset();

        auto &detections = out->detections;
        for (int i = 0; i<kNumAnchors; i++)
        {
          detections[i].score = Onnx::sigmoid(outputTensor[1].GetTensorData<float>()[i]);

          const float* offset = outputTensor[0].GetTensorData<float>()[i*12];
          const auto& anchor = anchors[i];

          // Box Decoding
          float cx = (offset[0] / scalef) + anchor.x;
          float cy = (offset[1] / scalef) + anchor.y;
          float w = offset[2] / scalef;
          float h = offset[3] / scalef;

          Detection d;
          d.score = score;
          d.width = w * static_cast<float>(iw);
          d.height = h * static_cast<float>(ih);
          d.xmin = (cx - w / 2.0f) * static_cast<float>(iw);
          d.ymin = (cy - h / 2.0f) * static_cast<float>(ih);

          // Keypoint Decoding
          for (int k = 0; k < kNumKeypoints; ++k) {
              float kx = ((offset[4 + k * 2] / scalef) + anchor.x) * static_cast<float>(iw);
              float ky = ((offset[4 + k * 2 + 1] / scalef) + anchor.y) * static_cast<float>(ih);
              d.keypoints.push_back({kx, ky});
          }

          detections.push_back(d);
        }

        return true;
      }
};

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

    qDebug() << "BlazePose output tensor has" << Nfloats << "floats";
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
