#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxBase.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace Onnx::BlazeFace
{

// BlazeFace outputs 6 facial keypoints per detection
constexpr int NUM_KEYPOINTS = 6;
constexpr const char* KEYPOINT_NAMES[]
    = {"right_eye", "left_eye", "nose", "mouth", "right_ear", "left_ear"};

struct Keypoint
{
  float x, y;
};

struct Detection
{
  float x, y, w, h;                        // Bounding box (normalized 0-1)
  float score;                             // Confidence score
  std::array<Keypoint, NUM_KEYPOINTS> keypoints;  // 6 facial keypoints
};

// Anchor configuration for BlazeFace
struct AnchorConfig
{
  int input_size;
  std::vector<int> strides;
  std::vector<int> anchors_per_stride;
};

// Generate anchors for BlazeFace model
inline std::vector<std::array<float, 2>> generateAnchors(const AnchorConfig& config)
{
  std::vector<std::array<float, 2>> anchors;

  for(size_t i = 0; i < config.strides.size(); ++i)
  {
    int stride = config.strides[i];
    int grid_size = config.input_size / stride;
    int num_anchors = config.anchors_per_stride[i];

    for(int y = 0; y < grid_size; ++y)
    {
      for(int x = 0; x < grid_size; ++x)
      {
        float cx = (x + 0.5f) / grid_size;
        float cy = (y + 0.5f) / grid_size;

        for(int a = 0; a < num_anchors; ++a)
        {
          anchors.push_back({cx, cy});
        }
      }
    }
  }

  return anchors;
}

// Default anchor configurations
inline AnchorConfig getFrontConfig()
{
  // 128x128 front-facing camera model
  // 512 anchors from 8x8 grid (2 per cell) + 384 anchors from 4x4 grid (2 per cell) = 896
  return {
      .input_size = 128,
      .strides = {8, 16},
      .anchors_per_stride = {2, 6}  // Adjusted to get 512 + 384 = 896
  };
}

inline AnchorConfig getBackConfig()
{
  // 256x256 back-facing camera model
  return {
      .input_size = 256,
      .strides = {16, 32},
      .anchors_per_stride = {2, 6}
  };
}

// Pre-computed anchors for 128x128 model (896 total)
inline const std::vector<std::array<float, 2>>& getAnchors128()
{
  static std::vector<std::array<float, 2>> anchors = []() {
    std::vector<std::array<float, 2>> result;
    result.reserve(896);

    // First feature map: 16x16 grid, 2 anchors per cell = 512 anchors
    for(int y = 0; y < 16; ++y)
    {
      for(int x = 0; x < 16; ++x)
      {
        float cx = (x + 0.5f) / 16.0f;
        float cy = (y + 0.5f) / 16.0f;
        result.push_back({cx, cy});
        result.push_back({cx, cy});
      }
    }

    // Second feature map: 8x8 grid, 6 anchors per cell = 384 anchors
    for(int y = 0; y < 8; ++y)
    {
      for(int x = 0; x < 8; ++x)
      {
        float cx = (x + 0.5f) / 8.0f;
        float cy = (y + 0.5f) / 8.0f;
        for(int a = 0; a < 6; ++a)
        {
          result.push_back({cx, cy});
        }
      }
    }

    return result;
  }();
  return anchors;
}

// Pre-computed anchors for 256x256 model (896 total)
inline const std::vector<std::array<float, 2>>& getAnchors256()
{
  static std::vector<std::array<float, 2>> anchors = []() {
    std::vector<std::array<float, 2>> result;
    result.reserve(896);

    // First feature map: 16x16 grid, 2 anchors per cell = 512 anchors
    for(int y = 0; y < 16; ++y)
    {
      for(int x = 0; x < 16; ++x)
      {
        float cx = (x + 0.5f) / 16.0f;
        float cy = (y + 0.5f) / 16.0f;
        result.push_back({cx, cy});
        result.push_back({cx, cy});
      }
    }

    // Second feature map: 8x8 grid, 6 anchors per cell = 384 anchors
    for(int y = 0; y < 8; ++y)
    {
      for(int x = 0; x < 8; ++x)
      {
        float cx = (x + 0.5f) / 8.0f;
        float cy = (y + 0.5f) / 8.0f;
        for(int a = 0; a < 6; ++a)
        {
          result.push_back({cx, cy});
        }
      }
    }

    return result;
  }();
  return anchors;
}

// Decode BlazeFace output tensors to detections
// Outputs:
//   scores1 [1, 512, 1], scores2 [1, 384, 1]
//   boxes1 [1, 512, 16], boxes2 [1, 384, 16]
// Each box has 16 values: [cx, cy, w, h, kp0_x, kp0_y, ..., kp5_x, kp5_y]
inline std::vector<Detection> decode(
    const float* scores1,
    const float* scores2,
    const float* boxes1,
    const float* boxes2,
    int num_anchors1,
    int num_anchors2,
    int input_size,
    float score_threshold = 0.5f)
{
  const auto& anchors = (input_size <= 128) ? getAnchors128() : getAnchors256();
  std::vector<Detection> detections;

  auto processAnchors = [&](const float* scores, const float* boxes, int num_anchors,
                            int anchor_offset) {
    for(int i = 0; i < num_anchors; ++i)
    {
      float score = sigmoid(scores[i]);
      if(score < score_threshold)
        continue;

      const float* box = boxes + i * 16;
      const auto& anchor = anchors[anchor_offset + i];

      Detection det;
      det.score = score;

      // Decode bounding box (relative to anchor)
      float cx = anchor[0] + box[0] / input_size;
      float cy = anchor[1] + box[1] / input_size;
      float w = box[2] / input_size;
      float h = box[3] / input_size;

      det.x = cx - w / 2.0f;
      det.y = cy - h / 2.0f;
      det.w = w;
      det.h = h;

      // Decode keypoints (relative to anchor)
      for(int k = 0; k < NUM_KEYPOINTS; ++k)
      {
        det.keypoints[k].x = anchor[0] + box[4 + k * 2] / input_size;
        det.keypoints[k].y = anchor[1] + box[4 + k * 2 + 1] / input_size;
      }

      detections.push_back(det);
    }
  };

  processAnchors(scores1, boxes1, num_anchors1, 0);
  processAnchors(scores2, boxes2, num_anchors2, num_anchors1);

  return detections;
}

// Non-Maximum Suppression
inline std::vector<Detection>
nms(std::vector<Detection>& detections, float iou_threshold = 0.3f)
{
  if(detections.empty())
    return {};

  // Sort by score descending
  std::sort(detections.begin(), detections.end(), [](const auto& a, const auto& b) {
    return a.score > b.score;
  });

  std::vector<Detection> result;
  std::vector<bool> suppressed(detections.size(), false);

  auto iou = [](const Detection& a, const Detection& b) -> float {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);

    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - inter_area;

    return (union_area > 0) ? inter_area / union_area : 0.0f;
  };

  for(size_t i = 0; i < detections.size(); ++i)
  {
    if(suppressed[i])
      continue;

    result.push_back(detections[i]);

    for(size_t j = i + 1; j < detections.size(); ++j)
    {
      if(!suppressed[j] && iou(detections[i], detections[j]) > iou_threshold)
      {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

// Process BlazeFace model output
inline bool processOutput(
    const ModelSpec& spec,
    std::span<Ort::Value> outputs,
    int input_size,
    float score_threshold,
    float nms_threshold,
    std::vector<Detection>& result)
{
  result.clear();

  if(outputs.size() < 4)
    return false;

  // BlazeFace outputs: scores1, scores2, boxes1, boxes2
  // Order may vary, identify by shape
  const float* scores1 = nullptr;
  const float* scores2 = nullptr;
  const float* boxes1 = nullptr;
  const float* boxes2 = nullptr;
  static constexpr int num_anchors1 = 512;
  static constexpr int num_anchors2 = 384;

  for(size_t i = 0; i < outputs.size(); ++i)
  {
    auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
    const float* data = outputs[i].GetTensorData<float>();

    if(shape.size() == 3)
    {
      int n = static_cast<int>(shape[1]);
      int c = static_cast<int>(shape[2]);

      if(c == 1)
      {
        // Score tensor
        if(n == 512)
          scores1 = data;
        else if(n == 384)
          scores2 = data;
      }
      else if(c == 16)
      {
        // Box tensor
        if(n == 512)
          boxes1 = data;
        else if(n == 384)
          boxes2 = data;
      }
    }
  }

  if(!scores1 || !scores2 || !boxes1 || !boxes2)
    return false;

  auto detections
      = decode(scores1, scores2, boxes1, boxes2, num_anchors1, num_anchors2, input_size,
               score_threshold);

  result = nms(detections, nms_threshold);

  return true;
}

} // namespace Onnx::BlazeFace
