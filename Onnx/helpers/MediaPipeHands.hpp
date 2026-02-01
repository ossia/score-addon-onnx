#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxBase.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <array>
#include <cmath>
#include <optional>
#include <span>
#include <vector>

namespace Onnx::MediaPipeHands
{

// MediaPipe Hand Landmark: 21 keypoints per hand
constexpr int NUM_LANDMARKS = 21;

// Landmark names
constexpr const char* LANDMARK_NAMES[] = {
    "wrist",      "thumb_cmc",  "thumb_mcp",  "thumb_ip",   "thumb_tip",
    "index_mcp",  "index_pip",  "index_dip",  "index_tip",  "middle_mcp",
    "middle_pip", "middle_dip", "middle_tip", "ring_mcp",   "ring_pip",
    "ring_dip",   "ring_tip",   "pinky_mcp",  "pinky_pip",  "pinky_dip",
    "pinky_tip"};

struct Landmark
{
  float x, y, z;
  float confidence() const noexcept { return 1.0f; }
};

struct HandResult
{
  std::vector<Landmark> landmarks;
  float handedness;   // 0 = left, 1 = right
  float hand_flag;    // Confidence that hand is present
  bool is_right_hand;
};

// Hand skeleton connections for visualization
// Each pair is (from_idx, to_idx)
struct HandConnections
{
  // Wrist to finger bases
  static constexpr std::array<std::pair<int, int>, 5> palm = {
      {{0, 1}, {0, 5}, {0, 9}, {0, 13}, {0, 17}}};

  // Thumb
  static constexpr std::array<std::pair<int, int>, 4> thumb = {
      {{1, 2}, {2, 3}, {3, 4}, {0, 1}}};

  // Index finger
  static constexpr std::array<std::pair<int, int>, 4> index = {
      {{5, 6}, {6, 7}, {7, 8}, {0, 5}}};

  // Middle finger
  static constexpr std::array<std::pair<int, int>, 4> middle = {
      {{9, 10}, {10, 11}, {11, 12}, {0, 9}}};

  // Ring finger
  static constexpr std::array<std::pair<int, int>, 4> ring = {
      {{13, 14}, {14, 15}, {15, 16}, {0, 13}}};

  // Pinky finger
  static constexpr std::array<std::pair<int, int>, 4> pinky = {
      {{17, 18}, {18, 19}, {19, 20}, {0, 17}}};

  // All connections for full skeleton
  static constexpr std::array<std::pair<int, int>, 21> all = {{
      // Thumb
      {0, 1},
      {1, 2},
      {2, 3},
      {3, 4},
      // Index
      {0, 5},
      {5, 6},
      {6, 7},
      {7, 8},
      // Middle
      {0, 9},
      {9, 10},
      {10, 11},
      {11, 12},
      // Ring
      {0, 13},
      {13, 14},
      {14, 15},
      {15, 16},
      // Pinky
      {0, 17},
      {17, 18},
      {18, 19},
      {19, 20},
      // Palm cross-connections
      {5, 9},
  }};
};

// Process MediaPipe Hand Landmark output
// Model: 033_Hand_Detection_and_Tracking/01_float32/model_float32.onnx
// Outputs (order may vary, identified by shape):
//   - Identity:0 [1, 1, 1, 1] - hand_flag (presence confidence)
//   - Identity_1:0 [1, 1, 1, 1] - handedness (left/right)
//   - Identity_2:0 [1, 63] - 21 keypoints × 3 (x, y, z)
inline bool processOutput(
    const ModelSpec& spec,
    std::span<Ort::Value> outputs,
    std::optional<HandResult>& result)
{
  result.reset();

  if(outputs.size() < 1)
    return false;

  // Get model input size from spec (NHWC: [N, H, W, C])
  float model_size = 256.0f;  // default
  if(!spec.inputs.empty() && spec.inputs[0].shape.size() == 4)
  {
    model_size = static_cast<float>(spec.inputs[0].shape[1]);
  }

  // Identify outputs by shape
  const float* landmark_data = nullptr;
  const float* hand_flag_data = nullptr;
  const float* handedness_data = nullptr;

  for(size_t i = 0; i < outputs.size(); ++i)
  {
    int64_t total = outputs[i].GetTensorTypeAndShapeInfo().GetElementCount();
    const float* data = outputs[i].GetTensorData<float>();

    if(total == NUM_LANDMARKS * 3)
    {
      // Landmarks tensor [1, 63]
      landmark_data = data;
    }
    else if(total == 1)
    {
      // Flag tensor [1, 1, 1, 1] or [1, 1] - need to distinguish
      // First one encountered is typically hand_flag, second is handedness
      if(!hand_flag_data)
        hand_flag_data = data;
      else if(!handedness_data)
        handedness_data = data;
    }
  }

  if(!landmark_data)
    return false;

  // Get hand flag
  float hand_flag = 1.0f;
  if(hand_flag_data)
  {
    hand_flag = *hand_flag_data;
    // Apply sigmoid if raw logit
    hand_flag = sigmoid(hand_flag);
  }

  if(hand_flag < 0.5f)
  {
    result.reset();
    return false;
  }

  // Get handedness
  float handedness = 0.5f;
  bool is_right = true;
  if(handedness_data)
  {
    handedness = *handedness_data;
    // Apply sigmoid if raw logit
    handedness = sigmoid(handedness);
    is_right = handedness > 0.5f;
  }

  // Parse landmarks - coordinates are in pixel coords (0 to model_size)
  std::vector<Landmark> landmarks(NUM_LANDMARKS);
  for(int i = 0; i < NUM_LANDMARKS; ++i)
  {
    // Normalize to [0, 1]
    landmarks[i].x = landmark_data[i * 3] / model_size;
    landmarks[i].y = landmark_data[i * 3 + 1] / model_size;
    landmarks[i].z = landmark_data[i * 3 + 2] / model_size;  // Z is relative depth
  }

  result = HandResult{
      .landmarks = std::move(landmarks),
      .handedness = handedness,
      .hand_flag = hand_flag,
      .is_right_hand = is_right};

  return true;
}

} // namespace Onnx::MediaPipeHands
