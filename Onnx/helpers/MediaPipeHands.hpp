#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxBase.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <algorithm>
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
    std::optional<HandResult>& result,
    float presence_thresh = 0.5f)
{
  result.reset();

  if(outputs.size() < 1)
    return false;

  // Get model input size — robust to NHWC [N,H,W,C] and NCHW [N,C,H,W]
  // (shape[1] is 3 for an NCHW export): take the largest concrete dim.
  float model_size = 256.0f;  // default
  if(!spec.inputs.empty() && spec.inputs[0].shape.size() == 4)
  {
    const auto& s = spec.inputs[0].shape;
    if(const auto m = std::max({s[1], s[2], s[3]}); m > 0)
      model_size = static_cast<float>(m);
  }

  // Identify outputs by shape
  const float* landmark_data = nullptr;
  const float* hand_flag_data = nullptr;
  const float* handedness_data = nullptr;

  for(size_t i = 0; i < outputs.size(); ++i)
  {
    auto info = outputs[i].GetTensorTypeAndShapeInfo();
    if(info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      continue; // fp16/u8 buffers read as float would over-read
    int64_t total = info.GetElementCount();
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
    // Values in [0,1] are taken as probabilities; only logits get sigmoided.
    // (Unconditionally sigmoiding a probability maps [0,1] onto [0.5,0.73],
    // which makes a 0.5 threshold never reject and 0.75 always reject.)
    if(hand_flag < 0.0f || hand_flag > 1.0f)
      hand_flag = sigmoid(hand_flag);
  }

  if(hand_flag < presence_thresh)
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
    // Same probability-vs-logit sniff as hand_flag: sigmoiding an already
    // [0,1] handedness pins it above 0.5, i.e. always "right hand".
    if(handedness < 0.0f || handedness > 1.0f)
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
