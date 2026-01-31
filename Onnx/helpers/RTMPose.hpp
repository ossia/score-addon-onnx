#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxBase.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <span>
#include <vector>

namespace Onnx::RTMPose
{

// COCO keypoint definitions (17 keypoints)
constexpr int COCO_NUM_KEYPOINTS = 17;
constexpr const char* COCO_KEYPOINT_NAMES[] = {
    "nose",          "left_eye",      "right_eye",    "left_ear",     "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow",   "right_elbow",  "left_wrist",
    "right_wrist",   "left_hip",      "right_hip",    "left_knee",    "right_knee",
    "left_ankle",    "right_ankle"};

// WholeBody keypoint count (133 keypoints)
// 0-16: body (17), 17-22: feet (6), 23-90: face (68), 91-111: left hand (21), 112-132: right hand (21)
constexpr int WHOLEBODY_NUM_KEYPOINTS = 133;

// COCO skeleton connections for visualization
struct COCOConnections
{
  static constexpr std::array<std::pair<int, int>, 19> skeleton = {{
      // Head
      {0, 1},
      {0, 2},
      {1, 3},
      {2, 4},
      // Torso
      {5, 6},
      {5, 11},
      {6, 12},
      {11, 12},
      // Left arm
      {5, 7},
      {7, 9},
      // Right arm
      {6, 8},
      {8, 10},
      // Left leg
      {11, 13},
      {13, 15},
      // Right leg
      {12, 14},
      {14, 16},
      // Neck approximation
      {0, 5},
      {0, 6},
      {5, 6},
  }};
};

struct Keypoint
{
  float x, y;
  float confidence;
};

struct PoseResult
{
  std::vector<Keypoint> keypoints;
  float mean_confidence;
};

// RTMPose configuration
struct Config
{
  int num_keypoints = COCO_NUM_KEYPOINTS;
  int input_width = 192;
  int input_height = 256;
  int simcc_bins_x = 384;  // input_width * split_ratio
  int simcc_bins_y = 512;  // input_height * split_ratio
  float split_ratio = 2.0f;
};

// SimCC (Simple Coordinate Classification) decoder
// RTMPose outputs classification logits over spatial bins instead of direct coordinates
namespace SimCC
{

// Decode SimCC output to keypoints
// simcc_x: [num_kpts, bins_x] - classification logits for x coordinate
// simcc_y: [num_kpts, bins_y] - classification logits for y coordinate
inline std::vector<Keypoint> decode(
    const float* simcc_x,
    const float* simcc_y,
    int num_keypoints,
    int bins_x,
    int bins_y,
    int input_width,
    int input_height,
    float split_ratio = 2.0f)
{
  std::vector<Keypoint> keypoints(num_keypoints);

  for(int k = 0; k < num_keypoints; ++k)
  {
    const float* x_logits = simcc_x + k * bins_x;
    const float* y_logits = simcc_y + k * bins_y;

    // Find argmax and max value for X
    int max_idx_x = 0;
    float max_val_x = x_logits[0];
    for(int i = 1; i < bins_x; ++i)
    {
      if(x_logits[i] > max_val_x)
      {
        max_val_x = x_logits[i];
        max_idx_x = i;
      }
    }

    // Find argmax and max value for Y
    int max_idx_y = 0;
    float max_val_y = y_logits[0];
    for(int i = 1; i < bins_y; ++i)
    {
      if(y_logits[i] > max_val_y)
      {
        max_val_y = y_logits[i];
        max_idx_y = i;
      }
    }

    // Convert bin index to normalized coordinate [0, 1]
    float x_coord = (max_idx_x / split_ratio) / input_width;
    float y_coord = (max_idx_y / split_ratio) / input_height;

    // Clamp to valid range
    keypoints[k].x = std::clamp(x_coord, 0.0f, 1.0f);
    keypoints[k].y = std::clamp(y_coord, 0.0f, 1.0f);

    // Confidence from sigmoid of max logits
    // Use geometric mean of x and y confidences
    float conf_x = 1.0f / (1.0f + std::exp(-max_val_x));
    float conf_y = 1.0f / (1.0f + std::exp(-max_val_y));
    keypoints[k].confidence = std::sqrt(conf_x * conf_y);
  }

  return keypoints;
}

} // namespace SimCC

// Output format detection
enum class OutputFormat
{
  SimCC,        // 2 outputs: simcc_x [1, K, bins_x], simcc_y [1, K, bins_y]
  PostProcessed // 1 output: [1, K, 3] with (x, y, score) already decoded
};

// Auto-detect configuration from output tensor shapes
inline Config detectConfig(std::span<Ort::Value> outputs, OutputFormat* out_format = nullptr)
{
  Config config;

  if(outputs.empty())
    return config;

  // Check for post-processed format: single output [1, K, 3]
  if(outputs.size() == 1)
  {
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if(shape.size() == 3 && shape[2] == 3)
    {
      config.num_keypoints = static_cast<int>(shape[1]);
      if(out_format)
        *out_format = OutputFormat::PostProcessed;
      return config;
    }
  }

  // SimCC format: 2 outputs
  if(outputs.size() >= 2)
  {
    // Get shapes: simcc_x [1, num_kpts, bins_x], simcc_y [1, num_kpts, bins_y]
    auto shape_x = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto shape_y = outputs[1].GetTensorTypeAndShapeInfo().GetShape();

    if(shape_x.size() >= 3 && shape_y.size() >= 3)
    {
      config.num_keypoints = static_cast<int>(shape_x[1]);
      config.simcc_bins_x = static_cast<int>(shape_x[2]);
      config.simcc_bins_y = static_cast<int>(shape_y[2]);

      // Infer input size from bins (assuming split_ratio = 2)
      config.split_ratio = 2.0f;
      config.input_width = config.simcc_bins_x / 2;
      config.input_height = config.simcc_bins_y / 2;

      if(out_format)
        *out_format = OutputFormat::SimCC;
    }
  }

  return config;
}

// Process RTMPose output (handles both SimCC and post-processed formats)
inline bool processOutput(
    const ModelSpec& spec,
    std::span<Ort::Value> outputs,
    const Config& config,
    std::optional<PoseResult>& result,
    OutputFormat format = OutputFormat::SimCC)
{
  result.reset();

  if(outputs.empty())
    return false;

  std::vector<Keypoint> keypoints;

  if(format == OutputFormat::PostProcessed)
  {
    // Post-processed format: [1, K, 3] with (x, y, score)
    // Coordinates are in pixel space relative to model input
    const float* data = outputs[0].GetTensorData<float>();

    keypoints.resize(config.num_keypoints);
    for(int k = 0; k < config.num_keypoints; ++k)
    {
      // x, y are in pixel coordinates, normalize to [0, 1]
      float x = data[k * 3 + 0] / config.input_width;
      float y = data[k * 3 + 1] / config.input_height;
      float score = data[k * 3 + 2];

      keypoints[k].x = std::clamp(x, 0.0f, 1.0f);
      keypoints[k].y = std::clamp(y, 0.0f, 1.0f);
      keypoints[k].confidence = score;
    }
  }
  else
  {
    // SimCC format: simcc_x [1, K, bins_x], simcc_y [1, K, bins_y]
    if(outputs.size() < 2)
      return false;

    const float* simcc_x = outputs[0].GetTensorData<float>();
    const float* simcc_y = outputs[1].GetTensorData<float>();

    keypoints = SimCC::decode(
        simcc_x,
        simcc_y,
        config.num_keypoints,
        config.simcc_bins_x,
        config.simcc_bins_y,
        config.input_width,
        config.input_height,
        config.split_ratio);
  }

  // Calculate mean confidence
  float sum_conf = 0.0f;
  for(const auto& kp : keypoints)
    sum_conf += kp.confidence;

  result = PoseResult{
      .keypoints = std::move(keypoints),
      .mean_confidence = sum_conf / config.num_keypoints};

  return true;
}

} // namespace Onnx::RTMPose
