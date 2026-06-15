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
  float z = 0.f; // RTMW3D: root-relative depth in METERS (simcc_z); else 0
  float confidence;
};

// RTMW3D z decode: z_m = (z_px / (input_h/2) - 1) * Z_RANGE, with Z_RANGE the
// codec constant from mmpose projects/rtmpose3d (also used by rtmlib).
constexpr float RTMW3D_Z_RANGE = 2.1744869f;

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
  // Degenerate shapes (e.g. a dynamic dim that didn't resolve) must not allocate
  // a huge/negative vector or index out of bounds.
  if(num_keypoints <= 0 || num_keypoints > 1024 || bins_x <= 0 || bins_y <= 0
     || input_width <= 0 || input_height <= 0)
    return {};

  std::vector<Keypoint> keypoints(num_keypoints);

  for(int k = 0; k < num_keypoints; ++k)
  {
    const float* x_logits = simcc_x + static_cast<size_t>(k) * bins_x;
    const float* y_logits = simcc_y + static_cast<size_t>(k) * bins_y;

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

  if(config.num_keypoints <= 0 || config.num_keypoints > 1024)
    return false;

  if(format == OutputFormat::PostProcessed)
  {
    // Post-processed format: [1, K, 3] with (x, y, score)
    // Coordinates are in pixel space relative to model input
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    if(info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      return false; // fp16/u8 buffers read as float would over-read
    const float* data = outputs[0].GetTensorData<float>();

    // Trust the actual element count, not the declared K: a dynamic-dim model
    // can resolve to fewer floats than the shape implied -> clamp before reading.
    const int64_t n = info.GetElementCount();
    const int K = static_cast<int>(
        std::min<int64_t>(config.num_keypoints, n / 3));
    keypoints.resize(K);
    for(int k = 0; k < K; ++k)
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

    auto info_x = outputs[0].GetTensorTypeAndShapeInfo();
    auto info_y = outputs[1].GetTensorTypeAndShapeInfo();
    if(info_x.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
       || info_y.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      return false; // fp16/u8 buffers read as float would over-read

    const float* simcc_x = outputs[0].GetTensorData<float>();
    const float* simcc_y = outputs[1].GetTensorData<float>();

    // Clamp the keypoint count to what each buffer actually holds: the x/y
    // tensors can have a different (dynamic) K than detectConfig inferred, so
    // reading num_keypoints*bins from the smaller buffer would run off the end.
    const int64_t nx = info_x.GetElementCount();
    const int64_t ny = info_y.GetElementCount();
    int nk = config.num_keypoints;
    if(config.simcc_bins_x > 0)
      nk = static_cast<int>(std::min<int64_t>(nk, nx / config.simcc_bins_x));
    if(config.simcc_bins_y > 0)
      nk = static_cast<int>(std::min<int64_t>(nk, ny / config.simcc_bins_y));

    keypoints = SimCC::decode(
        simcc_x,
        simcc_y,
        nk,
        config.simcc_bins_x,
        config.simcc_bins_y,
        config.input_width,
        config.input_height,
        config.split_ratio);

    // RTMW3D: a third simcc_z [1, K, bins_z] head carrying root-relative
    // depth. Decode validated against the real rtmw3d-l export + the rtmlib
    // reference: z_m = (argmax_z/split / (input_h/2) - 1) * Z_RANGE.
    if(outputs.size() >= 3 && !keypoints.empty())
    {
      auto info_z = outputs[2].GetTensorTypeAndShapeInfo();
      auto shape_z = info_z.GetShape();
      // Only accept the third output as a depth head when its keypoint axis
      // matches the x/y heads — an unrelated rank-3 aux output (scores etc.)
      // must not be decoded as depth.
      if(shape_z.size() == 3 && shape_z[2] > 1 && config.input_height > 0
         && shape_z[1] == static_cast<int64_t>(keypoints.size())
         && info_z.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      {
        const int bins_z = static_cast<int>(shape_z[2]);
        const float* simcc_z = outputs[2].GetTensorData<float>();
        const int64_t nz = info_z.GetElementCount();
        const int kz = static_cast<int>(std::min<int64_t>(
            static_cast<int64_t>(keypoints.size()), nz / bins_z));
        const float half_h = 0.5f * config.input_height;
        for(int k = 0; k < kz; ++k)
        {
          const float* z_logits = simcc_z + static_cast<size_t>(k) * bins_z;
          int max_idx = 0;
          float max_val = z_logits[0];
          for(int i = 1; i < bins_z; ++i)
            if(z_logits[i] > max_val)
            {
              max_val = z_logits[i];
              max_idx = i;
            }
          const float z_px = max_idx / config.split_ratio;
          keypoints[k].z = (z_px / half_h - 1.f) * RTMW3D_Z_RANGE;
        }
      }
    }
  }

  // Calculate mean confidence (compute before the move below).
  float sum_conf = 0.0f;
  for(const auto& kp : keypoints)
    sum_conf += kp.confidence;
  const float mean_conf
      = keypoints.empty() ? 0.f : sum_conf / static_cast<float>(keypoints.size());

  result = PoseResult{
      .keypoints = std::move(keypoints), .mean_confidence = mean_conf};

  return true;
}

} // namespace Onnx::RTMPose
