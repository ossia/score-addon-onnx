#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxBase.hpp>

#include <array>
#include <optional>
#include <span>
#include <vector>

namespace Onnx::FaceMesh
{

// MediaPipe FaceMesh: 468 landmarks
// FaceMesh V2: 478 landmarks (468 + 10 iris landmarks)
constexpr int NUM_LANDMARKS = 468;
constexpr int NUM_LANDMARKS_V2 = 478;

struct Landmark
{
  float x, y, z;

  float confidence() const noexcept { return 1.0f; }
};

struct FaceMeshResult
{
  std::vector<Landmark> landmarks;
  float face_flag; // Confidence that face is present
};

// Face contour indices for visualization
struct FaceContours
{
  // Lips outer contour
  static constexpr std::array<int, 16> lips_outer = {
      61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0};

  // Lips inner contour
  static constexpr std::array<int, 8> lips_inner = {78, 95, 88, 178, 87, 14, 317, 402};

  // Left eye contour
  static constexpr std::array<int, 16> left_eye = {
      33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246};

  // Right eye contour
  static constexpr std::array<int, 16> right_eye = {
      362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398};

  // Left eyebrow
  static constexpr std::array<int, 8> left_eyebrow = {
      70, 63, 105, 66, 107, 55, 65, 52};

  // Right eyebrow
  static constexpr std::array<int, 8> right_eyebrow = {
      300, 293, 334, 296, 336, 285, 295, 282};

  // Face oval
  static constexpr std::array<int, 36> face_oval = {
      10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
      397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
      172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109};

  // Nose bridge
  static constexpr std::array<int, 4> nose = {168, 6, 197, 195};

  // Left iris (V2 only, indices 468-472)
  static constexpr std::array<int, 5> left_iris = {468, 469, 470, 471, 472};

  // Right iris (V2 only, indices 473-477)
  static constexpr std::array<int, 5> right_iris = {473, 474, 475, 476, 477};
};

// Process FaceMesh output
// MediaPipe FaceMesh outputs:
// - landmarks: [1, 1404] = 468 * 3 (x, y, z) for FaceMesh
// - landmarks: [1, 1434] = 478 * 3 (x, y, z) for FaceMesh V2
// - face_flag: [1, 1] confidence score
inline bool processOutput(
    const ModelSpec& spec,
    std::span<Ort::Value> outputs,
    int num_landmarks,
    std::optional<FaceMeshResult>& result)
{
  result.reset();

  if(outputs.size() < 1)
    return false;

  // Determine number of landmarks from output size
  auto shape = outputs[0].GetTensorTypeAndShapeInfo();
  int total_elements = shape.GetElementCount();

  int detected_landmarks = total_elements / 3;
  if(detected_landmarks != NUM_LANDMARKS && detected_landmarks != NUM_LANDMARKS_V2)
  {
    // Try to use provided num_landmarks
    if(num_landmarks * 3 != total_elements)
      return false;
    detected_landmarks = num_landmarks;
  }

  const float* landmark_data = outputs[0].GetTensorData<float>();

  // Get face flag if available (second output)
  float face_flag = 1.0f;
  if(outputs.size() >= 2)
  {
    face_flag = outputs[1].GetTensorData<float>()[0];
    // Apply sigmoid if raw logit
    if(face_flag < -10.0f || face_flag > 10.0f)
      face_flag = 1.0f / (1.0f + std::exp(-face_flag));
  }

  if(face_flag < 0.5f)
  {
    result.reset();
    return false;
  }

  std::vector<Landmark> landmarks(detected_landmarks);
  for(int i = 0; i < detected_landmarks; ++i)
  {
    // Coordinates are already normalized to [0, 1] in MediaPipe models
    landmarks[i].x = landmark_data[i * 3];
    landmarks[i].y = landmark_data[i * 3 + 1];
    landmarks[i].z = landmark_data[i * 3 + 2];
  }

  result = FaceMeshResult{.landmarks = std::move(landmarks), .face_flag = face_flag};

  return true;
}

} // namespace Onnx::FaceMesh
