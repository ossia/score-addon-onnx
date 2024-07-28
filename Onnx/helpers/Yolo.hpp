#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>

namespace OnnxModels
{

struct YOLO_blob
{
  QList<QByteArray> classes;

  void loadClasses(QByteArray class_data) { classes = class_data.split('\n'); }

  struct blob_type
  {
    std::string name;
    struct
    {
      float x, y, w, h;
    } geometry;
    float confidence{};
  };

  void processOutput(
      const Onnx::ModelSpec& spec,
      std::span<Ort::Value> outputTensor,
      std::vector<blob_type>& out,
      int image_x = 0,
      int image_y = 0,
      int image_w = 640,
      int image_h = 640,
      int model_w = 640,
      int model_h = 640) const
  {
    out.clear();
    if (outputTensor.size() > 0)
    {
      // This model provides the elements in successive groups of 7 floats:
      // confidence, x, y, w, h, class, accuracy.
      const int Nfloats
          = outputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount();
      const int Nblobs = Nfloats / 7;
      const float* arr = outputTensor.front().GetTensorData<float>();
      for (int i = 0; i < Nblobs; i += 7)
      {
        const int class_type = static_cast<int>(arr[i + 5]);
        if (class_type < 0)
          continue;
        const float confidence = (arr[i + 0]);
        const float accuracy = (arr[i + 6]);
        const int original_cols = image_w;
        const int original_rows = image_h;

        // clang-format off
        const float x = arr[i + 1] / (float)model_w * original_cols;
        const float y = arr[i + 2] / (float)model_h * original_rows;
        const float w = (arr[i + 3] - arr[i + 1]) / (float)model_w * (float)original_cols;
        const float h = (arr[i + 4] - arr[i + 2]) / (float)model_h * (float)original_rows;
        // clang-format on

        std::string class_name = class_type < classes.size()
                                     ? classes[class_type].toStdString()
                                     : "unclassified";
        out.push_back(blob_type{
            .name = class_name,
            .geometry = {.x = x, .y = y, .w = w, .h = h},
            .confidence = confidence});
      }
    }
  }
};

struct YOLO_pose
{
  static constexpr int NUM_KPS = 17;
  QList<QByteArray> classes;

  void loadClasses(QByteArray class_data) { classes = class_data.split('\n'); }

  struct pose_data
  {
    struct rect
    {
      float x, y, w, h;
    } geometry;
    float confidence{};
    float keypoints[NUM_KPS][3]{};
  };

  struct pose_type
  {
    std::string name;
    struct rect
    {
      float x, y, w, h;
    } geometry;
    float confidence{};
    struct pos
    {
      int kp;
      float x, y;
    };
    std::vector<pos> keypoints;
  };

  static constexpr bool
  similar(pose_type::rect lhs, pose_type::rect rhs) noexcept
  {
    const float epsilon = std::min(0.1 * lhs.w, 0.1 * lhs.h);
    return std::abs(rhs.x - lhs.x) < epsilon
           && std::abs(rhs.y - lhs.y) < epsilon
           && std::abs(rhs.w - lhs.w) < epsilon
           && std::abs(rhs.h - lhs.h) < epsilon;
  }

  static constexpr bool
  hasSimilarRect(pose_type::rect r, std::span<pose_type> poses) noexcept
  {
    for (const auto& p : poses)
    {
      if (similar(r, p.geometry))
        return true;
    }
    return false;
  }
  void processOutput(
      const Onnx::ModelSpec& spec,
      std::span<Ort::Value> outputTensor,
      std::vector<pose_type>& out,
      int max_detect = 100,
      float min_confidence = 0.75,
      int image_x = 0,
      int image_y = 0,
      int image_w = 640,
      int image_h = 640,
      int model_w = 640,
      int model_h = 640) const
  {
    const int src_cols = image_w;
    const int src_rows = image_h;
    out.clear();
    if (outputTensor.size() > 0)
    {
      const int Nfloats
          = outputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount();
      if (Nfloats == 56 * 8400)
      {
        // 1. Grab the pose indiceswith global confidence > minimum
        const float* data = outputTensor.front().GetTensorData<float>();
        const float* recog = data + 4 * 8400;
        thread_local std::vector<int> idx;
        idx.clear();
#pragma omp simd
        for (int i = 0; i < 8400; i++)
        {
          if (recog[i] > min_confidence)
            idx.push_back(i);
        }

        // 2. Sort the resulting array. First element will be index of pose with highest confidence.
        std::stable_sort(
            idx.begin(),
            idx.end(),
            [&](int i1, int i2) { return recog[i1] > recog[i2]; });

        // 3. Add the pose to the list and process the keypoints
        for (auto j : idx)
        {
          pose_data p;
#pragma omp simd
          for (int k = 0; k < 56; k++)
          {
            reinterpret_cast<float*>(&p)[k] = (data + k * 8400)[j];
          }
          auto rect = pose_type::rect{
              p.geometry.x - p.geometry.w / 2,
              p.geometry.y - p.geometry.h / 2,
              p.geometry.w,
              p.geometry.h};

          // Filter out rects we already added
          if (hasSimilarRect(rect, out))
            continue;

          out.push_back(
              pose_type{.geometry = rect, .confidence = p.confidence});
          auto& kps = out.back().keypoints;
          for (int i = 0; i < NUM_KPS; i++)
          {
            const auto& [x, y, c] = p.keypoints[i];
            if (c > min_confidence)
            {
              kps.push_back({i, x, y});
            }
          }
        }
      }
    }
  }
};
}
