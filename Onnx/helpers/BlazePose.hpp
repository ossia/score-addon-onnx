#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <vector>

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

  struct Point {
    float x;
    float y;
  };

  struct Detection {
    float xmin, ymin, width, height;
    float score;
    std::vector<Point> keypoints;
  };

  struct pose_align
  {
    std::vector<Detection> detections;
    
  };

  // Non-Maximum Suppression modified from helpers/BlazeFace.hpp
  static std::vector<Detection> 
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
      float x1 = std::max(a.xmin, b.xmin);
      float y1 = std::max(a.ymin, b.ymin);
      float x2 = std::min(a.xmin + a.width, b.xmin + b.width);
      float y2 = std::min(a.ymin + a.height, b.ymin + b.height);

      float inter_w = std::max(0.0f, x2 - x1);
      float inter_h = std::max(0.0f, y2 - y1);
      float inter_area = inter_w * inter_h;

      float area_a = a.width * a.height;
      float area_b = b.width * b.height;
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

  static bool processOutput(
      const Onnx::ModelSpec& spec,
      std::span<Ort::Value> outputTensor,
      std::optional<pose_align>& out,
      int iw, int ih)
      { 
        out.emplace();
        if (outputTensor.size() == 0)
          return false;
        auto& detections = out->detections;
        const float* raw_coords = outputTensor[0].GetTensorData<float>();

        std::vector<Detection> candidate_detections;
        for (int i = 0; i<kNumAnchors; i++)
        {
          if (Onnx::sigmoid(outputTensor[1].GetTensorData<float>()[i]) < 0.5f) // todo: infer from min_confidence input
            continue;

          const float* offset = &raw_coords[i*12];
          const auto& anchor = anchors[i];

          // Box Decoding
          float cx = (offset[0] / scalef) + anchor.x;
          float cy = (offset[1] / scalef) + anchor.y;
          float w = offset[2] / scalef;
          float h = offset[3] / scalef;

          Detection d;
          d.score = Onnx::sigmoid(outputTensor[1].GetTensorData<float>()[i]);
          d.width = w * static_cast<float>(iw);
          d.height = h * static_cast<float>(ih);
          d.xmin = (cx - w / 2.0f) * static_cast<float>(iw);
          d.ymin = (cy - h / 2.0f) * static_cast<float>(ih);

          // Keypoint Decoding
          for (int k = 0; k < 4; ++k) {
              float kx = ((offset[4 + k * 2] / scalef) + anchor.x) * static_cast<float>(iw);
              float ky = ((offset[4 + k * 2 + 1] / scalef) + anchor.y) * static_cast<float>(ih);
              d.keypoints.push_back({kx, ky});
          }

          candidate_detections.push_back(d);
        }

        detections = nms(candidate_detections);

        if (detections.empty())
        {          
          out.reset();
          return false;
        }

        return true;
      }

  static QTransform getROITransform(const Detection& det, int target_size = 256)
  {
    auto hip_center = det.keypoints[0];
    auto shoulder_center = det.keypoints[2];
    auto scale_center = det.keypoints[3];

    // calculate center of ROI
    float cx = (hip_center.x + shoulder_center.x) / 2.0f;
    float cy = (hip_center.y + shoulder_center.y) / 2.0f;

    // calculate rotation angle
    float dx = shoulder_center.x - hip_center.x;
    float dy = shoulder_center.y - hip_center.y;
    float angle = qRadiansToDegrees(std::atan2(dx, -dy));

    // determine crop size and scale
    float dist_to_scale = std::sqrt(std::pow(hip_center.x - scale_center.x, 2) + 
                                    std::pow(hip_center.y - scale_center.y, 2));

    float crop_size = dist_to_scale * 2.0f; // Assume Crop size is twice the distance from hip to scale center
    float scale = static_cast<float>(target_size) / crop_size;

    // Create the transformation matrix
    QTransform M;
    M.translate(target_size / 2.0, target_size / 2.0); // Move to center of output
    M.rotate(-angle); // Rotate by calculated angle
    M.scale(scale, scale); // Scale to fit target size
    M.translate(-cx, -cy); // Move the center of the person to the origin
    return M;
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

  static pose_data transformToOriginalImage(std::optional<pose_data>& roiPose, const QTransform& M)
  {
    pose_data originalPose;
    // get the inverse matrix
    QTransform M_inv = M.inverted();

    // 2. Map each keypoint
    for (int i = 0; i < NUM_KPS; ++i) {
        const keypoint& kp = roiPose->keypoints[i];
        
        // // Skip projecting if the point wasn't detected/confident (optional)
        // if (kp.confidence() < 0.1) {
        //     originalPose.keypoints[i] = kp;
        //     continue;
        // }

        // Use QTransform::map to transform the X and Y coordinates
        qreal ox, oy;
        M_inv.map(static_cast<qreal>(kp.x), static_cast<qreal>(kp.y), &ox, &oy);

        // 3. Assign back to the new structure
        originalPose.keypoints[i].x = static_cast<float>(ox);
        originalPose.keypoints[i].y = static_cast<float>(oy);
        
        // Z, visibility, and presence are preserved as-is
        originalPose.keypoints[i].z = kp.z;
        originalPose.keypoints[i].visibility = kp.visibility;
        originalPose.keypoints[i].presence = kp.presence;
    }

    return originalPose;
  }
};
}