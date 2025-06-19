#pragma once
#include <ossia/detail/pod_vector.hpp>

#include <boost/dynamic_bitset.hpp>

#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>

namespace OnnxModels::Yolo
{

// intersection-over-union for blobs
static float iou(const auto& box1, const auto& box2)
{
  float x1 = std::max(box1.x, box2.x);
  float y1 = std::max(box1.y, box2.y);
  float x2 = std::min(box1.x + box1.w, box2.x + box2.w);
  float y2 = std::min(box1.y + box1.h, box2.y + box2.h);

  float intersection_w = std::max(0.0f, x2 - x1);
  float intersection_h = std::max(0.0f, y2 - y1);
  float intersection_area = intersection_w * intersection_h;

  float box1_area = box1.w * box1.h;
  float box2_area = box2.w * box2.h;
  float union_area = box1_area + box2_area - intersection_area;

  return union_area > 1e-6 ? intersection_area / union_area : 0.0f;
}

// non-maximum supression
static void nms(auto& candidates, float threshold)
{
  using type = std::remove_cvref_t<decltype(candidates)>::value_type;
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const type& a, const type& b)
      { return a.confidence > b.confidence; });

  for (size_t i = 0; i < candidates.size(); ++i)
  {
    if (candidates[i].suppressed)
      continue;

    for (size_t j = i + 1; j < candidates.size(); ++j)
    {
      if (candidates[j].suppressed)
        continue;

      if (iou(candidates[i], candidates[j]) > threshold)
        candidates[j].suppressed = true;
    }
  }
}

struct YOLO_blob
{
  struct blob_type
  {
    std::string name;
    struct
    {
      float x, y, w, h;
    } geometry;
    float confidence{};
  };

  struct fast_blob_type
  {
    float x, y, w, h;
    float confidence;
    int class_id;
    bool suppressed;
  };
  std::vector<fast_blob_type> candidates;

  void processOutput(
      std::span<std::string> classes,
      const Onnx::ModelSpec& spec,
      std::span<Ort::Value> outputTensor,
      std::vector<blob_type>& out,
      int image_x,
      int image_y,
      int image_w,
      int image_h,
      int model_w,
      int model_h,
      float iou_threshold,
      float confidence_threshold)
  {
    out.clear();
    if (outputTensor.size() > 0)
    {
      // This model provides the elements in successive groups of 7 floats:
      // confidence, x, y, w, h, class, accuracy.
      const int Nfloats
          = outputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount();

      if (Nfloats == (classes.size() + 4) * 8400)
      {
        processOutput_v8(
            classes,
            spec,
            outputTensor,
            out,
            image_x, // 0
            image_y, // 0
            image_w, // 640
            image_h, // 640
            model_w, // 640
            model_h, // 640
            Nfloats,
            iou_threshold,
            confidence_threshold);
      }
      else
      {
        processOutput_v7(
            classes,
            spec,
            outputTensor,
            out,
            image_x,
            image_y,
            image_w,
            image_h,
            model_w,
            model_h,
            Nfloats,
            confidence_threshold);
      }
    }
  }

  // Note: made for https://github.com/WongKinYiu/yolov7
  // This model provides the elements in successive groups of 7 floats:
  // confidence, x, y, w, h, class, accuracy.
  void processOutput_v7(
      std::span<std::string> classes,
      const Onnx::ModelSpec& spec,
      std::span<Ort::Value> outputTensor,
      std::vector<blob_type>& out,
      int image_x,
      int image_y,
      int image_w,
      int image_h,
      int model_w,
      int model_h,
      int Nfloats,
      float confidence_threshold)
  {
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
      const float x =  arr[i + 1]               / (float)model_w * (float)original_cols;
      const float y =  arr[i + 2]               / (float)model_h * (float)original_rows;
      const float w = (arr[i + 3] - arr[i + 1]) / (float)model_w * (float)original_cols;
      const float h = (arr[i + 4] - arr[i + 2]) / (float)model_h * (float)original_rows;
      // clang-format on

      if (confidence > confidence_threshold)
      {
        if (class_type >= 0 && class_type < std::ssize(classes))
          out.push_back(
              blob_type{
                  .name = classes[class_type],
                  .geometry = {.x = x, .y = y, .w = w, .h = h},
                  .confidence = confidence});
      }
    }
  }

  // Implementation for YOLOv8..12
  void processOutput_v8(
      std::span<std::string> classes,
      const Onnx::ModelSpec& spec,
      std::span<Ort::Value> outputTensor,
      std::vector<blob_type>& out,
      int image_x,
      int image_y,
      int image_w,
      int image_h,
      int model_w,
      int model_h,
      int Nfloats,
      float iou_threshold,
      float confidence_threshold)
  {
    const float* data = outputTensor.front().GetTensorData<float>();

    const int num_proposals = 8400;
    const int num_classes
        = classes.size(); // Default weight use 80 COCO classes

    if (num_classes != classes.size())
      return;

    const float x_factor = (float)image_w / model_w;
    const float y_factor = (float)image_h / model_h;

    candidates.clear();
    candidates.reserve(num_proposals / 10);

    const float* x_data = data;
    const float* y_data = data + num_proposals;
    const float* w_data = data + 2 * num_proposals;
    const float* h_data = data + 3 * num_proposals;
    const float* class_data_start = data + 4 * num_proposals;

    for (int i = 0; i < num_proposals; ++i)
    {
      // Find the class with the highest score for the current proposal.
      float max_class_score = 0.0f;
      int best_class_id = -1;

      for (int j = 0; j < num_classes; ++j)
      {
        // Access score for class j, proposal i.
        // The scores are stored class by class: [class0_p0, class0_p1, ..., class1_p0, ...].
        const float score = class_data_start[j * num_proposals + i];
        if (score > max_class_score)
        {
          max_class_score = score;
          best_class_id = j;
        }
      }

      if (max_class_score > confidence_threshold)
      {
        const float cx = x_data[i];
        const float cy = y_data[i];
        const float w = w_data[i];
        const float h = h_data[i];

        // Convert from center-based to top-left corner based
        candidates.push_back(
            {(cx - w / 2.0f) * x_factor,
             (cy - h / 2.0f) * y_factor,
             w * x_factor,
             h * y_factor,
             max_class_score,
             best_class_id,
             false});
      }
    }

    if (candidates.empty())
      return;

    nms(candidates, iou_threshold);

    // Populate the final output vector with the non-suppressed boxes.
    for (const auto& box : candidates)
    {
      if (!box.suppressed && box.class_id >= 0
          && box.class_id < std::ssize(classes))
      {
        out.push_back(
            {classes[box.class_id],
             {box.x + image_x, box.y + image_y, box.w, box.h},
             box.confidence});
      }
    }
  }
};

struct YOLO_pose
{
  static constexpr int NUM_KPS = 17;
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

  // FIXME use NMS instead
  static bool similar(pose_type::rect lhs, pose_type::rect rhs) noexcept
  {
    const float epsilon = std::min(0.1 * lhs.w, 0.1 * lhs.h);
    return std::abs(rhs.x - lhs.x) < epsilon
           && std::abs(rhs.y - lhs.y) < epsilon
           && std::abs(rhs.w - lhs.w) < epsilon
           && std::abs(rhs.h - lhs.h) < epsilon;
  }

  static bool
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
        thread_local ossia::pod_vector<int> idx;
        idx.clear();
        idx.resize(8400, boost::container::default_init);

        int k = 0;
#pragma omp simd
        for (int i = 0; i < 8400; i++)
        {
          if (recog[i] > min_confidence)
            idx[k++] = i;
        }
        if (k == 0)
          return;
        idx.resize(k);

        // 2. Sort the resulting array. First element will be index of pose with highest confidence.
        std::stable_sort(
            idx.data(),
            idx.data() + k,
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

struct YOLO_segmentation
{
  // Holds the results for a single segmented object
  struct segmentation_type
  {
    std::string name;
    struct
    {
      float x, y, w, h;
    } geometry;
    float confidence{};
    // The final binary mask, with dimensions of the original image.
    boost::dynamic_bitset<> mask;
  };

  // Main processing function for segmentation model outputs
  static void processOutput(
      std::span<std::string> classes,
      std::span<Ort::Value> outputTensors,
      std::vector<segmentation_type>& out,
      int image_w,
      int image_h,
      int model_w,
      int model_h,
      float iou_threshold = 0.45f,
      float conf_threshold = 0.25f)
  {
    out.clear();
    // Segmentation models have two outputs: predictions and prototypes
    if (outputTensors.size() != 2)
      return;

    // 1. Get Tensors and their shapes
    auto& predictions_tensor
        = outputTensors[0]; // [1, 4+num_classes+num_masks, 8400]
    auto& protos_tensor
        = outputTensors[1]; // [1, num_masks, mask_height, mask_width]

    auto pred_shape
        = predictions_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const int num_proposals = pred_shape[2]; // 8400
    const int num_channels = pred_shape[1];
    const int num_classes = classes.size();
    const int num_mask_coeffs = num_channels - 4 - num_classes; // should be 32

    auto protos_shape = protos_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const int mask_height = protos_shape[2];
    const int mask_width = protos_shape[3];

    const float* pred_data = predictions_tensor.GetTensorData<float>();
    const float* protos_data = protos_tensor.GetTensorData<float>();

    // 2. Decode predictions and filter by confidence
    struct Candidate
    {
      int class_id;
      float confidence;
      float x, y, w, h; // Bounding box
      const float* mask_coeffs;
      bool suppressed{};
    };
    std::vector<Candidate> candidates;

    const float x_factor = (float)image_w / model_w;
    const float y_factor = (float)image_h / model_h;

    for (int i = 0; i < num_proposals; ++i)
    {
      // Find class with max score
      float max_score = 0;
      int class_id = -1;
      const float* class_scores_start = pred_data + (4 * num_proposals) + i;
      for (int j = 0; j < num_classes; ++j)
      {
        float score = class_scores_start[j * num_proposals];
        if (score > max_score)
        {
          max_score = score;
          class_id = j;
        }
      }

      if (max_score > conf_threshold)
      {
        const float* box_start = pred_data + i;
        float cx = box_start[0 * num_proposals];
        float cy = box_start[1 * num_proposals];
        float w = box_start[2 * num_proposals];
        float h = box_start[3 * num_proposals];

        if (w * h >= (0.9 * image_w * image_h))
          continue;

        candidates.push_back(
            {class_id,
             max_score,
             (cx - w / 2.f) * x_factor,
             (cy - h / 2.f) * y_factor,
             w * x_factor,
             h * y_factor,
             pred_data + ((4 + num_classes) * num_proposals) + i,
             false});
      }
    }

    nms(candidates, iou_threshold);

    for (size_t i = 0; i < candidates.size(); ++i)
    {
      if (candidates[i].suppressed)
        continue;
      const auto& det = candidates[i];

      boost::dynamic_bitset<> low_res_mask(mask_width * mask_height);
      // Matrix multiplication: mask_coeffs [1, 32] x protos [32, 160*160]
      for (int p_idx = 0; p_idx < mask_width * mask_height; ++p_idx)
      {
        float sum = 0;
        for (int c_idx = 0; c_idx < num_mask_coeffs; ++c_idx)
        {
          sum += det.mask_coeffs[c_idx * num_proposals]
                 * protos_data[c_idx * (mask_width * mask_height) + p_idx];
        }
        // FIXME choose this threshold?
        if ((1.f / (1.f + std::exp(-sum))) > 0.5f)
          low_res_mask.set(p_idx);
      }

      // Scale mask to original image size
      float gain = std::min(
          (float)mask_width / image_w, (float)mask_height / image_h);
      float pad_w = ((float)mask_width - image_w * gain) / 2.f;
      float pad_h = ((float)mask_height - image_h * gain) / 2.f;

      int crop_x1 = std::round(pad_w);
      int crop_y1 = std::round(pad_h);
      int crop_x2 = mask_width - std::round(pad_w);
      int crop_y2 = mask_height - std::round(pad_h);

      boost::dynamic_bitset<> final_mask(image_w * image_h);
      for (int y = 0; y < image_h; ++y)
      {
        for (int x = 0; x < image_w; ++x)
        {
          // Map final mask coordinate to low_res_mask coordinate
          float low_res_x
              = (x * (crop_x2 - crop_x1) / (float)image_w) + crop_x1;
          float low_res_y
              = (y * (crop_y2 - crop_y1) / (float)image_h) + crop_y1;
          int m_x = std::clamp((int)low_res_x, 0, mask_width - 1);
          int m_y = std::clamp((int)low_res_y, 0, mask_height - 1);

          if (low_res_mask[m_y * mask_width + m_x])
          {
            bool in_box
                = (x >= det.x && x < (det.x + det.w) && y >= det.y
                   && y < (det.y + det.h));
            if (in_box)
              final_mask.set(y * image_w + x);
          }
        }
      }

      out.push_back(
          {classes[det.class_id],
           {det.x, det.y, det.w, det.h},
           det.confidence,
           std::move(final_mask)});
    }
  }
};
}
