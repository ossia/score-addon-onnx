#include "TRTPose.hpp"

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Trt.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <cstdio>

namespace OnnxModels::TRTPose
{
TRTPoseDetector::TRTPoseDetector() noexcept
{
  inputs.resolution.value = {512, 512};
}
TRTPoseDetector::~TRTPoseDetector() = default;

void TRTPoseDetector::operator()()
try
{
#if 0
  qDebug("============ BEGIN FRAME");
#endif
  if (!available)
    return;
  if (inputs.model.current_model_invalid)
    return;

  auto& in_tex = inputs.image.texture;

  if (!in_tex.changed)
    return;

  if (!this->ctx)
  {
    this->ctx = std::make_unique<Onnx::OnnxRunContext>(
        this->inputs.model.file.bytes);
    spec = ctx->readModelSpec();
  }

  auto& ctx = *this->ctx;

  // Get model input resolution
  int input_width = inputs.resolution.value.x;
  int input_height = inputs.resolution.value.y;

  // Check if we can read actual dimensions from model
  if (!spec.inputs.empty() && spec.inputs[0].shape.size() >= 4)
  {
    if (spec.inputs[0].shape[3] > 0)
      input_width = static_cast<int>(spec.inputs[0].shape[3]);
    if (spec.inputs[0].shape[2] > 0)
      input_height = static_cast<int>(spec.inputs[0].shape[2]);
  }

  // Build the input tensor with the shared Qt-free helper (bilinear fill+crop
  // resize to the model resolution, RGBA->RGB pack, NCHW with ImageNet
  // normalization). Replaces the former QImage scale/centre-pad path.
  auto t = Onnx::nchw_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      input_width,
      input_height,
      storage,
      {0.485f * 255.0f,
       0.456f * 255.0f,
       0.406f * 255.0f}, // ImageNet mean scaled to [0,255]
      {0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f}
      // ImageNet std scaled to [0,255]
  );

  Ort::Value input_tensors[1] = {std::move(t.value)};

  // TRT-Pose has 2 outputs: confidence maps [1,18,H,W] and PAF [1,42,H,W]
  if (spec.output_names_char.size() != 2)
  {
    std::fprintf(
        stderr, "ERROR: Model should have 2 outputs, has: %zu\n",
        spec.output_names_char.size());
    return;
  }

  Ort::Value output_tensors[2]{Ort::Value{nullptr}, Ort::Value{nullptr}};

  ctx.infer(spec, input_tensors, output_tensors);

  // Extract output data
  auto cmap_info = output_tensors[0].GetTensorTypeAndShapeInfo();
  auto paf_info = output_tensors[1].GetTensorTypeAndShapeInfo();

  auto cmap_shape = cmap_info.GetShape();
  auto paf_shape = paf_info.GetShape();

  const float* cmap_data = output_tensors[0].GetTensorData<float>();
  const float* paf_data = output_tensors[1].GetTensorData<float>();

  // Extract dimensions (assuming [1, channels, height, width])
  int output_height = static_cast<int>(cmap_shape[2]);
  int output_width = static_cast<int>(cmap_shape[3]);

  // Debug: analyze confidence maps
  float max_confidence = -std::numeric_limits<float>::max();
  float min_confidence = std::numeric_limits<float>::max();
  size_t total_pixels = 18 * output_height * output_width;

  for (size_t i = 0; i < total_pixels; i++)
  {
    max_confidence = std::max(max_confidence, cmap_data[i]);
    min_confidence = std::min(min_confidence, cmap_data[i]);
  }

  // Log confidence range for debugging
  // qDebug() << "TRT-Pose confidence range: [" << min_confidence << ", "
  //          << max_confidence << "]";

  // Process pose detection with user-controlled threshold
  Onnx::TRT_pose::Config config;
  config.confidence_threshold = inputs.confidence_threshold.value;
  config.paf_threshold
      = inputs.paf_threshold.value; // PAF threshold as fraction of confidence
  config.max_peaks_per_part = 10;   // Reasonable limit to reduce noise
  config.peak_window_size = 5;

  // qDebug() << "Using confidence threshold:" << config.confidence_threshold << "PAF threshold:" << config.paf_threshold;

  Onnx::TRT_pose pose_processor(config);
  
  // Get raw peaks for debugging before full processing
  auto debug_peaks = pose_processor.findPeaks(cmap_data, output_height, output_width);
  pose_processor.refinePeaks(
      debug_peaks, cmap_data, output_height, output_width);

  // Get PAF scores for debugging (back to original method with fixes)
  auto debug_paf_scores = pose_processor.scorePAF(debug_peaks, paf_data, output_height, output_width);

  auto detected_persons = pose_processor.processOutput(
      cmap_data, paf_data, output_height, output_width);

  // qDebug() << "Detected persons:" << detected_persons.size();

  // Convert to ossia format
  outputs.detection.value.clear();
  for (const auto& person : detected_persons)
  {
    DetectedTRTPose detected_pose;
    detected_pose.name = "person";
    detected_pose.probability = person.total_score;

    // Calculate bounding box from keypoints
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    bool has_valid_keypoints = false;
    for (int i = 0; i < 18; i++)
    {
      const auto& kp = person.keypoints[i];
      if (kp.confidence > config.confidence_threshold)
      {
        // Convert from model coordinates to image coordinates
        float img_x = (kp.x / output_width) * in_tex.width;
        float img_y = (kp.y / output_height) * in_tex.height;

        min_x = std::min(min_x, img_x);
        max_x = std::max(max_x, img_x);
        min_y = std::min(min_y, img_y);
        max_y = std::max(max_y, img_y);

        detected_pose.keypoints.push_back(Keypoint{
            .kp = i,
            .x = img_x / in_tex.width, // Normalize to [0,1]
            .y = img_y / in_tex.height // Normalize to [0,1]
        });
        has_valid_keypoints = true;
      }
    }

    if (has_valid_keypoints)
    {
      // Set bounding box (normalized coordinates)
      detected_pose.geometry.x = min_x / in_tex.width;
      detected_pose.geometry.y = min_y / in_tex.height;
      detected_pose.geometry.w = (max_x - min_x) / in_tex.width;
      detected_pose.geometry.h = (max_y - min_y) / in_tex.height;

      outputs.detection.value.push_back(std::move(detected_pose));
    }
  }

  // Add pose skeleton if persons detected. The pose overlay is an RGBA8888
  // image (black background); composite it additively (Qt's CompositionMode_Plus)
  // onto the input frame, then write the result to the output texture (opaque).
  auto pose_overlay = pose_processor.visualizePoses(
      detected_persons,
      in_tex.width,
      in_tex.height,
      output_width,
      output_height);

  const unsigned char* base = reinterpret_cast<const unsigned char*>(in_tex.bytes);
  const unsigned char* ov = pose_overlay.pixels.data();
  outputs.image.create(in_tex.width, in_tex.height);
  const int N = in_tex.width * in_tex.height * 4;
  for (int i = 0; i < N; i += 4)
  {
    for (int c = 0; c < 3; ++c)
      outputs.image.texture.bytes[i + c]
          = static_cast<unsigned char>(std::min(255, base[i + c] + ov[i + c]));
    outputs.image.texture.bytes[i + 3] = 255;
  }
  outputs.image.texture.changed = true;

  std::swap(storage, t.storage);
}
catch (...)
{
  inputs.model.current_model_invalid = true;
}
}
