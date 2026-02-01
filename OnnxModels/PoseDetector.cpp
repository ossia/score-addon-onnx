#include "PoseDetector.hpp"

#include <QImage>
#include <QPainter>

#include <Onnx/helpers/BlazeFace.hpp>
#include <Onnx/helpers/BlazePose.hpp>
#include <Onnx/helpers/FaceMesh.hpp>
#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/MediaPipeHands.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/RTMPose.hpp>
#include <Onnx/helpers/Yolo.hpp>
#include <cmath>

#include <algorithm>
#include <array>

namespace OnnxModels
{

// Helper to transform model-space coordinates back to input image space
// Accounts for the KeepAspectRatioByExpanding + center crop preprocessing
struct CoordTransform
{
  float crop_x{0}, crop_y{0};   // Crop offset in scaled space
  float scaled_w{1}, scaled_h{1}; // Size after scaling (before crop)
  float model_w{1}, model_h{1};   // Model input size

  // Initialize transform from source and model dimensions
  void init(int source_w, int source_h, int model_w_, int model_h_)
  {
    model_w = static_cast<float>(model_w_);
    model_h = static_cast<float>(model_h_);

    float source_aspect = static_cast<float>(source_w) / source_h;
    float model_aspect = model_w / model_h;

    if(source_aspect > model_aspect)
    {
      // Source is wider - scale based on height, crop width
      float scale = model_h / source_h;
      scaled_w = source_w * scale;
      scaled_h = model_h;
      crop_x = (scaled_w - model_w) / 2.0f;
      crop_y = 0;
    }
    else
    {
      // Source is taller - scale based on width, crop height
      float scale = model_w / source_w;
      scaled_w = model_w;
      scaled_h = source_h * scale;
      crop_x = 0;
      crop_y = (scaled_h - model_h) / 2.0f;
    }
  }

  // Transform normalized [0,1] model coordinates to normalized [0,1] source coordinates
  void modelToSource(float mx, float my, float& sx, float& sy) const
  {
    // Model normalized -> model pixels
    float mx_pix = mx * model_w;
    float my_pix = my * model_h;

    // Add crop offset to get to scaled source space
    float scaled_x = mx_pix + crop_x;
    float scaled_y = my_pix + crop_y;

    // Normalize by scaled source size
    sx = scaled_x / scaled_w;
    sy = scaled_y / scaled_h;
  }

  // Transform pixel coordinates in model space to normalized [0,1] source coordinates
  void modelPixelsToSource(float mx_pix, float my_pix, float& sx, float& sy) const
  {
    // Add crop offset to get to scaled source space
    float scaled_x = mx_pix + crop_x;
    float scaled_y = my_pix + crop_y;

    // Normalize by scaled source size
    sx = scaled_x / scaled_w;
    sy = scaled_y / scaled_h;
  }
};

// Skeleton connection definitions
namespace Skeletons
{
// clang-format off
// BlazePose connections (33 keypoints)
static constexpr std::array<std::pair<int, int>, 35> blazepose = {{
    // Face
    {0, 1}, {1, 2}, {2, 3}, {3, 7},     // left eye
    {0, 4}, {4, 5}, {5, 6}, {6, 8},     // right eye
    {9, 10},                             // mouth
    // Torso
    {11, 12}, {11, 23}, {12, 24}, {23, 24},
    // Left arm
    {11, 13}, {13, 15}, {15, 17}, {15, 19}, {15, 21}, {17, 19},
    // Right arm
    {12, 14}, {14, 16}, {16, 18}, {16, 20}, {16, 22}, {18, 20},
    // Left leg
    {23, 25}, {25, 27}, {27, 29}, {27, 31}, {29, 31},
    // Right leg
    {24, 26}, {26, 28}, {28, 30}, {28, 32}, {30, 32},
}};

// COCO 17 connections (for RTMPose COCO and ViTPose)
static constexpr std::array<std::pair<int, int>, 19> coco17 = {{
    // Head
    {0, 1}, {0, 2}, {1, 3}, {2, 4},
    // Torso
    {5, 6}, {5, 11}, {6, 12}, {11, 12},
    // Left arm
    {5, 7}, {7, 9},
    // Right arm
    {6, 8}, {8, 10},
    // Left leg
    {11, 13}, {13, 15},
    // Right leg
    {12, 14}, {14, 16},
    // Neck approximation
    {0, 5}, {0, 6}, {5, 6},
}};

// WholeBody partial connections (body part only for 133 keypoints)
// Body: 0-16, Feet: 17-22, Face: 23-90, Left Hand: 91-111, Right Hand: 112-132
static constexpr std::array<std::pair<int, int>, 19> wholebody_body = {{
    // Same as COCO for the first 17 keypoints
    {0, 1}, {0, 2}, {1, 3}, {2, 4},
    {5, 6}, {5, 11}, {6, 12}, {11, 12},
    {5, 7}, {7, 9},
    {6, 8}, {8, 10},
    {11, 13}, {13, 15},
    {12, 14}, {14, 16},
    {0, 5}, {0, 6}, {5, 6},
}};

// Hand connections (21 keypoints)
// 0:wrist, 1-4:thumb, 5-8:index, 9-12:middle, 13-16:ring, 17-20:pinky
static constexpr std::array<std::pair<int, int>, 20> hands = {{
    // Thumb
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    // Index
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    // Middle
    {0, 9}, {9, 10}, {10, 11}, {11, 12},
    // Ring
    {0, 13}, {13, 14}, {14, 15}, {15, 16},
    // Pinky
    {0, 17}, {17, 18}, {18, 19}, {19, 20},
    // Palm
    //{5, 9},
}};

// BlazeFace connections (6 keypoints: right_eye, left_eye, nose, mouth, right_ear, left_ear)
static constexpr std::array<std::pair<int, int>, 5> blazeface = {{
    {0, 2},  // right_eye to nose
    {1, 2},  // left_eye to nose
    {2, 3},  // nose to mouth
    {0, 4},  // right_eye to right_ear
    {1, 5},  // left_eye to left_ear
}};

// MobileFaceNet / dlib 68 landmarks connections
// 0-16: Jaw, 17-21: Left eyebrow, 22-26: Right eyebrow, 27-35: Nose
// 36-41: Left eye, 42-47: Right eye, 48-59: Outer lip, 60-67: Inner lip
static constexpr std::array<std::pair<int, int>, 67> dlib68 = {{
    // Jaw line (0-16)
    {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8},
    {8, 9}, {9, 10}, {10, 11}, {11, 12}, {12, 13}, {13, 14}, {14, 15}, {15, 16},
    // Left eyebrow (17-21)
    {17, 18}, {18, 19}, {19, 20}, {20, 21},
    // Right eyebrow (22-26)
    {22, 23}, {23, 24}, {24, 25}, {25, 26},
    // Nose bridge (27-30)
    {27, 28}, {28, 29}, {29, 30},
    // Nose bottom (31-35)
    {31, 32}, {32, 33}, {33, 34}, {34, 35},
    {30, 33},  // bridge to tip
    // Left eye (36-41)
    {36, 37}, {37, 38}, {38, 39}, {39, 40}, {40, 41}, {41, 36},
    // Right eye (42-47)
    {42, 43}, {43, 44}, {44, 45}, {45, 46}, {46, 47}, {47, 42},
    // Outer lip (48-59)
    {48, 49}, {49, 50}, {50, 51}, {51, 52}, {52, 53}, {53, 54},
    {54, 55}, {55, 56}, {56, 57}, {57, 58}, {58, 59}, {59, 48},
    // Inner lip (60-67)
    {60, 61}, {61, 62}, {62, 63}, {63, 64}, {64, 65}, {65, 66}, {66, 67}, {67, 60},
}};

// clang-format on
} // namespace Skeletons

// Color schemes for different body parts
namespace Colors
{
static const QColor head{255, 255, 0};      // Yellow
static const QColor torso{255, 255, 255};   // White
static const QColor left_arm{0, 255, 255};  // Cyan
static const QColor right_arm{255, 0, 255}; // Magenta
static const QColor left_leg{0, 255, 0};    // Green
static const QColor right_leg{255, 128, 0}; // Orange
static const QColor face{200, 200, 255};    // Light blue
static const QColor hand{255, 200, 100};    // Light orange
} // namespace Colors

PoseDetector::PoseDetector() noexcept
{
  // inputs.image.request_width = 256;
  // inputs.image.request_height = 256;
}

PoseDetector::~PoseDetector() = default;

// Get body part color based on keypoint index for COCO format
static QColor getCOCOColor(int idx)
{
  if(idx == 0)
    return Colors::head; // Nose
  else if(idx <= 4)
    return Colors::head; // Eyes/ears
  else if(idx == 5 || idx == 6)
    return Colors::torso; // Shoulders
  else if(idx == 7 || idx == 9)
    return Colors::left_arm;
  else if(idx == 8 || idx == 10)
    return Colors::right_arm;
  else if(idx == 11 || idx == 12)
    return Colors::torso; // Hips
  else if(idx == 13 || idx == 15)
    return Colors::left_leg;
  else if(idx == 14 || idx == 16)
    return Colors::right_leg;
  return Colors::torso;
}

// Get body part color for BlazePose
static QColor getBlazePoseColor(int idx)
{
  if(idx <= 10)
    return Colors::head; // Face landmarks
  else if(idx == 11 || idx == 12)
    return Colors::torso; // Shoulders
  else if(idx >= 13 && idx <= 22 && idx % 2 == 1)
    return Colors::left_arm;
  else if(idx >= 13 && idx <= 22 && idx % 2 == 0)
    return Colors::right_arm;
  else if(idx == 23 || idx == 24)
    return Colors::torso; // Hips
  else if(idx >= 25 && idx <= 32 && idx % 2 == 1)
    return Colors::left_leg;
  else if(idx >= 25 && idx <= 32 && idx % 2 == 0)
    return Colors::right_leg;
  return Colors::torso;
}

// Get body part color for WholeBody 133
static QColor getWholeBodyColor(int idx)
{
  if(idx <= 16)
    return getCOCOColor(idx); // Body keypoints same as COCO
  else if(idx <= 22)
    return Colors::left_leg.lighter(120); // Feet
  else if(idx <= 90)
    return Colors::face; // Face landmarks
  else if(idx <= 111)
    return Colors::left_arm.lighter(120); // Left hand
  else
    return Colors::right_arm.lighter(120); // Right hand
}

// Get finger color for hand keypoints
static QColor getHandColor(int idx)
{
  if (idx <= 4)
    return QColor(255, 100, 100); // Thumb - red
  else if (idx <= 8)
    return QColor(100, 255, 100); // Index - green
  else if (idx <= 12)
    return QColor(100, 100, 255); // Middle - blue
  else if (idx <= 16)
    return QColor(255, 255, 100); // Ring - yellow
  else
    return QColor(255, 100, 255); // Pinky - magenta
}

// Get color for BlazeFace keypoints
static QColor getBlazeFaceColor(int idx)
{
  switch(idx)
  {
    case 0: return QColor(255, 0, 0);    // right_eye - red
    case 1: return QColor(0, 0, 255);    // left_eye - blue
    case 2: return QColor(0, 255, 0);    // nose - green
    case 3: return QColor(255, 255, 0);  // mouth - yellow
    case 4: return QColor(255, 0, 255);  // right_ear - magenta
    case 5: return QColor(0, 255, 255);  // left_ear - cyan
    default: return Colors::face;
  }
}

// Get color for dlib 68 face landmarks
static QColor getDlib68Color(int idx)
{
  if(idx <= 16)
    return QColor(200, 200, 200);  // Jaw - gray
  else if(idx <= 21)
    return QColor(255, 200, 100);  // Left eyebrow - orange
  else if(idx <= 26)
    return QColor(255, 200, 100);  // Right eyebrow - orange
  else if(idx <= 35)
    return QColor(0, 255, 0);      // Nose - green
  else if(idx <= 41)
    return QColor(0, 255, 255);    // Left eye - cyan
  else if(idx <= 47)
    return QColor(0, 255, 255);    // Right eye - cyan
  else if(idx <= 59)
    return QColor(255, 100, 100);  // Outer lip - pink
  else
    return QColor(255, 50, 50);    // Inner lip - red
}

void PoseDetector::drawSkeleton(const DetectedPose& pose, PoseWorkflow workflow)
{
  auto& in_tex = inputs.image.texture;
  const float min_conf = inputs.min_confidence;
  const bool draw_lines = inputs.draw_skeleton.value;
  const bool skeleton_only = (inputs.output_mode.value == OutputMode::SkeletonOnly);

  // Create output image
  QImage img;
  if(skeleton_only)
  {
    img = QImage(in_tex.width, in_tex.height, QImage::Format_RGBA8888);
    img.fill(Qt::black);
  }
  else
  {
    img = QImage(in_tex.bytes, in_tex.width, in_tex.height, QImage::Format_RGBA8888);
  }

  QPainter p(&img);
  p.setRenderHint(QPainter::Antialiasing);

  const auto& kps = pose.keypoints;
  const int num_kps = static_cast<int>(kps.size());
  const int w = in_tex.width;
  const int h = in_tex.height;

  // Helper to convert keypoint to pixel coordinates
  auto toPoint = [&](int idx) -> QPointF {
    return QPointF(kps[idx].x * w, kps[idx].y * h);
  };

  // Helper to safely set alpha (clamp to valid range)
  auto safeAlpha
      = [](float conf) -> float { return std::clamp(conf, 0.5f, 1.0f); };

  // Select skeleton connections and color function based on workflow
  auto getColor = [&](int idx) -> QColor {
    switch(workflow)
    {
      case PoseWorkflow::BlazePose:
        return getBlazePoseColor(idx);
      case PoseWorkflow::RTMPose_COCO:
        // RTMPose COCO body has 17 keypoints, Hand has 21 keypoints
        if(num_kps == 21)
          return getHandColor(idx);
        return getCOCOColor(idx);
      case PoseWorkflow::RTMPose_Whole:
        if (num_kps == 21)
          return getHandColor(idx);
        return getWholeBodyColor(idx);
      case PoseWorkflow::MediaPipeHands:
        return getHandColor(idx);
      case PoseWorkflow::BlazeFace:
        return getBlazeFaceColor(idx);
      case PoseWorkflow::FaceMesh:
        return Colors::face;
      case PoseWorkflow::MobileFaceNet:
        return getDlib68Color(idx);
      default:
        return getCOCOColor(idx);
    }
  };

  // Draw skeleton connections
  if(draw_lines)
  {
    auto drawConnections = [&](const auto& connections, int max_idx) {
      for(const auto& [from, to] : connections)
      {
        if(from < 0 || to < 0)
          continue;
        if(from >= num_kps || to >= num_kps || from >= max_idx || to >= max_idx)
          continue;

        float conf = std::min(kps[from].confidence, kps[to].confidence);
        if(conf < min_conf)
          continue;

        QColor color = getColor(from);
        color.setAlphaF(safeAlpha(conf));
        p.setPen(QPen(color, 2));
        p.drawLine(toPoint(from), toPoint(to));
      }
    };

    switch(workflow)
    {
      case PoseWorkflow::BlazePose:
        drawConnections(Skeletons::blazepose, 33);
        break;
      case PoseWorkflow::RTMPose_COCO:
        // RTMPose COCO body has 17 keypoints, Hand has 21 keypoints
        if(num_kps == 21)
          drawConnections(Skeletons::hands, 21);
        else
          drawConnections(Skeletons::coco17, 17);
        break;
      case PoseWorkflow::RTMPose_Whole:
        if (num_kps == 21)
          drawConnections(Skeletons::hands, 21);
        else
          drawConnections(Skeletons::wholebody_body, 17);
        break;
      case PoseWorkflow::MediaPipeHands:
        drawConnections(Skeletons::hands, 21);
        break;
      case PoseWorkflow::BlazeFace:
        drawConnections(Skeletons::blazeface, 6);
        break;
      case PoseWorkflow::FaceMesh:
        // FaceMesh has too many points for skeleton lines, just draw contours
        // Draw face oval, eyes, lips as closed contours
        {
          auto drawContour = [&](const auto& indices, QColor color) {
            color.setAlphaF(0.8f);
            p.setPen(QPen(color, 1));
            for(size_t i = 0; i + 1 < indices.size(); ++i)
            {
              int from = indices[i];
              int to = indices[i + 1];
              if(from < 0 || to < 0)
                continue;
              if(from >= num_kps || to >= num_kps)
                continue;
              p.drawLine(toPoint(from), toPoint(to));
            }
          };
          drawContour(Onnx::FaceMesh::FaceContours::face_oval, QColor(200, 200, 255));
          drawContour(Onnx::FaceMesh::FaceContours::left_eye, QColor(0, 255, 255));
          drawContour(Onnx::FaceMesh::FaceContours::right_eye, QColor(0, 255, 255));
          drawContour(Onnx::FaceMesh::FaceContours::lips_outer, QColor(255, 100, 100));
          drawContour(Onnx::FaceMesh::FaceContours::left_eyebrow, QColor(255, 255, 0));
          drawContour(Onnx::FaceMesh::FaceContours::right_eyebrow, QColor(255, 255, 0));
        }
        break;
      case PoseWorkflow::MobileFaceNet:
        drawConnections(Skeletons::dlib68, 68);
        break;
      default:
        drawConnections(Skeletons::coco17, 17);
        break;
    }
  }

  // Draw keypoints
  p.setPen(Qt::NoPen);
  for(int i = 0; i < num_kps; ++i)
  {
    float conf = kps[i].confidence;
    if(conf < min_conf)
      continue;

    QColor color = getColor(i);
    color.setAlphaF(safeAlpha(conf));
    p.setBrush(color);

    // Size based on keypoint importance
    int radius = 4;
    if(workflow == PoseWorkflow::RTMPose_Whole)
    {
      if(i <= 16)
        radius = 4;
      else if(i <= 22)
        radius = 3; // Feet
      else if(i <= 90)
        radius = 1; // Face
      else
        radius = 2; // Hands
    }
    else if(workflow == PoseWorkflow::BlazePose)
    {
      radius = (i <= 10) ? 3 : 4; // Face vs body
    }
    else if(workflow == PoseWorkflow::MediaPipeHands)
    {
      // Fingertips larger
      constexpr int fingertips[] = {4, 8, 12, 16, 20};
      bool is_tip = false;
      for(int tip : fingertips)
        if(i == tip) is_tip = true;
      radius = is_tip ? 5 : (i == 0 ? 6 : 3); // Wrist largest, tips medium, others small
    }
    else if(workflow == PoseWorkflow::FaceMesh)
    {
      radius = 1; // Very small for dense face mesh
    }
    else if(workflow == PoseWorkflow::BlazeFace)
    {
      radius = 5; // Larger for sparse face keypoints
    }
    else if(workflow == PoseWorkflow::MobileFaceNet)
    {
      radius = 2; // Small for 68 face landmarks
    }

    p.drawEllipse(toPoint(i), radius, radius);
  }

  // Copy to output
  outputs.image.create(in_tex.width, in_tex.height);
  memcpy(outputs.image.texture.bytes, img.constBits(), w * h * 4);
  outputs.image.texture.changed = true;
}

// Generate geometry output based on format setting
void PoseDetector::generateGeometryOutput(const DetectedPose& pose, PoseWorkflow workflow)
{
  auto& out = outputs.geometry.value;
  out.clear();

  const auto& kps = pose.keypoints;
  if(kps.empty())
    return;

  const float min_conf = inputs.min_confidence;
  const auto format = inputs.data_format.value;

  switch (format)
  {
    case KeypointOutputFormat::Raw:
    {
      // x, y, z, confidence for each keypoint
      out.reserve(kps.size() * 4);
      for (const auto& kp : kps)
      {
        out.push_back(kp.x);
        out.push_back(kp.y);
        out.push_back(kp.z);
        out.push_back(kp.confidence);
      }
      break;
    }
    case KeypointOutputFormat::XYArray:
    {
      out.reserve(kps.size() * 3);
      for (const auto& kp : kps)
      {
        if (kp.confidence >= min_conf)
        {
          out.push_back(kp.x);
          out.push_back(kp.y);
        }
      }
      break;
    }
    case KeypointOutputFormat::XYZArray:
    {
      out.reserve(kps.size() * 3);
      for (const auto& kp : kps)
      {
        if (kp.confidence >= min_conf)
        {
          out.push_back(kp.x);
          out.push_back(kp.y);
          out.push_back(kp.z);
        }
      }
      break;
    }

    case KeypointOutputFormat::LineArray:
    {
      // xyz pairs for GL_LINES (bone connections)
      auto addLine = [&](int from, int to)
      {
        if (from >= static_cast<int>(kps.size())
            || to >= static_cast<int>(kps.size()))
          return;
        if (from < 0 || to < 0)
          return;
        if (kps[from].confidence < min_conf || kps[to].confidence < min_conf)
          return;

        out.push_back(kps[from].x);
        out.push_back(kps[from].y);
        out.push_back(kps[from].z);
        out.push_back(kps[to].x);
        out.push_back(kps[to].y);
        out.push_back(kps[to].z);
      };

      // Select skeleton based on workflow
      switch (workflow)
      {
        case PoseWorkflow::BlazePose:
          out.reserve(Skeletons::blazepose.size() * 6);
          for (const auto& [from, to] : Skeletons::blazepose)
            addLine(from, to);
          break;

        case PoseWorkflow::RTMPose_COCO:
          // RTMPose COCO body has 17 keypoints, Hand has 21 keypoints
          if (kps.size() == 21)
          {
            out.reserve(Skeletons::hands.size() * 6);
            for (const auto& [from, to] : Skeletons::hands)
              addLine(from, to);
          }
          else
          {
            out.reserve(Skeletons::coco17.size() * 6);
            for (const auto& [from, to] : Skeletons::coco17)
              addLine(from, to);
          }
          break;

        case PoseWorkflow::ViTPose:
        case PoseWorkflow::YOLOPose:
          out.reserve(Skeletons::coco17.size() * 6);
          for (const auto& [from, to] : Skeletons::coco17)
            addLine(from, to);
          break;

        case PoseWorkflow::RTMPose_Whole:
          out.reserve(Skeletons::wholebody_body.size() * 6);
          for (const auto& [from, to] : Skeletons::wholebody_body)
            addLine(from, to);
          break;

        case PoseWorkflow::MediaPipeHands:
          out.reserve(Skeletons::hands.size() * 6);
          for (const auto& [from, to] : Skeletons::hands)
            addLine(from, to);
          break;

        case PoseWorkflow::BlazeFace:
          out.reserve(Skeletons::blazeface.size() * 6);
          for (const auto& [from, to] : Skeletons::blazeface)
            addLine(from, to);
          break;

        case PoseWorkflow::MobileFaceNet:
          out.reserve(Skeletons::dlib68.size() * 6);
          for (const auto& [from, to] : Skeletons::dlib68)
            addLine(from, to);
          break;

        case PoseWorkflow::FaceMesh:
        {
          // FaceMesh uses contour indices, generate lines from contours
          auto addContour = [&](const auto& indices)
          {
            for (size_t i = 0; i + 1 < indices.size(); ++i)
              addLine(indices[i], indices[i + 1]);
          };
          addContour(Onnx::FaceMesh::FaceContours::face_oval);
          addContour(Onnx::FaceMesh::FaceContours::left_eye);
          addContour(Onnx::FaceMesh::FaceContours::right_eye);
          addContour(Onnx::FaceMesh::FaceContours::lips_outer);
          addContour(Onnx::FaceMesh::FaceContours::left_eyebrow);
          addContour(Onnx::FaceMesh::FaceContours::right_eyebrow);
          break;
        }

        case PoseWorkflow::Auto:
          // Should not happen, but fall through to COCO
          for (const auto& [from, to] : Skeletons::coco17)
            addLine(from, to);
          break;
      }
    }
    break;
  }
}

void PoseDetector::runBlazePose()
{
  auto& in_tex = inputs.image.texture;
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();

  // Detect model size from input shape (NHWC: [N, H, W, C])
  int model_size = 256; // default
  if (!spec.inputs.empty() && spec.inputs[0].shape.size() == 4)
  {
    model_size = static_cast<int>(spec.inputs[0].shape[1]);
  }

  // BlazePose uses NHWC format with [0, 1] normalization
  auto t = Onnx::nhwc_rgb_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      model_size,
      model_size,
      storage);

  Ort::Value tensor_inputs[1] = {std::move(t.value)};

  Ort::Value tensor_outputs[5]{
      Ort::Value{nullptr},
      Ort::Value{nullptr},
      Ort::Value{nullptr},
      Ort::Value{nullptr},
      Ort::Value{nullptr}};

  ctx.infer(
      spec,
      tensor_inputs,
      std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()));

  std::optional<Blazepose::BlazePose_fullbody::pose_data> out;
  Blazepose::BlazePose_fullbody::processOutput(spec, tensor_outputs, out);

  if(out)
  {
    // Setup coordinate transform for mapping back to source image
    CoordTransform xform;
    xform.init(in_tex.width, in_tex.height, model_size, model_size);

    DetectedPose detected;
    detected.keypoints.reserve(33);

    float sum_conf = 0.0f;
    for (int i = 0; i < 33; ++i)
    {
      const auto& kp = out->keypoints[i];
      // BlazePose outputs pixel coordinates, transform to source image space
      float sx, sy;
      xform.modelPixelsToSource(kp.x, kp.y, sx, sy);
      detected.keypoints.push_back({sx, sy, kp.z / model_size, kp.presence});
      sum_conf += kp.presence;
    }
    detected.mean_confidence = sum_conf / 33.0f;
    outputs.detection.value = std::move(detected);

    drawSkeleton(*outputs.detection.value, PoseWorkflow::BlazePose);
    generateGeometryOutput(*outputs.detection.value, PoseWorkflow::BlazePose);
  }
  else
  {
    outputs.detection.value.reset();
    outputs.geometry.value.clear();
    // Pass through input
    outputs.image.create(in_tex.width, in_tex.height);
    memcpy(outputs.image.texture.bytes, in_tex.bytes, in_tex.width * in_tex.height * 4);
    outputs.image.texture.changed = true;
  }

  std::swap(storage, t.storage);
}

void PoseDetector::runRTMPose()
{
  auto& in_tex = inputs.image.texture;
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();

  // Detect model size from input shape (NCHW: [N, C, H, W])
  int model_w = 192, model_h = 256; // default
  if(!spec.inputs.empty() && spec.inputs[0].shape.size() == 4)
  {
    model_h = static_cast<int>(spec.inputs[0].shape[2]);
    model_w = static_cast<int>(spec.inputs[0].shape[3]);
  }

  // RTMPose uses NCHW format with ImageNet normalization
  auto t = Onnx::nchw_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      model_w,
      model_h,
      storage,
      {123.675f, 116.28f, 103.53f},
      {58.395f, 57.12f, 57.375f});

  Ort::Value tensor_outputs[5]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}};

  // Check if model needs bbox input (post-processed models have 2 inputs)
  if(spec.inputs.size() >= 2)
  {
    // Post-processed model: provide bboxes_width_height as [model_w, model_h]
    std::array<int64_t, 2> bbox_wh = {model_w, model_h};
    std::array<int64_t, 2> bbox_shape = {1, 2};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto bbox_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, bbox_wh.data(), bbox_wh.size(), bbox_shape.data(), bbox_shape.size());

    Ort::Value tensor_inputs[2] = {std::move(t.value), std::move(bbox_tensor)};
    ctx.infer(
        spec, tensor_inputs,
        std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()));
  }
  else
  {
    // SimCC model: single image input
    Ort::Value tensor_inputs[1] = {std::move(t.value)};
    ctx.infer(
        spec, tensor_inputs,
        std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()));
  }

  // Auto-detect config and format from output shapes
  auto output_span = std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size());
  Onnx::RTMPose::OutputFormat format;
  auto config = Onnx::RTMPose::detectConfig(output_span, &format);
  config.input_width = model_w;
  config.input_height = model_h;

  std::optional<Onnx::RTMPose::PoseResult> result;
  Onnx::RTMPose::processOutput(spec, output_span, config, result, format);

  if(result)
  {
    // Setup coordinate transform for mapping back to source image
    CoordTransform xform;
    xform.init(in_tex.width, in_tex.height, model_w, model_h);

    DetectedPose detected;
    detected.keypoints.reserve(result->keypoints.size());
    for(const auto& kp : result->keypoints)
    {
      // RTMPose outputs normalized [0,1] coords, transform to source image space
      float sx, sy;
      xform.modelToSource(kp.x, kp.y, sx, sy);
      detected.keypoints.push_back({sx, sy, 0.0f, kp.confidence});
    }
    detected.mean_confidence = result->mean_confidence;
    outputs.detection.value = std::move(detected);

    auto workflow = (config.num_keypoints > 17)
        ? PoseWorkflow::RTMPose_Whole
        : PoseWorkflow::RTMPose_COCO;
    drawSkeleton(*outputs.detection.value, workflow);
    generateGeometryOutput(*outputs.detection.value, workflow);
  }
  else
  {
    outputs.detection.value.reset();
    outputs.geometry.value.clear();
    outputs.image.create(in_tex.width, in_tex.height);
    memcpy(outputs.image.texture.bytes, in_tex.bytes, in_tex.width * in_tex.height * 4);
    outputs.image.texture.changed = true;
  }

  std::swap(storage, t.storage);
}

void PoseDetector::runViTPose()
{
  auto& in_tex = inputs.image.texture;
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();

  // Detect model size from input shape (NCHW: [N, C, H, W])
  int model_w = 192, model_h = 256; // default
  if(!spec.inputs.empty() && spec.inputs[0].shape.size() == 4)
  {
    model_h = static_cast<int>(spec.inputs[0].shape[2]);
    model_w = static_cast<int>(spec.inputs[0].shape[3]);
  }

  // ViTPose uses NCHW format with ImageNet normalization
  auto t = Onnx::nchw_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      model_w,
      model_h,
      storage,
      {123.675f, 116.28f, 103.53f},
      {58.395f, 57.12f, 57.375f});

  Ort::Value tensor_inputs[1] = {std::move(t.value)};

  Ort::Value tensor_outputs[5]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}};

  ctx.infer(
      spec, tensor_inputs,
      std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()));

  // ViTPose outputs heatmaps: [1, num_keypoints, hm_h, hm_w]
  if(spec.output_names_char.empty())
  {
    outputs.detection.value.reset();
    outputs.geometry.value.clear();
    return;
  }

  auto& heatmap_output = tensor_outputs[0];
  auto shape = heatmap_output.GetTensorTypeAndShapeInfo().GetShape();

  if(shape.size() != 4)
  {
    outputs.detection.value.reset();
    outputs.geometry.value.clear();
    return;
  }

  const int num_keypoints = static_cast<int>(shape[1]);
  const int hm_h = static_cast<int>(shape[2]);
  const int hm_w = static_cast<int>(shape[3]);
  const float* heatmaps = heatmap_output.GetTensorData<float>();

  // Setup coordinate transform for mapping back to source image
  CoordTransform xform;
  xform.init(in_tex.width, in_tex.height, model_w, model_h);

  DetectedPose detected;
  detected.keypoints.reserve(num_keypoints);

  float sum_conf = 0.0f;
  const float scale_x = 1.0f / hm_w;
  const float scale_y = 1.0f / hm_h;

  for(int k = 0; k < num_keypoints; ++k)
  {
    const float* hm = heatmaps + k * hm_h * hm_w;

    // Find peak
    int max_idx = 0;
    float max_val = hm[0];
    for(int i = 1; i < hm_h * hm_w; ++i)
    {
      if(hm[i] > max_val)
      {
        max_val = hm[i];
        max_idx = i;
      }
    }

    int hm_x = max_idx % hm_w;
    int hm_y = max_idx / hm_w;

    // Sub-pixel refinement
    float dx = 0.0f, dy = 0.0f;
    if(hm_x > 0 && hm_x < hm_w - 1)
    {
      dx = 0.25f * (hm[hm_y * hm_w + hm_x + 1] - hm[hm_y * hm_w + hm_x - 1]);
    }
    if(hm_y > 0 && hm_y < hm_h - 1)
    {
      dy = 0.25f * (hm[(hm_y + 1) * hm_w + hm_x] - hm[(hm_y - 1) * hm_w + hm_x]);
    }

    // Normalized coordinates in model space
    float mx = (hm_x + dx + 0.5f) * scale_x;
    float my = (hm_y + dy + 0.5f) * scale_y;

    // Transform to source image space
    float sx, sy;
    xform.modelToSource(std::clamp(mx, 0.0f, 1.0f), std::clamp(my, 0.0f, 1.0f), sx, sy);

    detected.keypoints.push_back({sx, sy, 0.0f, max_val});
    sum_conf += max_val;
  }

  detected.mean_confidence = sum_conf / num_keypoints;
  outputs.detection.value = std::move(detected);

  drawSkeleton(*outputs.detection.value, PoseWorkflow::ViTPose);
  generateGeometryOutput(*outputs.detection.value, PoseWorkflow::ViTPose);

  std::swap(storage, t.storage);
}

void PoseDetector::runYOLOPose()
{
  auto& in_tex = inputs.image.texture;
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();

  // Detect model size from input shape (NCHW: [N, C, H, W])
  int model_size = 640; // default
  if(!spec.inputs.empty() && spec.inputs[0].shape.size() == 4)
  {
    model_size = static_cast<int>(spec.inputs[0].shape[2]);
  }

  // YOLO uses NCHW format with [0, 1] normalization (divide by 255)
  auto t = Onnx::nchw_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      model_size,
      model_size,
      storage,
      {0.f, 0.f, 0.f},
      {255.f, 255.f, 255.f});

  Ort::Value tensor_inputs[1] = {std::move(t.value)};

  Ort::Value tensor_outputs[1]{Ort::Value{nullptr}};

  ctx.infer(spec, tensor_inputs, tensor_outputs);

  // Use YOLO pose processor
  static const Yolo::YOLO_pose yolo_pose;
  std::vector<Yolo::YOLO_pose::pose_type> poses;
  yolo_pose.processOutput(
      spec,
      tensor_outputs,
      poses,
      100,                      // max_detect
      inputs.min_confidence,    // min_confidence
      0, 0,                     // image offset
      model_size, model_size,   // image size
      model_size, model_size);  // model size

  if(!poses.empty())
  {
    // Setup coordinate transform for mapping back to source image
    CoordTransform xform;
    xform.init(in_tex.width, in_tex.height, model_size, model_size);

    // Use the first (highest confidence) detection
    const auto& pose = poses[0];

    DetectedPose detected;
    detected.keypoints.reserve(17);

    // YOLO outputs sparse keypoints, fill in all 17 with defaults
    for(int i = 0; i < 17; ++i)
    {
      detected.keypoints.push_back({0.0f, 0.0f, 0.0f, 0.0f});
    }

    // Fill in detected keypoints (transform from model pixels to source normalized)
    for(const auto& kp : pose.keypoints)
    {
      if(kp.kp >= 0 && kp.kp < 17)
      {
        float sx, sy;
        xform.modelPixelsToSource(kp.x, kp.y, sx, sy);
        detected.keypoints[kp.kp] = {sx, sy, 0.0f, 1.0f};
      }
    }
    detected.mean_confidence = pose.confidence;
    outputs.detection.value = std::move(detected);

    drawSkeleton(*outputs.detection.value, PoseWorkflow::YOLOPose);
    generateGeometryOutput(*outputs.detection.value, PoseWorkflow::YOLOPose);
  }
  else
  {
    outputs.detection.value.reset();
    outputs.geometry.value.clear();
    outputs.image.create(in_tex.width, in_tex.height);
    memcpy(outputs.image.texture.bytes, in_tex.bytes, in_tex.width * in_tex.height * 4);
    outputs.image.texture.changed = true;
  }

  std::swap(storage, t.storage);
}

void PoseDetector::runMediaPipeHands()
{
  auto& in_tex = inputs.image.texture;
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();

  // Detect model size from input shape (NHWC: [N, H, W, C])
  int model_size = 224; // default
  if(!spec.inputs.empty() && spec.inputs[0].shape.size() == 4)
  {
    model_size = static_cast<int>(spec.inputs[0].shape[1]);
  }

  // MediaPipe Hands uses NHWC format with [0, 1] normalization
  auto t = Onnx::nhwc_rgb_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      model_size,
      model_size,
      storage);

  Ort::Value tensor_inputs[1] = {std::move(t.value)};

  Ort::Value tensor_outputs[5]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}};

  ctx.infer(
      spec, tensor_inputs,
      std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()));

  std::optional<Onnx::MediaPipeHands::HandResult> result;
  Onnx::MediaPipeHands::processOutput(
      spec,
      std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()),
      result);

  if(result)
  {
    // Setup coordinate transform for mapping back to source image
    CoordTransform xform;
    xform.init(in_tex.width, in_tex.height, model_size, model_size);

    DetectedPose detected;
    detected.keypoints.reserve(result->landmarks.size());

    for(const auto& lm : result->landmarks)
    {
      // MediaPipe outputs normalized [0,1] coords, transform to source image space
      float sx, sy;
      xform.modelToSource(lm.x, lm.y, sx, sy);
      detected.keypoints.push_back({sx, sy, lm.z, 1.0f});
    }
    detected.mean_confidence = result->hand_flag;
    outputs.detection.value = std::move(detected);

    drawSkeleton(*outputs.detection.value, PoseWorkflow::MediaPipeHands);
    generateGeometryOutput(*outputs.detection.value, PoseWorkflow::MediaPipeHands);
  }
  else
  {
    outputs.detection.value.reset();
    outputs.geometry.value.clear();
    outputs.image.create(in_tex.width, in_tex.height);
    memcpy(outputs.image.texture.bytes, in_tex.bytes, in_tex.width * in_tex.height * 4);
    outputs.image.texture.changed = true;
  }

  std::swap(storage, t.storage);
}

void PoseDetector::runFaceMesh()
{
  auto& in_tex = inputs.image.texture;
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();

  // Detect model size from input shape (NHWC: [N, H, W, C])
  int model_size = 192; // default
  if(!spec.inputs.empty() && spec.inputs[0].shape.size() == 4)
  {
    model_size = static_cast<int>(spec.inputs[0].shape[1]);
  }

  // FaceMesh uses NHWC format with [0, 1] normalization
  auto t = Onnx::nhwc_rgb_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      model_size,
      model_size,
      storage);

  Ort::Value tensor_inputs[1] = {std::move(t.value)};

  Ort::Value tensor_outputs[5]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}};

  ctx.infer(
      spec, tensor_inputs,
      std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()));

  std::optional<Onnx::FaceMesh::FaceMeshResult> result;
  Onnx::FaceMesh::processOutput(
      spec,
      std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()),
      Onnx::FaceMesh::NUM_LANDMARKS,
      result);

  if(result)
  {
    // Setup coordinate transform for mapping back to source image
    CoordTransform xform;
    xform.init(in_tex.width, in_tex.height, model_size, model_size);

    DetectedPose detected;
    detected.keypoints.reserve(result->landmarks.size());

    for(const auto& lm : result->landmarks)
    {
      // FaceMesh outputs normalized [0,1] coords, transform to source image space
      float sx, sy;
      xform.modelToSource(lm.x, lm.y, sx, sy);
      detected.keypoints.push_back({sx, sy, lm.z, 1.0f});
    }
    detected.mean_confidence = result->face_flag;
    outputs.detection.value = std::move(detected);

    drawSkeleton(*outputs.detection.value, PoseWorkflow::FaceMesh);
    generateGeometryOutput(*outputs.detection.value, PoseWorkflow::FaceMesh);
  }
  else
  {
    outputs.detection.value.reset();
    outputs.geometry.value.clear();
    outputs.image.create(in_tex.width, in_tex.height);
    memcpy(outputs.image.texture.bytes, in_tex.bytes, in_tex.width * in_tex.height * 4);
    outputs.image.texture.changed = true;
  }

  std::swap(storage, t.storage);
}

void PoseDetector::runBlazeFace()
{
  auto& in_tex = inputs.image.texture;
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();

  // Detect model size from input shape (NHWC: [N, H, W, C])
  int model_size = 128; // default
  if(!spec.inputs.empty() && spec.inputs[0].shape.size() == 4)
  {
    model_size = static_cast<int>(spec.inputs[0].shape[1]);
  }

  // BlazeFace uses NHWC format with [0, 1] normalization
  auto t = Onnx::nhwc_rgb_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      model_size,
      model_size,
      storage);

  Ort::Value tensor_inputs[1] = {std::move(t.value)};

  Ort::Value tensor_outputs[5]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}};

  ctx.infer(
      spec, tensor_inputs,
      std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()));

  std::vector<Onnx::BlazeFace::Detection> detections;
  Onnx::BlazeFace::processOutput(
      spec,
      std::span<Ort::Value>(tensor_outputs, spec.output_names_char.size()),
      model_size,
      inputs.min_confidence,
      0.3f, // NMS threshold
      detections);

  if(!detections.empty())
  {
    // Setup coordinate transform for mapping back to source image
    CoordTransform xform;
    xform.init(in_tex.width, in_tex.height, model_size, model_size);

    // Use the first (highest confidence) detection
    const auto& det = detections[0];
    if (!ossia::safe_isnan(det.score))
    {
      DetectedPose detected;
      detected.keypoints.reserve(6);

      for (int k = 0; k < 6; ++k)
      {
        // BlazeFace outputs normalized [0,1] coords, transform to source image space
        float sx, sy;
        xform.modelToSource(det.keypoints[k].x, det.keypoints[k].y, sx, sy);
        detected.keypoints.push_back({sx, sy, 0.0f, det.score});
      }
      detected.mean_confidence = det.score;
      outputs.detection.value = std::move(detected);

      drawSkeleton(*outputs.detection.value, PoseWorkflow::BlazeFace);
      generateGeometryOutput(
          *outputs.detection.value, PoseWorkflow::BlazeFace);
    }
  }
  else
  {
    outputs.detection.value.reset();
    outputs.geometry.value.clear();
    outputs.image.create(in_tex.width, in_tex.height);
    memcpy(outputs.image.texture.bytes, in_tex.bytes, in_tex.width * in_tex.height * 4);
    outputs.image.texture.changed = true;
  }

  std::swap(storage, t.storage);
}

void PoseDetector::runMobileFaceNet()
{
  auto& in_tex = inputs.image.texture;
  auto& ctx = *this->ctx;
  auto spec = ctx.readModelSpec();

  // Detect model size from input shape (NCHW: [N, C, H, W])
  int model_size = 112; // default
  if(!spec.inputs.empty() && spec.inputs[0].shape.size() == 4)
  {
    model_size = static_cast<int>(spec.inputs[0].shape[2]);
  }

  // NCHW format with ImageNet normalization
  // mean = [0.485, 0.456, 0.406] * 255, std = [0.229, 0.224, 0.225] * 255
  auto t = Onnx::nchw_tensorFromRGBA(
      spec.inputs[0],
      in_tex.bytes,
      in_tex.width,
      in_tex.height,
      model_size,
      model_size,
      storage,
      {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f},
      {0.229f * 255.f, 0.224f * 255.f, 0.225f * 255.f});

  Ort::Value tensor_inputs[1] = {std::move(t.value)};

  Ort::Value tensor_outputs[1]{Ort::Value{nullptr}};

  ctx.infer(spec, tensor_inputs, tensor_outputs);

  // MobileFaceNet outputs [1, 68, 2] - 68 landmarks with normalized (x, y) in [0, 1]
  auto& output = tensor_outputs[0];
  auto shape = output.GetTensorTypeAndShapeInfo().GetShape();

  if(shape.size() >= 2)
  {
    // Setup coordinate transform for mapping back to source image
    CoordTransform xform;
    xform.init(in_tex.width, in_tex.height, model_size, model_size);

    const int num_landmarks = static_cast<int>(shape[1]);
    const float* data = output.GetTensorData<float>();

    DetectedPose detected;
    detected.keypoints.reserve(num_landmarks);

    for(int i = 0; i < num_landmarks; ++i)
    {
      // Output is normalized [0,1], convert to model pixels then to source
      float mx_pix = data[i * 2] * model_size;
      float my_pix = data[i * 2 + 1] * model_size;
      float sx, sy;
      xform.modelPixelsToSource(mx_pix, my_pix, sx, sy);
      detected.keypoints.push_back({sx, sy, 0.0f, 1.0f});
    }
    detected.mean_confidence = 1.0f; // MobileFaceNet doesn't output confidence
    outputs.detection.value = std::move(detected);

    drawSkeleton(*outputs.detection.value, PoseWorkflow::MobileFaceNet);
    generateGeometryOutput(*outputs.detection.value, PoseWorkflow::MobileFaceNet);
  }
  else
  {
    outputs.detection.value.reset();
    outputs.geometry.value.clear();
    outputs.image.create(in_tex.width, in_tex.height);
    memcpy(outputs.image.texture.bytes, in_tex.bytes, in_tex.width * in_tex.height * 4);
    outputs.image.texture.changed = true;
  }

  std::swap(storage, t.storage);
}

// Detect workflow from model input/output structure
PoseWorkflow PoseDetector::detectWorkflowFromModel()
{
  if(!this->ctx)
    return PoseWorkflow::BlazePose; // Default fallback

  auto spec = this->ctx->readModelSpec();

  if(spec.inputs.empty())
    return PoseWorkflow::BlazePose;

  const auto& input = spec.inputs[0];
  const auto& input_shape = input.shape;

  // Determine if NHWC or NCHW based on input shape
  // NHWC: [N, H, W, C] where C is typically 3
  // NCHW: [N, C, H, W] where C is typically 3
  bool is_nhwc = false;
  int input_h = 0, input_w = 0;

  if(input_shape.size() == 4)
  {
    if(input_shape[3] == 3)
    {
      // NHWC format: [N, H, W, 3]
      is_nhwc = true;
      input_h = static_cast<int>(input_shape[1]);
      input_w = static_cast<int>(input_shape[2]);
    }
    else if(input_shape[1] == 3)
    {
      // NCHW format: [N, 3, H, W]
      is_nhwc = false;
      input_h = static_cast<int>(input_shape[2]);
      input_w = static_cast<int>(input_shape[3]);
    }
  }

  // Analyze outputs
  const auto& outputs = spec.outputs;
  int num_outputs = static_cast<int>(outputs.size());

  // Check for specific output patterns
  bool has_simcc = false;
  bool has_heatmap = false;
  int64_t total_output_size = 0;
  int64_t first_output_size = 0;
  std::vector<int64_t> output_shapes;

  for(const auto& out : outputs)
  {
    int64_t size = 1;
    for(auto dim : out.shape)
    {
      if(dim > 0)
        size *= dim;
    }
    output_shapes.push_back(size);
    total_output_size += size;

    if(output_shapes.size() == 1)
      first_output_size = size;

    // Check for SimCC pattern: output names contain "simcc"
    if(out.name.contains("simcc", Qt::CaseInsensitive))
      has_simcc = true;

    // Check for heatmap pattern: 4D output with shape [N, K, H, W] where H, W are small
    if(out.shape.size() == 4 && out.shape[2] <= 128 && out.shape[3] <= 128
       && out.shape[1] > 1 && out.shape[1] <= 150)
    {
      has_heatmap = true;
    }
  }

  // --- Detection logic ---

  // 1. SimCC models (RTMPose)
  if(has_simcc)
  {
    // Check number of keypoints from simcc_x shape
    for(const auto& out : outputs)
    {
      if(out.name.contains("simcc_x", Qt::CaseInsensitive) && out.shape.size() >= 2)
      {
        int num_keypoints = static_cast<int>(out.shape[1]);
        if(num_keypoints > 50)
          return PoseWorkflow::RTMPose_Whole;
        else
          return PoseWorkflow::RTMPose_COCO;
      }
    }
    return PoseWorkflow::RTMPose_COCO;
  }

  // 2. NHWC models (MediaPipe family)
  if(is_nhwc)
  {
    // BlazeFace: 128x128 input, outputs with 896 anchors
    if(input_h == 128 && input_w == 128)
    {
      for(const auto& out : outputs)
      {
        if(out.shape.size() >= 2 && out.shape[1] == 896)
          return PoseWorkflow::BlazeFace;
      }
    }

    // FaceMesh: 192x192 input, output with 1404 values (468×3)
    if(input_h == 192 && input_w == 192)
    {
      for(const auto& out : outputs)
      {
        int64_t size = 1;
        for(auto dim : out.shape)
          if(dim > 0)
            size *= dim;
        if(size == 1404 || size == 1434) // 468×3 or 478×3 (with iris)
          return PoseWorkflow::FaceMesh;
      }
    }

    // MediaPipe Hands: 224x224 or 256x256 input, output with 63 values (21×3)
    if((input_h == 224 && input_w == 224) || (input_h == 256 && input_w == 256))
    {
      for(const auto& out : outputs)
      {
        int64_t size = 1;
        for(auto dim : out.shape)
          if(dim > 0)
            size *= dim;
        if(size == 63) // 21×3
          return PoseWorkflow::MediaPipeHands;
      }
    }

    // BlazePose: 256x256 input, output with 195 values (39×5)
    if(input_h == 256 && input_w == 256)
    {
      for(const auto& out : outputs)
      {
        int64_t size = 1;
        for(auto dim : out.shape)
          if(dim > 0)
            size *= dim;
        if(size == 195) // 39×5
          return PoseWorkflow::BlazePose;
      }
    }

    // Generic NHWC fallback based on output size
    if(first_output_size == 1404 || first_output_size == 1434)
      return PoseWorkflow::FaceMesh;
    if(first_output_size == 63)
      return PoseWorkflow::MediaPipeHands;
    if(first_output_size == 195)
      return PoseWorkflow::BlazePose;

    return PoseWorkflow::BlazePose; // Default NHWC
  }

  // 3. NCHW models
  if(!is_nhwc)
  {
    // MobileFaceNet: 112x112 input, output [N, 68, 2]
    if(input_h == 112 && input_w == 112)
    {
      for(const auto& out : outputs)
      {
        if(out.shape.size() >= 2 && out.shape[1] == 68)
          return PoseWorkflow::MobileFaceNet;
      }
    }

    // ViTPose: heatmap output
    if(has_heatmap)
    {
      return PoseWorkflow::ViTPose;
    }

    // YOLO Pose: typically 640x640 input, single output with detection format
    if(input_h >= 320 && input_w >= 320 && num_outputs == 1)
    {
      const auto& out = outputs[0];
      // YOLO output is typically [1, num_detections, values_per_detection]
      // or [1, values_per_detection, num_detections]
      if(out.shape.size() == 3)
      {
        // Check if it looks like YOLO pose output (56 = 4 box + 1 conf + 17 kpts × 3)
        // or similar detection format
        return PoseWorkflow::YOLOPose;
      }
    }

    // RTMPose without simcc in name: check for typical shapes
    // RTMPose outputs two tensors with matching keypoint dimension
    if(num_outputs == 2)
    {
      const auto& out0 = outputs[0];
      const auto& out1 = outputs[1];
      if(out0.shape.size() >= 2 && out1.shape.size() >= 2)
      {
        if(out0.shape[1] == out1.shape[1])
        {
          int num_kpts = static_cast<int>(out0.shape[1]);
          if(num_kpts == 17 || num_kpts == 21 || num_kpts == 133)
          {
            return (num_kpts > 50) ? PoseWorkflow::RTMPose_Whole : PoseWorkflow::RTMPose_COCO;
          }
        }
      }
    }

    // RTMPose with post-processing: single output [N, K, 3] where K is keypoint count
    // The 3 values are (x, y, score) - already decoded from SimCC
    if(num_outputs == 1)
    {
      const auto& out = outputs[0];
      // Check for [N, K, 3] shape where K is typical keypoint count
      if(out.shape.size() == 3 && out.shape[2] == 3)
      {
        int num_kpts = static_cast<int>(out.shape[1]);
        if(num_kpts == 17 || num_kpts == 21 || num_kpts == 133)
        {
          return (num_kpts > 50) ? PoseWorkflow::RTMPose_Whole : PoseWorkflow::RTMPose_COCO;
        }
      }
    }

    // ViTPose: 4D heatmap output [N, K, H, W]
    if(has_heatmap)
    {
      return PoseWorkflow::ViTPose;
    }

    return PoseWorkflow::BlazePose; // Default NCHW fallback
  }

  return PoseWorkflow::BlazePose; // Ultimate fallback
}

void PoseDetector::operator()()
try
{
  if (!available)
    return;

  if(this->inputs.model.current_model_invalid)
    return;

  auto& in_tex = inputs.image.texture;

  if(!in_tex.changed)
    return;
  if (!in_tex.bytes)
    return;

  // Reset context if workflow changed
  PoseWorkflow current_workflow = inputs.workflow.value;
  if (current_workflow != m_last_workflow)
  {
    ctx.reset();
    m_last_workflow = current_workflow;
  }
  if (inputs.model.file.filename != m_last_model)
  {
    ctx.reset();
    m_last_model = std::string(inputs.model.file.filename);
  }

  if (!this->ctx)
  {
    this->ctx = std::make_unique<Onnx::OnnxRunContext>(
        this->inputs.model.file.bytes);

    // Auto-detect workflow if in Auto mode
    if (current_workflow == PoseWorkflow::Auto)
    {
      m_detected_workflow = detectWorkflowFromModel();
    }
  }

  // Use detected workflow if in Auto mode
  PoseWorkflow effective_workflow = (current_workflow == PoseWorkflow::Auto)
                                        ? m_detected_workflow
                                        : current_workflow;

  // Dispatch to workflow-specific handler
  switch(effective_workflow)
  {
    case PoseWorkflow::Auto:
      // Should not reach here, but fallback to BlazePose
      runBlazePose();
      break;
    case PoseWorkflow::BlazePose:
      runBlazePose();
      break;
    case PoseWorkflow::RTMPose_COCO:
    case PoseWorkflow::RTMPose_Whole:
      runRTMPose();
      break;
    case PoseWorkflow::ViTPose:
      runViTPose();
      break;
    case PoseWorkflow::YOLOPose:
      runYOLOPose();
      break;
    case PoseWorkflow::MediaPipeHands:
      runMediaPipeHands();
      break;
    case PoseWorkflow::FaceMesh:
      runFaceMesh();
      break;
    case PoseWorkflow::BlazeFace:
      runBlazeFace();
      break;
    case PoseWorkflow::MobileFaceNet:
      runMobileFaceNet();
      break;
  }
}
catch(...)
{
  inputs.model.current_model_invalid = true;
}

} // namespace OnnxModels
