#include "PoseDetector.hpp"

#include <ossia/math/safe_math.hpp>

#include <QImage>
#include <QPainter>
#include <QTransform>

#include <Onnx/helpers/BlazeFace.hpp>
#include <Onnx/helpers/BlazePose.hpp>
#include <Onnx/helpers/Detection.hpp>
#include <Onnx/helpers/FaceMesh.hpp>
#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/MediaPipeHands.hpp>
#include <Onnx/helpers/ModelRole.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/ROI.hpp>
#include <Onnx/helpers/RTMPose.hpp>
#include <Onnx/helpers/Yolo.hpp>
#include <cmath>

#include <algorithm>
#include <array>

namespace OnnxModels
{

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

namespace
{
// Map a classified model role to the PoseWorkflow used for drawing/skeletons.
PoseWorkflow workflowForRole(const Onnx::ModelRole& r)
{
  using K = Onnx::ModelKind;
  switch(r.kind)
  {
    case K::BlazePoseLandmark:
    case K::BlazePoseDetector:
      return PoseWorkflow::BlazePose;
    case K::HandLandmark:
    case K::PalmDetector:
      return PoseWorkflow::MediaPipeHands;
    case K::FaceMeshLandmark:
      return PoseWorkflow::FaceMesh;
    case K::MobileFaceNet:
      return PoseWorkflow::MobileFaceNet;
    case K::SimccPose:
      return (r.num_keypoints > 50) ? PoseWorkflow::RTMPose_Whole
                                    : PoseWorkflow::RTMPose_COCO;
    case K::HeatmapPose:
      return PoseWorkflow::ViTPose;
    case K::YoloPose:
    case K::RtmoPose:
      return PoseWorkflow::YOLOPose;
    case K::BlazeFaceDetector:
      return PoseWorkflow::BlazeFace;
    default:
      return PoseWorkflow::BlazePose;
  }
}

// Lightweight ModelSpec -> ModelIO view for the classifier.
Onnx::ModelIO toModelIO(const Onnx::ModelSpec& s)
{
  Onnx::ModelIO io;
  io.inputs.reserve(s.inputs.size());
  io.outputs.reserve(s.outputs.size());
  for(const auto& p : s.inputs)
    io.inputs.push_back({p.name.toStdString(), p.shape});
  for(const auto& p : s.outputs)
    io.outputs.push_back({p.name.toStdString(), p.shape});
  return io;
}

// NHWC float tensor from an RGBA crop with affine pixel mapping out = px*a + b
// ([0,1]: a=1,b=0 ; [-1,1]: a=2,b=-1).
Onnx::FloatTensor nhwcDetectorTensor(
    Onnx::ModelSpec::Port& port, const QImage& crop, int mw, int mh,
    boost::container::vector<float>& storage, float a, float b)
{
  storage.resize(3 * mw * mh, boost::container::default_init);
  const unsigned char* ptr = crop.constBits();
  const int bpl = crop.bytesPerLine();
  float* dst = storage.data();
  int di = 0;
  for(int y = 0; y < mh; ++y)
  {
    const unsigned char* row = ptr + y * bpl;
    for(int x = 0; x < mw; ++x)
    {
      dst[di++] = (row[4 * x + 0] / 255.f) * a + b;
      dst[di++] = (row[4 * x + 1] / 255.f) * a + b;
      dst[di++] = (row[4 * x + 2] / 255.f) * a + b;
    }
  }
  Onnx::FloatTensor f{
      .storage = {},
      .value = Onnx::vec_to_tensor<float>(storage, port.shape)};
  f.storage = std::move(storage);
  return f;
}

// NCHW float tensor for end2end detectors (YOLOX/RTMDet). These expect BGR
// channel order; out = (px - mean)/std per channel (mean/std given in BGR).
Onnx::FloatTensor nchwBgrDetectorTensor(
    Onnx::ModelSpec::Port& port, const QImage& crop, int mw, int mh,
    boost::container::vector<float>& storage, std::array<float, 3> mean_bgr,
    std::array<float, 3> std_bgr)
{
  storage.resize(3 * mw * mh, boost::container::default_init);
  const unsigned char* ptr = crop.constBits();
  const int bpl = crop.bytesPerLine();
  float* b_plane = storage.data();
  float* g_plane = b_plane + mw * mh;
  float* r_plane = g_plane + mw * mh;
  int p = 0;
  for(int y = 0; y < mh; ++y)
  {
    const unsigned char* row = ptr + y * bpl;
    for(int x = 0; x < mw; ++x, ++p)
    {
      const float R = row[4 * x + 0];
      const float G = row[4 * x + 1];
      const float B = row[4 * x + 2];
      b_plane[p] = (B - mean_bgr[0]) / std_bgr[0];
      g_plane[p] = (G - mean_bgr[1]) / std_bgr[1];
      r_plane[p] = (R - mean_bgr[2]) / std_bgr[2];
    }
  }
  Onnx::FloatTensor f{
      .storage = {},
      .value = Onnx::vec_to_tensor<float>(storage, port.shape)};
  f.storage = std::move(storage);
  return f;
}
} // namespace

Onnx::ModelRole PoseDetector::roleForWorkflow(PoseWorkflow w) const
{
  // Keep the real model's input dims/layout; override the kind by selection.
  Onnx::ModelRole r = m_landmark_role;
  using K = Onnx::ModelKind;
  using S = Onnx::ModelStage;
  using D = Onnx::ModelDomain;
  switch(w)
  {
    case PoseWorkflow::BlazePose:
      r.kind = K::BlazePoseLandmark; r.stage = S::Landmark; r.domain = D::Body;
      break;
    case PoseWorkflow::RTMPose_COCO:
    case PoseWorkflow::RTMPose_Whole:
      r.kind = K::SimccPose; r.stage = S::Landmark; r.domain = D::Body;
      break;
    case PoseWorkflow::ViTPose:
      r.kind = K::HeatmapPose; r.stage = S::Landmark; r.domain = D::Body;
      break;
    case PoseWorkflow::YOLOPose:
      r.kind = K::YoloPose; r.stage = S::SingleStage; r.domain = D::Body;
      break;
    case PoseWorkflow::MediaPipeHands:
      r.kind = K::HandLandmark; r.stage = S::Landmark; r.domain = D::Hand;
      break;
    case PoseWorkflow::FaceMesh:
      r.kind = K::FaceMeshLandmark; r.stage = S::Landmark; r.domain = D::Face;
      break;
    case PoseWorkflow::BlazeFace:
      r.kind = K::BlazeFaceDetector; r.stage = S::Detector; r.domain = D::Face;
      break;
    case PoseWorkflow::MobileFaceNet:
      r.kind = K::MobileFaceNet; r.stage = S::Landmark; r.domain = D::Face;
      break;
    case PoseWorkflow::Auto:
    default:
      break;
  }
  return r;
}

void PoseDetector::passthrough(const QImage& src)
{
  outputs.detection.value.reset();
  outputs.geometry.value.clear();
  outputs.image.create(src.width(), src.height());
  std::memcpy(
      outputs.image.texture.bytes, src.constBits(),
      src.width() * src.height() * 4);
  outputs.image.texture.changed = true;

  // Coast: keep temporal state for a few frames so a brief detection dropout
  // doesn't restart smoothing/tracking and lurch on re-acquisition. Only after
  // a sustained loss do we drop everything.
  ++m_lost_frames;
  if(m_lost_frames > 8)
  {
    m_smoother.reset();
    m_roi_smoother.reset();
    m_tracking = false;
    m_last_keypoints.clear();
  }
}

void PoseDetector::applySmoothing(DetectedPose& pose)
{
  m_lost_frames = 0; // we have a pose this frame
  if(!inputs.smoothing.value || pose.keypoints.empty())
    return;
  const float amt
      = std::clamp(static_cast<float>(inputs.smoothing_amount.value), 0.f, 1.f);

  // Log-interpolated cutoff: amt 0 -> ~30 Hz (≈ passthrough), amt 1 -> 0.01 Hz
  // (very smooth). The old linear 3.0..0.1 floor was ~10x too high, so "max"
  // barely smoothed (alpha ~0.39 at dt=1).
  const float min_cutoff = 30.0f * std::pow(0.01f / 30.0f, amt);
  const float beta = 0.02f; // gentle speed adaptivity

  // Filter the raw normalized coordinates. NOTE: an earlier "scale-aware"
  // variant divided each coord by the per-frame keypoint-bbox diagonal. Because
  // the One-Euro STATE is kept in those units while the bbox size changes every
  // frame, a frame with a small/sparse bbox multiplied the lagged state by a
  // tiny scale and collapsed *every* point toward (0,0) — the "all points jump
  // to the top-left" glitch. Scale normalization needs a STABLE per-subject
  // scale; revisit once per-track IDs (ByteTrack) provide one.
  m_smoother.configure(min_cutoff, beta);
  m_smoother.ensure(pose.keypoints.size() * 3);
  for(size_t i = 0; i < pose.keypoints.size(); ++i)
  {
    auto& k = pose.keypoints[i];
    k.x = m_smoother.f[i * 3 + 0].filter(k.x, 1.0f);
    k.y = m_smoother.f[i * 3 + 1].filter(k.y, 1.0f);
    k.z = m_smoother.f[i * 3 + 2].filter(k.z, 1.0f);
  }
}

// ROI rect (image px) from a detection, mirroring the per-kind dialect choice.
Onnx::ROI::Rect PoseDetector::detectionRect(
    const Onnx::ModelRole& role, const Onnx::Detection::Detection& det, int W,
    int H)
{
  const int mw = role.input_w > 0 ? role.input_w : 256;
  const int mh = role.input_h > 0 ? role.input_h : 256;
  switch(role.kind)
  {
    case Onnx::ModelKind::BlazePoseLandmark:
      return Onnx::ROI::mediapipeRect(det, W, H, Onnx::ROI::poseParams());
    case Onnx::ModelKind::HandLandmark:
      return Onnx::ROI::mediapipeRect(det, W, H, Onnx::ROI::handParams());
    case Onnx::ModelKind::FaceMeshLandmark:
    case Onnx::ModelKind::MobileFaceNet:
      return Onnx::ROI::mediapipeRect(det, W, H, Onnx::ROI::faceParams());
    default:
    {
      const QRectF box(
          det.box().x() * W, det.box().y() * H, det.w * W, det.h * H);
      return Onnx::ROI::topdownRect(box, mw, mh, 1.25f);
    }
  }
}

// ROI rect derived from the previous frame's landmarks (MediaPipe tracking
// loop): bbox of the confident keypoints, expanded, with a rotation taken from
// a domain-appropriate keypoint pair so the crop follows the subject.
Onnx::ROI::Rect PoseDetector::roiRectFromKeypoints(
    PoseWorkflow draw, const std::vector<PoseKeypoint>& kps, int W, int H,
    int model_w, int model_h)
{
  Onnx::ROI::Rect r{};
  if(kps.empty())
    return r;

  // bbox of confident keypoints in image px
  float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
  for(const auto& k : kps)
  {
    if(k.confidence < 0.2f)
      continue;
    const float x = k.x * W, y = k.y * H;
    minx = std::min(minx, x); maxx = std::max(maxx, x);
    miny = std::min(miny, y); maxy = std::max(maxy, y);
  }
  if(maxx < minx)
    return r;
  r.cx = 0.5f * (minx + maxx);
  r.cy = 0.5f * (miny + maxy);
  const float bw = maxx - minx, bh = maxy - miny;

  // rotation + scale + squareness by domain
  auto kp = [&](int i) -> QPointF {
    return (i >= 0 && i < (int)kps.size())
               ? QPointF(kps[i].x * W, kps[i].y * H)
               : QPointF(0, 0);
  };
  auto mid = [&](int a, int b) {
    return QPointF((kp(a).x() + kp(b).x()) * 0.5, (kp(a).y() + kp(b).y()) * 0.5);
  };
  QPointF p0, p1; // rotation axis p0->p1; target = vertical/up
  bool rotated = true;
  float scale = 1.25f, target_deg = 90.f;
  bool square = true;

  switch(draw)
  {
    case PoseWorkflow::BlazePose:
      p0 = mid(23, 24); p1 = mid(11, 12); scale = 1.25f; // hips->shoulders
      break;
    case PoseWorkflow::RTMPose_COCO:
    case PoseWorkflow::RTMPose_Whole:
    case PoseWorkflow::ViTPose:
    case PoseWorkflow::YOLOPose:
      // top-down (axis-aligned) crop; no rotation, model aspect handled later
      rotated = false; scale = 1.25f; square = false;
      break;
    case PoseWorkflow::MediaPipeHands:
      p0 = kp(0); p1 = kp(9); scale = 2.0f; // wrist->middle MCP
      break;
    case PoseWorkflow::FaceMesh:
      // eye corners (FaceMesh indices); keeps the face roughly upright
      p0 = kp(33); p1 = kp(263); scale = 1.5f; target_deg = 0.f;
      break;
    case PoseWorkflow::MobileFaceNet:
      p0 = kp(36); p1 = kp(45); scale = 1.5f; target_deg = 0.f; // eye corners
      break;
    default:
      rotated = false; scale = 1.3f; square = true;
      break;
  }

  if(rotated)
  {
    const float target = target_deg * float(M_PI) / 180.0f;
    r.angle = target
              - std::atan2(-(float(p1.y() - p0.y())), float(p1.x() - p0.x()));
    const float size = std::max(bw, bh) * scale;
    r.w = size;
    r.h = size;
  }
  else if(square)
  {
    const float size = std::max(bw, bh) * scale;
    r.w = size; r.h = size; r.angle = 0.f;
  }
  else
  {
    // top-down: aspect-fix to the model so the crop isn't stretched
    float sw = bw * scale, sh = bh * scale;
    const float a = float(model_w) / float(model_h);
    if(sw > sh * a)
      sh = sw / a;
    else
      sw = sh * a;
    r.w = sw; r.h = sh; r.angle = 0.f;
  }
  return r;
}

Onnx::ROI::Rect PoseDetector::smoothRoi(Onnx::ROI::Rect r)
{
  if(!inputs.track_roi.value && !inputs.smoothing.value)
    return r;
  // ROI is smoothed a bit more aggressively than keypoints (it should be very
  // stable for a still subject); reuse the smoothing amount, biased lower.
  const float amt
      = std::clamp(static_cast<float>(inputs.smoothing_amount.value), 0.f, 1.f);
  const float min_cutoff = 15.0f * std::pow(0.005f / 15.0f, amt);
  m_roi_smoother.configure(min_cutoff, 0.01f);
  float v[5] = {r.cx, r.cy, r.w, r.h, r.angle};
  m_roi_smoother.smooth(v, 1.0f);
  return Onnx::ROI::Rect{v[0], v[1], v[2], v[3], v[4]};
}

std::vector<Onnx::Detection::Detection>
PoseDetector::runDetector(const Onnx::ModelRole& role, const QImage& src)
{
  if(!this->det_ctx)
    return {};

  auto& dctx = *this->det_ctx;
  auto spec = dctx.readModelSpec();
  if(spec.inputs.empty() || spec.inputs[0].shape.size() != 4)
    return {};

  const int model = role.input_w > 0 ? role.input_w : 128;

  // The model's declared anchor count (from the box output's second dim).
  int num_boxes = 0;
  for(const auto& o : spec.outputs)
    if(o.shape.size() == 3
       && (o.shape.back() == 12 || o.shape.back() == 16 || o.shape.back() == 18))
      num_boxes = static_cast<int>(o.shape[1]);

  // Map detections from the letterboxed model square back to image-normalized
  // [0,1] coordinates (works for both centered and top-left letterboxes).
  const float iw = src.width(), ih = src.height();
  auto removeLetterbox
      = [&](std::vector<Onnx::Detection::Detection>& dets,
            const Onnx::ROI::LetterboxResult& lb) {
          auto fix = [&](float nx, float ny, float& ox, float& oy) {
            ox = ((nx * model - lb.pad_x) / lb.scale) / iw;
            oy = ((ny * model - lb.pad_y) / lb.scale) / ih;
          };
          for(auto& d : dets)
          {
            fix(d.xc, d.yc, d.xc, d.yc);
            d.w = (d.w * model / lb.scale) / iw;
            d.h = (d.h * model / lb.scale) / ih;
            for(auto& k : d.keypoints)
            {
              float ox, oy;
              fix(static_cast<float>(k.x()), static_cast<float>(k.y()), ox, oy);
              k = QPointF(ox, oy);
            }
          }
        };

  // --- End2end person/hand detector (YOLOX / RTMDet): BGR NCHW, top-left
  // letterbox, dets+labels output. ---
  if(role.kind == Onnx::ModelKind::PersonDetector)
  {
    auto lb = Onnx::ROI::letterbox(src, model, model, /*center=*/false);
    // Heuristic: small input (<=320) -> RTMDet (mean/std); else YOLOX (raw).
    std::array<float, 3> mean_bgr{0, 0, 0}, std_bgr{1, 1, 1};
    if(model <= 320)
    {
      mean_bgr = {103.53f, 116.28f, 123.675f};
      std_bgr = {57.375f, 57.12f, 58.395f};
    }
    Ort::Value input_value{nullptr};
    {
      auto t = nchwBgrDetectorTensor(
          spec.inputs[0], lb.img, model, model, det_storage, mean_bgr, std_bgr);
      input_value = std::move(t.value);
      std::swap(det_storage, t.storage);
    }
    Ort::Value ins[1] = {std::move(input_value)};
    Ort::Value outs[4]{
        Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
        Ort::Value{nullptr}};
    const size_t n_out = std::min<size_t>(4, spec.output_names_char.size());
    dctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));

    auto dets = Onnx::Detection::decodeEnd2End(
        std::span<Ort::Value>(outs, n_out), model, /*keep_label=*/0, 0.3f);
    removeLetterbox(dets, lb);
    return dets;
  }

  // --- SSD-anchor detectors (BlazePose / palm / BlazeFace) ---
  // Candidate anchor configs for this family; pick the one whose generated
  // anchor count matches the model so ordering stays aligned.
  std::vector<Onnx::Detection::SsdParams> candidates;
  float a = 1.f, b = 0.f; // pixel mapping out = px*a + b
  switch(role.kind)
  {
    case Onnx::ModelKind::BlazePoseDetector:
      candidates = {
          Onnx::Detection::blazePoseParams(model),
          Onnx::Detection::blazePoseParams(128),
          Onnx::Detection::blazePoseParams(224)};
      a = 2.f; b = -1.f; // [-1,1]
      break;
    case Onnx::ModelKind::PalmDetector:
      candidates = {Onnx::Detection::palmParams(model)};
      a = 1.f; b = 0.f; // [0,1]
      break;
    case Onnx::ModelKind::BlazeFaceDetector:
      candidates = {Onnx::Detection::blazeFaceParams(model)};
      a = 2.f; b = -1.f; // [-1,1]
      break;
    default:
      return {};
  }

  Onnx::Detection::SsdParams params = candidates.front();
  for(auto& c : candidates)
    if(num_boxes > 0
       && static_cast<int>(Onnx::Detection::generateAnchors(c).size())
              == num_boxes)
    {
      params = c;
      break;
    }
  params.input_size = model;

  auto lb = Onnx::ROI::letterbox(src, model, model, /*center=*/true);

  Ort::Value input_value{nullptr};
  {
    // Detectors are usually NHWC, but some PINTO exports are NCHW.
    if(role.nhwc)
    {
      auto t = nhwcDetectorTensor(
          spec.inputs[0], lb.img, model, model, det_storage, a, b);
      input_value = std::move(t.value);
      std::swap(det_storage, t.storage);
    }
    else
    {
      // out = px/255*a + b  <=>  (px - mean)/std with std=255/a, mean=-b*std
      const float std_v = 255.f / a;
      const float mean_v = -b * std_v;
      auto t = Onnx::nchw_tensorFromRGBA(
          spec.inputs[0], lb.img.constBits(), model, model, model, model,
          det_storage, {mean_v, mean_v, mean_v}, {std_v, std_v, std_v});
      input_value = std::move(t.value);
      std::swap(det_storage, t.storage);
    }
  }

  Ort::Value ins[1] = {std::move(input_value)};
  Ort::Value outs[6]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}};
  const size_t n_out = std::min<size_t>(6, spec.output_names_char.size());
  dctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));

  auto dets = Onnx::Detection::decode(
      std::span<Ort::Value>(outs, n_out), params, 0.5f);
  dets = Onnx::Detection::nms(std::move(dets), 0.3f);
  removeLetterbox(dets, lb);
  return dets;
}

void PoseDetector::runLandmark(
    const Onnx::ModelRole& role, PoseWorkflow draw, const QImage& src,
    const QTransform& M)
{
  auto& lctx = *this->ctx;
  auto spec = lctx.readModelSpec();
  if(spec.inputs.empty())
  {
    passthrough(src);
    return;
  }

  int mw = role.input_w > 0 ? role.input_w : 256;
  int mh = role.input_h > 0 ? role.input_h : 256;
  if(spec.inputs[0].shape.size() == 4)
  {
    if(role.nhwc)
    {
      mh = static_cast<int>(spec.inputs[0].shape[1]);
      mw = static_cast<int>(spec.inputs[0].shape[2]);
    }
    else
    {
      mh = static_cast<int>(spec.inputs[0].shape[2]);
      mw = static_cast<int>(spec.inputs[0].shape[3]);
    }
  }

  QImage crop = Onnx::ROI::warpCrop(src, M, mw, mh);

  // Build the input tensor (layout + normalization by role).
  std::optional<Onnx::FloatTensor> t;
  if(role.nhwc)
  {
    t.emplace(Onnx::nhwc_rgb_tensorFromRGBA(
        spec.inputs[0], crop.constBits(), mw, mh, mw, mh, storage));
  }
  else if(role.kind == Onnx::ModelKind::MobileFaceNet)
  {
    t.emplace(Onnx::nchw_tensorFromRGBA(
        spec.inputs[0], crop.constBits(), mw, mh, mw, mh, storage,
        {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f},
        {0.229f * 255.f, 0.224f * 255.f, 0.225f * 255.f}));
  }
  else
  {
    t.emplace(Onnx::nchw_tensorFromRGBA(
        spec.inputs[0], crop.constBits(), mw, mh, mw, mh, storage,
        {123.675f, 116.28f, 103.53f}, {58.395f, 57.12f, 57.375f}));
  }

  Ort::Value outs[5]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}};
  const size_t n_out = std::min<size_t>(5, spec.output_names_char.size());

  // Some RTMPose exports take a second [w,h] bbox input.
  std::array<int64_t, 2> bbox_wh{mw, mh};
  std::array<int64_t, 2> bbox_shape{1, 2};
  if(spec.inputs.size() >= 2)
  {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto bbox_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, bbox_wh.data(), bbox_wh.size(), bbox_shape.data(),
        bbox_shape.size());
    Ort::Value ins[2] = {std::move(t->value), std::move(bbox_tensor)};
    lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
  }
  else
  {
    Ort::Value ins[1] = {std::move(t->value)};
    lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
  }

  auto outspan = std::span<Ort::Value>(outs, n_out);

  // Decode keypoints in MODEL-PIXEL space (x,y in [0,mw]x[0,mh]).
  struct MKP
  {
    float x, y, z, conf;
  };
  std::vector<MKP> kps;

  switch(role.kind)
  {
    case Onnx::ModelKind::BlazePoseLandmark:
    {
      // Handle every variant: full-body 195 (=39*5), upper-body 155 (=31*5)
      // or 124 (=31*4). Find the landmark vector output and derive the layout.
      const float* data = nullptr;
      int total = 0;
      for(size_t i = 0; i < outspan.size(); ++i)
      {
        const int64_t n
            = outspan[i].GetTensorTypeAndShapeInfo().GetElementCount();
        if(n == 195 || n == 155 || n == 124)
        {
          data = outspan[i].GetTensorData<float>();
          total = static_cast<int>(n);
          break;
        }
      }
      if(data)
      {
        const int stride = (total % 5 == 0) ? 5 : 4; // x,y,z,vis[,pres]
        const int K = total / stride;
        const int body = std::min(K, 33); // first K are body, rest are aux
        kps.reserve(body);
        for(int i = 0; i < body; ++i)
        {
          const float* kp = data + i * stride;
          const float vis = 1.0f / (1.0f + std::exp(-kp[3]));
          const float pres
              = (stride == 5) ? 1.0f / (1.0f + std::exp(-kp[4])) : vis;
          kps.push_back({kp[0], kp[1], kp[2] / mw, pres});
        }
      }
      break;
    }
    case Onnx::ModelKind::HandLandmark:
    {
      std::optional<Onnx::MediaPipeHands::HandResult> r;
      Onnx::MediaPipeHands::processOutput(spec, outspan, r);
      if(r)
      {
        kps.reserve(r->landmarks.size());
        for(const auto& lm : r->landmarks)
          kps.push_back({lm.x * mw, lm.y * mh, lm.z, r->hand_flag});
      }
      break;
    }
    case Onnx::ModelKind::FaceMeshLandmark:
    {
      std::optional<Onnx::FaceMesh::FaceMeshResult> r;
      Onnx::FaceMesh::processOutput(
          spec, outspan, Onnx::FaceMesh::NUM_LANDMARKS, r);
      if(r)
      {
        kps.reserve(r->landmarks.size());
        for(const auto& lm : r->landmarks)
          kps.push_back({lm.x * mw, lm.y * mh, lm.z, 1.0f});
      }
      break;
    }
    case Onnx::ModelKind::SimccPose:
    {
      Onnx::RTMPose::OutputFormat fmt;
      auto config = Onnx::RTMPose::detectConfig(outspan, &fmt);
      config.input_width = mw;
      config.input_height = mh;
      std::optional<Onnx::RTMPose::PoseResult> r;
      Onnx::RTMPose::processOutput(spec, outspan, config, r, fmt);
      if(r)
      {
        kps.reserve(r->keypoints.size());
        for(const auto& kp : r->keypoints)
          kps.push_back({kp.x * mw, kp.y * mh, 0.0f, kp.confidence});
      }
      break;
    }
    case Onnx::ModelKind::HeatmapPose:
    {
      if(!outspan.empty())
      {
        auto shape = outspan[0].GetTensorTypeAndShapeInfo().GetShape();
        if(shape.size() == 4)
        {
          const int K = static_cast<int>(shape[1]);
          const int hh = static_cast<int>(shape[2]);
          const int hw = static_cast<int>(shape[3]);
          const float* hmaps = outspan[0].GetTensorData<float>();
          kps.reserve(K);
          for(int k = 0; k < K; ++k)
          {
            const float* hm = hmaps + k * hh * hw;
            int max_idx = 0;
            float max_val = hm[0];
            for(int i = 1; i < hh * hw; ++i)
              if(hm[i] > max_val)
              {
                max_val = hm[i];
                max_idx = i;
              }
            const int hx = max_idx % hw;
            const int hy = max_idx / hw;
            float dx = 0.f, dy = 0.f;
            if(hx > 0 && hx < hw - 1)
              dx = 0.25f * (hm[hy * hw + hx + 1] - hm[hy * hw + hx - 1]);
            if(hy > 0 && hy < hh - 1)
              dy = 0.25f * (hm[(hy + 1) * hw + hx] - hm[(hy - 1) * hw + hx]);
            const float mx = (hx + dx + 0.5f) * mw / hw;
            const float my = (hy + dy + 0.5f) * mh / hh;
            kps.push_back({mx, my, 0.0f, max_val});
          }
        }
      }
      break;
    }
    case Onnx::ModelKind::MobileFaceNet:
    {
      if(!outspan.empty())
      {
        auto shape = outspan[0].GetTensorTypeAndShapeInfo().GetShape();
        if(shape.size() >= 2)
        {
          const int n = static_cast<int>(shape[1]);
          const float* data = outspan[0].GetTensorData<float>();
          kps.reserve(n);
          for(int i = 0; i < n; ++i)
            kps.push_back({data[i * 2] * mw, data[i * 2 + 1] * mh, 0.0f, 1.0f});
        }
      }
      break;
    }
    default:
      break;
  }

  if(t)
    std::swap(storage, t->storage);

  if(kps.empty())
  {
    passthrough(src);
    return;
  }

  // Map model-pixel keypoints back through M -> image-normalized [0,1].
  const float iw = src.width(), ih = src.height();
  DetectedPose detected;
  detected.keypoints.reserve(kps.size());
  float sum_conf = 0.0f;
  for(const auto& k : kps)
  {
    const QPointF p = M.map(QPointF(k.x, k.y));
    detected.keypoints.push_back(
        {static_cast<float>(p.x() / iw), static_cast<float>(p.y() / ih), k.z,
         k.conf});
    sum_conf += k.conf;
  }
  detected.mean_confidence = sum_conf / detected.keypoints.size();
  applySmoothing(detected);
  outputs.detection.value = std::move(detected);

  drawSkeleton(*outputs.detection.value, draw);
  generateGeometryOutput(*outputs.detection.value, draw);
}

void PoseDetector::runDetectorAsPose(
    const Onnx::ModelRole& role, const QImage& src)
{
  auto dets = runDetector(role, src);
  if(dets.empty())
  {
    passthrough(src);
    return;
  }

  const auto& d = dets.front();
  if(ossia::safe_isnan(d.score))
  {
    passthrough(src);
    return;
  }

  DetectedPose detected;
  detected.keypoints.reserve(d.keypoints.size());
  for(const auto& k : d.keypoints)
    detected.keypoints.push_back(
        {static_cast<float>(k.x()), static_cast<float>(k.y()), 0.0f, d.score});
  detected.mean_confidence = d.score;
  applySmoothing(detected);
  outputs.detection.value = std::move(detected);

  const PoseWorkflow draw = workflowForRole(role);
  drawSkeleton(*outputs.detection.value, draw);
  generateGeometryOutput(*outputs.detection.value, draw);
}

void PoseDetector::runYOLOPose(const QImage& src, const QTransform& M)
{
  auto& lctx = *this->ctx;
  auto spec = lctx.readModelSpec();

  int model_size = 640;
  if(!spec.inputs.empty() && spec.inputs[0].shape.size() == 4)
    model_size = static_cast<int>(spec.inputs[0].shape[2]);

  auto t = Onnx::nchw_tensorFromRGBA(
      spec.inputs[0], src.constBits(), src.width(), src.height(), model_size,
      model_size, storage, {0.f, 0.f, 0.f}, {255.f, 255.f, 255.f});

  Ort::Value ins[1] = {std::move(t.value)};
  Ort::Value outs[1]{Ort::Value{nullptr}};
  lctx.infer(spec, ins, outs);

  static const Yolo::YOLO_pose yolo_pose;
  std::vector<Yolo::YOLO_pose::pose_type> poses;
  yolo_pose.processOutput(
      spec, outs, poses, 100, inputs.min_confidence, 0, 0, model_size,
      model_size, model_size, model_size);

  if(poses.empty())
  {
    passthrough(src);
    std::swap(storage, t.storage);
    return;
  }

  const float iw = src.width(), ih = src.height();
  const auto& pose = poses[0];
  DetectedPose detected;
  detected.keypoints.assign(17, PoseKeypoint{0.f, 0.f, 0.f, 0.f});
  for(const auto& kp : pose.keypoints)
  {
    if(kp.kp >= 0 && kp.kp < 17)
    {
      const QPointF p = M.map(QPointF(kp.x, kp.y));
      detected.keypoints[kp.kp]
          = {static_cast<float>(p.x() / iw), static_cast<float>(p.y() / ih),
             0.0f, 1.0f};
    }
  }
  detected.mean_confidence = pose.confidence;
  applySmoothing(detected);
  outputs.detection.value = std::move(detected);

  drawSkeleton(*outputs.detection.value, PoseWorkflow::YOLOPose);
  generateGeometryOutput(*outputs.detection.value, PoseWorkflow::YOLOPose);

  std::swap(storage, t.storage);
}

void PoseDetector::runRTMO(const QImage& src)
{
  auto& lctx = *this->ctx;
  auto spec = lctx.readModelSpec();
  if(spec.inputs.empty())
  {
    passthrough(src);
    return;
  }

  // NCHW input; RTMO is 640x640.
  int model = 640;
  if(spec.inputs[0].shape.size() == 4 && spec.inputs[0].shape[3] > 0)
    model = static_cast<int>(spec.inputs[0].shape[3]);

  auto lb = Onnx::ROI::letterbox(src, model, model, /*center=*/false);

  Ort::Value input_value{nullptr};
  {
    // RTMO: raw BGR, no normalization (like YOLOX).
    auto t = nchwBgrDetectorTensor(
        spec.inputs[0], lb.img, model, model, storage, {0, 0, 0}, {1, 1, 1});
    input_value = std::move(t.value);
    std::swap(storage, t.storage);
  }

  Ort::Value ins[1] = {std::move(input_value)};
  Ort::Value outs[4]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}};
  const size_t n_out = std::min<size_t>(4, spec.output_names_char.size());
  lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));

  // dets [1,N,5] (xyxy+score), keypoints [1,N,K,3] (x,y,score) in model px.
  const float* dets = nullptr;
  const float* kpt = nullptr;
  int N = 0, K = 0;
  for(size_t i = 0; i < n_out; ++i)
  {
    if(!outs[i].IsTensor())
      continue;
    auto sh = outs[i].GetTensorTypeAndShapeInfo().GetShape();
    if(sh.size() == 3 && sh[2] == 5)
    {
      dets = outs[i].GetTensorData<float>();
      N = static_cast<int>(sh[1]);
    }
    else if(sh.size() == 4)
    {
      kpt = outs[i].GetTensorData<float>();
      N = static_cast<int>(sh[1]);
      K = static_cast<int>(sh[2]);
    }
  }
  if(!dets || !kpt || K == 0)
  {
    passthrough(src);
    return;
  }

  int best = -1;
  float bestc = std::max(0.3f, static_cast<float>(inputs.min_confidence));
  for(int i = 0; i < N; ++i)
    if(dets[i * 5 + 4] > bestc)
    {
      bestc = dets[i * 5 + 4];
      best = i;
    }
  if(best < 0)
  {
    passthrough(src);
    return;
  }

  const float iw = src.width(), ih = src.height();
  DetectedPose detected;
  detected.keypoints.reserve(K);
  for(int k = 0; k < K; ++k)
  {
    const float* p = kpt + (static_cast<size_t>(best) * K + k) * 3;
    detected.keypoints.push_back(
        {((p[0] - lb.pad_x) / lb.scale) / iw,
         ((p[1] - lb.pad_y) / lb.scale) / ih, 0.0f, p[2]});
  }
  detected.mean_confidence = bestc;
  applySmoothing(detected);
  outputs.detection.value = std::move(detected);

  drawSkeleton(*outputs.detection.value, PoseWorkflow::YOLOPose);
  generateGeometryOutput(*outputs.detection.value, PoseWorkflow::YOLOPose);
}

void PoseDetector::operator()()
try
{
  if(!available)
    return;
  if(this->inputs.model.current_model_invalid)
    return;

  auto& in_tex = inputs.image.texture;
  if(!in_tex.changed)
    return;
  if(!in_tex.bytes)
    return;

  const bool have_det = !inputs.det_model.current_model_invalid
                        && inputs.det_model.file.bytes.size() >= 32;

  // Reset contexts on workflow / model change.
  const PoseWorkflow wf = inputs.workflow.value;
  bool reinit = false;
  if(wf != m_last_workflow)
  {
    ctx.reset();
    det_ctx.reset();
    m_last_workflow = wf;
    reinit = true;
  }
  if(inputs.model.file.filename != m_last_model)
  {
    ctx.reset();
    m_last_model = std::string(inputs.model.file.filename);
    reinit = true;
  }
  if(inputs.det_model.file.filename != m_last_det_model)
  {
    det_ctx.reset();
    m_last_det_model = std::string(inputs.det_model.file.filename);
    reinit = true;
  }
  // A model/workflow change invalidates the temporal tracking/smoothing state.
  if(reinit)
  {
    m_tracking = false;
    m_last_keypoints.clear();
    m_roi_smoother.reset();
    m_smoother.reset();
    m_lost_frames = 0;
  }

  if(!this->ctx)
  {
    this->ctx
        = std::make_unique<Onnx::OnnxRunContext>(this->inputs.model.file.bytes);
    m_landmark_role = Onnx::classify(toModelIO(this->ctx->readModelSpec()));
  }
  if(have_det && !this->det_ctx)
  {
    this->det_ctx = std::make_unique<Onnx::OnnxRunContext>(
        this->inputs.det_model.file.bytes);
    m_detector_role = Onnx::classify(toModelIO(this->det_ctx->readModelSpec()));
  }

  const Onnx::ModelRole role
      = (wf == PoseWorkflow::Auto) ? m_landmark_role : roleForWorkflow(wf);
  const PoseWorkflow draw
      = (wf == PoseWorkflow::Auto) ? workflowForRole(role) : wf;

  QImage src(
      reinterpret_cast<const uchar*>(in_tex.bytes), in_tex.width,
      in_tex.height, QImage::Format_RGBA8888);

  // --- Two-stage: detector + landmark ---
  if(have_det && role.stage == Onnx::ModelStage::Landmark)
  {
    const int mw = role.input_w > 0 ? role.input_w : 256;
    const int mh = role.input_h > 0 ? role.input_h : 256;

    // --- ROI: tracking loop (skip detector) vs fresh detection -------------
    Onnx::ROI::Rect rect;
    if(inputs.track_roi.value && m_tracking && !m_last_keypoints.empty())
    {
      // Derive the ROI from last frame's landmarks — no detector this frame.
      rect = roiRectFromKeypoints(
          draw, m_last_keypoints, in_tex.width, in_tex.height, mw, mh);
    }
    else
    {
      auto dets = runDetector(m_detector_role, src);
      if(dets.empty())
      {
        passthrough(src);
        return;
      }
      rect = detectionRect(role, dets.front(), in_tex.width, in_tex.height);
      m_roi_smoother.reset(); // fresh acquisition: don't blend across the gap
    }

    // Stabilize the crop so a still subject yields a still ROI.
    rect = smoothRoi(rect);
    const QTransform M = Onnx::ROI::rectToTransform(rect, mw, mh);
    runLandmark(role, draw, src, M);

    // --- Tracking gate: keep tracking only if the landmark model is confident
    if(inputs.track_roi.value && outputs.detection.value
       && outputs.detection.value->mean_confidence
              >= std::max(0.2f, static_cast<float>(inputs.min_confidence)))
    {
      m_tracking = true;
      m_last_keypoints = outputs.detection.value->keypoints;
    }
    else
    {
      m_tracking = false; // lost -> re-detect next frame
      m_last_keypoints.clear();
    }
    return;
  }

  // --- Single model ---
  switch(role.stage)
  {
    case Onnx::ModelStage::Detector:
      runDetectorAsPose(role, src);
      break;
    case Onnx::ModelStage::SingleStage:
    {
      if(role.kind == Onnx::ModelKind::RtmoPose)
      {
        runRTMO(src);
      }
      else
      {
        const int ms = role.input_w > 0 ? role.input_w : 640;
        const QTransform M = Onnx::ROI::wholeFrameTransform(
            in_tex.width, in_tex.height, ms, ms);
        runYOLOPose(src, M);
      }
      break;
    }
    case Onnx::ModelStage::Landmark:
    default:
    {
      const int mw = role.input_w > 0 ? role.input_w : 256;
      const int mh = role.input_h > 0 ? role.input_h : 256;
      const QTransform M = Onnx::ROI::wholeFrameTransform(
          in_tex.width, in_tex.height, mw, mh);
      runLandmark(role, draw, src, M);
      break;
    }
  }
}
catch(...)
{
  inputs.model.current_model_invalid = true;
}

} // namespace OnnxModels
