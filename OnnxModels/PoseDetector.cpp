#include "PoseDetector.hpp"

#include <ossia/math/safe_math.hpp>

#include <QImage>
#include <QTransform>

#include <Onnx/helpers/CtxOverlay.hpp>

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
#include <utility>

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

// AP10K / APT36K quadruped, 17 keypoints:
// 0 L-eye 1 R-eye 2 nose 3 neck 4 tail-root
// 5 L-shoulder 6 L-elbow 7 L-front-paw  8 R-shoulder 9 R-elbow 10 R-front-paw
// 11 L-hip 12 L-knee 13 L-back-paw  14 R-hip 15 R-knee 16 R-back-paw
static constexpr std::array<std::pair<int, int>, 17> ap10k = {{
    {0, 1}, {0, 2}, {1, 2}, {2, 3},          // head
    {3, 4},                                   // spine (neck -> tail root)
    {3, 5}, {5, 6}, {6, 7},                   // left front leg
    {3, 8}, {8, 9}, {9, 10},                  // right front leg
    {4, 11}, {11, 12}, {12, 13},              // left back leg
    {4, 14}, {14, 15}, {15, 16},              // right back leg
}};

// clang-format on
} // namespace Skeletons

// Color schemes for different body parts (float RGBA in [0,1]).
using Onnx::Rgba;
namespace Colors
{
static const Rgba head = Onnx::rgb8(255, 255, 0);      // Yellow
static const Rgba torso = Onnx::rgb8(255, 255, 255);   // White
static const Rgba left_arm = Onnx::rgb8(0, 255, 255);  // Cyan
static const Rgba right_arm = Onnx::rgb8(255, 0, 255); // Magenta
static const Rgba left_leg = Onnx::rgb8(0, 255, 0);    // Green
static const Rgba right_leg = Onnx::rgb8(255, 128, 0); // Orange
static const Rgba face = Onnx::rgb8(200, 200, 255);    // Light blue
} // namespace Colors

PoseDetector::PoseDetector() noexcept
{
  // inputs.image.request_width = 256;
  // inputs.image.request_height = 256;
}

PoseDetector::~PoseDetector() = default;

// Get body part color based on keypoint index for COCO format
static Rgba getCOCOColor(int idx)
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
static Rgba getBlazePoseColor(int idx)
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
static Rgba getWholeBodyColor(int idx)
{
  if(idx <= 16)
    return getCOCOColor(idx); // Body keypoints same as COCO
  else if(idx <= 22)
    return Onnx::lighter(Colors::left_leg, 120); // Feet
  else if(idx <= 90)
    return Colors::face; // Face landmarks
  else if(idx <= 111)
    return Onnx::lighter(Colors::left_arm, 120); // Left hand
  else
    return Onnx::lighter(Colors::right_arm, 120); // Right hand
}

// Get finger color for hand keypoints
static Rgba getHandColor(int idx)
{
  if (idx <= 4)
    return Onnx::rgb8(255, 100, 100); // Thumb - red
  else if (idx <= 8)
    return Onnx::rgb8(100, 255, 100); // Index - green
  else if (idx <= 12)
    return Onnx::rgb8(100, 100, 255); // Middle - blue
  else if (idx <= 16)
    return Onnx::rgb8(255, 255, 100); // Ring - yellow
  else
    return Onnx::rgb8(255, 100, 255); // Pinky - magenta
}

// Get color for BlazeFace keypoints
static Rgba getBlazeFaceColor(int idx)
{
  switch(idx)
  {
    case 0: return Onnx::rgb8(255, 0, 0);    // right_eye - red
    case 1: return Onnx::rgb8(0, 0, 255);    // left_eye - blue
    case 2: return Onnx::rgb8(0, 255, 0);    // nose - green
    case 3: return Onnx::rgb8(255, 255, 0);  // mouth - yellow
    case 4: return Onnx::rgb8(255, 0, 255);  // right_ear - magenta
    case 5: return Onnx::rgb8(0, 255, 255);  // left_ear - cyan
    default: return Colors::face;
  }
}

// Get color for dlib 68 face landmarks
static Rgba getDlib68Color(int idx)
{
  if(idx <= 16)
    return Onnx::rgb8(200, 200, 200);  // Jaw - gray
  else if(idx <= 21)
    return Onnx::rgb8(255, 200, 100);  // Left eyebrow - orange
  else if(idx <= 26)
    return Onnx::rgb8(255, 200, 100);  // Right eyebrow - orange
  else if(idx <= 35)
    return Onnx::rgb8(0, 255, 0);      // Nose - green
  else if(idx <= 41)
    return Onnx::rgb8(0, 255, 255);    // Left eye - cyan
  else if(idx <= 47)
    return Onnx::rgb8(0, 255, 255);    // Right eye - cyan
  else if(idx <= 59)
    return Onnx::rgb8(255, 100, 100);  // Outer lip - pink
  else
    return Onnx::rgb8(255, 50, 50);    // Inner lip - red
}

// AP10K animal: head + 4 limbs colored like the human limbs
static Rgba getAP10KColor(int idx)
{
  if(idx <= 3) return Colors::head;             // eyes/nose/neck
  if(idx == 4) return Colors::torso;            // tail root
  if(idx <= 7) return Colors::left_arm;         // left front leg
  if(idx <= 10) return Colors::right_arm;       // right front leg
  if(idx <= 13) return Colors::left_leg;        // left back leg
  return Colors::right_leg;                      // right back leg
}

// Stable, well-distributed color for a track id. Golden-ratio hue stepping so
// consecutive ids are maximally distinct, and a given id is ALWAYS the same
// color (id 1 -> the same hue every frame, every session).
static Rgba getTrackColor(int id)
{
  if(id < 0)
    return Colors::torso;
  constexpr double golden = 0.618033988749895;
  const double hue = std::fmod(0.11 + static_cast<double>(id) * golden, 1.0);
  return Onnx::hsv(static_cast<float>(hue), 0.85f, 1.0f);
}

// NaN/Inf-robust finiteness check. Uses ossia's bit-pattern variants because
// std::isnan/isinf are compiled away under -ffast-math / -ffinite-math-only.
static inline bool finitef(float v) noexcept
{
  return !ossia::safe_isnan(v) && !ossia::safe_isinf(v);
}

// Fill pose.box (top-left normalized form) from the confident keypoints' bbox,
// unless it is already set (box-only detections set it from the detector).
static void fillBoxFromKeypoints(DetectedPose& pose)
{
  if(pose.box.w > 0.f && pose.box.h > 0.f)
    return;
  float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
  int n = 0;
  for(const auto& k : pose.keypoints)
  {
    if(k.confidence < 0.2f)
      continue;
    ++n;
    minx = std::min(minx, k.x); maxx = std::max(maxx, k.x);
    miny = std::min(miny, k.y); maxy = std::max(maxy, k.y);
  }
  if(n < 1 || maxx <= minx)
    return;
  pose.box = {minx, miny, maxx - minx, maxy - miny};
}

// Draw one pose's connections + points with an already-open painter (the output
// image / compositing is owned by the caller — drawSkeleton or drawAllSkeletons).
void PoseDetector::drawOnePose(
    Overlay& ov, const DetectedPose& pose, PoseWorkflow workflow, int w, int h)
{
  const float min_conf = inputs.min_confidence;
  const bool draw_lines = inputs.draw_skeleton.value;

  const auto& kps = pose.keypoints;
  const int num_kps = static_cast<int>(kps.size());

  // Helper to convert keypoint to pixel coordinates
  auto toPoint = [&](int idx) -> QPointF {
    return QPointF(kps[idx].x * w, kps[idx].y * h);
  };

  // Helper to safely set alpha (clamp to valid range)
  auto safeAlpha
      = [](float conf) -> float { return std::clamp(conf, 0.5f, 1.0f); };

  // When this pose carries a persistent track id, draw the whole individual in
  // one stable per-id color (id 1 always the same color, etc.). Otherwise fall
  // back to the per-keypoint-type palette below.
  const bool use_track_color = pose.track_id >= 0;
  const Rgba track_color
      = use_track_color ? getTrackColor(pose.track_id) : Rgba{};

  // Select skeleton connections and color function based on workflow
  auto getColor = [&](int idx) -> Rgba {
    if(use_track_color)
      return track_color;
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
      case PoseWorkflow::RTMPoseFace:
        return Colors::face;
      case PoseWorkflow::AnimalPose:
        return getAP10KColor(idx);
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

        Rgba color = Onnx::withAlpha(getColor(from), safeAlpha(conf));
        ov.lineWidth(2.f);
        ov.color(color);
        ov.line(toPoint(from), toPoint(to));
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
          // `closed` contours (oval/eyes/lips) wrap the last point back to the
          // first; open ones (eyebrows) stop at the last segment.
          auto drawContour = [&](const auto& indices, Rgba color, bool closed) {
            color = Onnx::withAlpha(color, 0.8f);
            ov.color(color);
            ov.lineWidth(1.f);
            const size_t n = indices.size();
            for(size_t i = 0; i < n; ++i)
            {
              if(!closed && i + 1 >= n)
                break;
              int from = indices[i];
              int to = indices[(i + 1) % n];
              if(from < 0 || to < 0)
                continue;
              if(from >= num_kps || to >= num_kps)
                continue;
              ov.line(toPoint(from), toPoint(to));
            }
          };
          auto cc = [&](Rgba base) { return use_track_color ? track_color : base; };
          drawContour(Onnx::FaceMesh::FaceContours::face_oval, cc(Onnx::rgb8(200, 200, 255)), true);
          drawContour(Onnx::FaceMesh::FaceContours::left_eye, cc(Onnx::rgb8(0, 255, 255)), true);
          drawContour(Onnx::FaceMesh::FaceContours::right_eye, cc(Onnx::rgb8(0, 255, 255)), true);
          drawContour(Onnx::FaceMesh::FaceContours::lips_outer, cc(Onnx::rgb8(255, 100, 100)), true);
          drawContour(Onnx::FaceMesh::FaceContours::left_eyebrow, cc(Onnx::rgb8(255, 255, 0)), false);
          drawContour(Onnx::FaceMesh::FaceContours::right_eyebrow, cc(Onnx::rgb8(255, 255, 0)), false);
        }
        break;
      case PoseWorkflow::MobileFaceNet:
        drawConnections(Skeletons::dlib68, 68);
        break;
      case PoseWorkflow::RTMPoseFace:
        // 106 LaPa landmarks: dlib-style contours if it happens to be 68,
        // otherwise the dense points are drawn below (no fixed 106 skeleton).
        if(num_kps == 68)
          drawConnections(Skeletons::dlib68, 68);
        break;
      case PoseWorkflow::AnimalPose:
        drawConnections(Skeletons::ap10k, 17);
        break;
      default:
        drawConnections(Skeletons::coco17, 17);
        break;
    }
  }

  // Draw keypoints (landmark dots), gated independently of the skeleton lines.
  for(int i = 0; inputs.draw_landmarks.value && i < num_kps; ++i)
  {
    float conf = kps[i].confidence;
    if(conf < min_conf)
      continue;

    Rgba color = Onnx::withAlpha(getColor(i), safeAlpha(conf));
    ov.color(color);

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
    else if(workflow == PoseWorkflow::FaceMesh
            || workflow == PoseWorkflow::RTMPoseFace)
    {
      radius = 1; // Very small for dense face landmarks
    }
    else if(workflow == PoseWorkflow::BlazeFace)
    {
      radius = 5; // Larger for sparse face keypoints
    }
    else if(workflow == PoseWorkflow::MobileFaceNet)
    {
      radius = 2; // Small for 68 face landmarks
    }

    ov.fillCircle(toPoint(i), static_cast<float>(radius));
  }

  // Bounding box: always in Box Detection, opt-in elsewhere via Draw Boxes.
  if((workflow == PoseWorkflow::BoxDetection || inputs.draw_boxes.value)
     && pose.box.w > 0.f && pose.box.h > 0.f
     && pose.mean_confidence >= min_conf)
  {
    Rgba bc = Onnx::withAlpha(
        use_track_color ? track_color
                        : getTrackColor(std::max(0, pose.class_id)),
        0.9f);
    ov.color(bc);
    ov.lineWidth(2.f);
    const QRectF r(
        pose.box.x * w, pose.box.y * h, pose.box.w * w, pose.box.h * h);
    ov.strokeRect(r);

    QString lbl;
    if(pose.track_id >= 0)
      lbl += QStringLiteral("#%1 ").arg(pose.track_id);
    if(pose.class_id >= 0)
      lbl += QStringLiteral("c%1 ").arg(pose.class_id);
    lbl += QString::number(pose.mean_confidence, 'f', 2);
    ov.text(14.f, QPointF(r.x() + 2, std::max(10.0, r.y() - 3)), lbl);
  }
}

// Prepare the output texture before drawing: copy the input frame in, or fill
// opaque black for skeleton-only mode. The ctx overlay then draws onto it.
static void fillCanvas(
    unsigned char* dst, const unsigned char* src, int w, int h,
    bool skeleton_only)
{
  const size_t n = static_cast<size_t>(w) * h * 4;
  if(skeleton_only)
  {
    for(size_t i = 0; i < n; i += 4)
    {
      dst[i] = dst[i + 1] = dst[i + 2] = 0;
      dst[i + 3] = 255;
    }
  }
  else
  {
    std::memcpy(dst, src, n);
  }
}

void PoseDetector::drawSkeleton(const DetectedPose& pose, PoseWorkflow workflow)
{
  auto& in_tex = inputs.image.texture;
  const int w = in_tex.width, h = in_tex.height;
  const bool skeleton_only
      = (inputs.output_mode.value == OutputMode::SkeletonOnly);

  outputs.image.create(w, h);
  auto* dst = reinterpret_cast<unsigned char*>(outputs.image.texture.bytes);
  fillCanvas(
      dst, reinterpret_cast<const unsigned char*>(in_tex.bytes), w, h,
      skeleton_only);
  {
    Overlay ov(dst, w, h);
    drawOnePose(ov, pose, workflow, w, h);
  } // Overlay dtor rasterizes the queued drawing into dst
  outputs.image.texture.changed = true;
}

void PoseDetector::drawAllSkeletons(PoseWorkflow workflow)
{
  auto& in_tex = inputs.image.texture;
  const int w = in_tex.width, h = in_tex.height;
  const bool skeleton_only
      = (inputs.output_mode.value == OutputMode::SkeletonOnly);

  outputs.image.create(w, h);
  auto* dst = reinterpret_cast<unsigned char*>(outputs.image.texture.bytes);
  fillCanvas(
      dst, reinterpret_cast<const unsigned char*>(in_tex.bytes), w, h,
      skeleton_only);
  {
    Overlay ov(dst, w, h);
    for(const auto& pose : m_instances)
      drawOnePose(ov, pose, workflow, w, h);
  }
  outputs.image.texture.changed = true;
}

// Append one pose's flattened geometry (current Data Format) to `out`. Does NOT
// clear — the caller owns the buffer (single-pose clears; multi accumulates).
void PoseDetector::appendGeometry(
    std::vector<float>& out, const DetectedPose& pose, PoseWorkflow workflow)
{
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

        case PoseWorkflow::AnimalPose:
          out.reserve(Skeletons::ap10k.size() * 6);
          for (const auto& [from, to] : Skeletons::ap10k)
            addLine(from, to);
          break;

        case PoseWorkflow::RTMPoseFace:
          // 106 LaPa landmarks have no fixed line skeleton; emit dlib lines only
          // when it is the 68-point layout, else leave points-only.
          if (kps.size() == 68)
            for (const auto& [from, to] : Skeletons::dlib68)
              addLine(from, to);
          break;

        case PoseWorkflow::FaceMesh:
        {
          // FaceMesh uses contour indices, generate lines from contours.
          // Closed contours (oval/eyes/lips) wrap back to the first point.
          auto addContour = [&](const auto& indices, bool closed)
          {
            const size_t n = indices.size();
            for (size_t i = 0; i < n; ++i)
            {
              if (!closed && i + 1 >= n)
                break;
              addLine(indices[i], indices[(i + 1) % n]);
            }
          };
          addContour(Onnx::FaceMesh::FaceContours::face_oval, true);
          addContour(Onnx::FaceMesh::FaceContours::left_eye, true);
          addContour(Onnx::FaceMesh::FaceContours::right_eye, true);
          addContour(Onnx::FaceMesh::FaceContours::lips_outer, true);
          addContour(Onnx::FaceMesh::FaceContours::left_eyebrow, false);
          addContour(Onnx::FaceMesh::FaceContours::right_eyebrow, false);
          break;
        }

        case PoseWorkflow::BoxDetection:
          // Box-only: no skeleton; the box is carried in poses_geometry.
          break;

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

void PoseDetector::generateGeometryOutput(
    const DetectedPose& pose, PoseWorkflow workflow)
{
  auto& out = outputs.geometry.value;
  out.clear();
  appendGeometry(out, pose, workflow);
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
      if(r.domain == Onnx::ModelDomain::Animal)
        return PoseWorkflow::AnimalPose;
      if(r.domain == Onnx::ModelDomain::Face)
        return PoseWorkflow::RTMPoseFace;
      return (r.num_keypoints > 50) ? PoseWorkflow::RTMPose_Whole
                                    : PoseWorkflow::RTMPose_COCO;
    case K::HeatmapPose:
      if(r.domain == Onnx::ModelDomain::Animal)
        return PoseWorkflow::AnimalPose;
      return PoseWorkflow::ViTPose;
    case K::YoloPose:
    case K::RtmoPose:
      return PoseWorkflow::YOLOPose;
    case K::BlazeFaceDetector:
      return PoseWorkflow::BlazeFace;
    case K::PersonDetector:
    case K::MultiClassDetector:
    case K::YoloxDetector:
      return PoseWorkflow::BoxDetection;
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
  // One image; force batch to 1 so the tensor element count matches `storage`.
  std::vector<std::int64_t> shape = port.shape;
  if(!shape.empty())
    shape[0] = 1;
  Onnx::FloatTensor f{
      .storage = {},
      .value = Onnx::vec_to_tensor<float>(storage, shape)};
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
  // One image; force batch to 1 so the tensor element count matches `storage`.
  std::vector<std::int64_t> shape = port.shape;
  if(!shape.empty())
    shape[0] = 1;
  Onnx::FloatTensor f{
      .storage = {},
      .value = Onnx::vec_to_tensor<float>(storage, shape)};
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
    case PoseWorkflow::AnimalPose:
      // keep the classified kind (SimccPose or HeatmapPose); just mark animal
      r.stage = S::Landmark; r.domain = D::Animal;
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
    case PoseWorkflow::RTMPoseFace:
      r.kind = K::SimccPose; r.stage = S::Landmark; r.domain = D::Face;
      break;
    case PoseWorkflow::BoxDetection:
      // Detection-only: the relevant model is the Detection Model; the box path
      // dispatches on m_detector_role directly, so leave the landmark role as-is.
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
  outputs.poses.value.clear();
  outputs.poses_geometry.value.clear();
  outputs.count.value = 0;
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

  // One-Euro is a jitter-vs-lag trade governed by TWO params (see the original
  // paper + MediaPipe's LandmarksSmoothingCalculator):
  //   * min_cutoff — the smoothing when the point is (nearly) still. Low = very
  //     smooth at rest. This is what kills jiggle.
  //   * beta       — how fast the cutoff opens up with speed. This is what keeps
  //     motion responsive (low lag). For NORMALIZED [0,1] coords the per-frame
  //     velocity is tiny (~0.01-0.1), so beta must be O(1-10), not 0.02 — that
  //     was why max smoothing felt "too smooth/laggy": no speed adaptivity.
  // Map the single Amount slider onto both: more amount => lower rest-cutoff and
  // (slightly) higher beta so heavy smoothing still tracks fast motion.
  const float min_cutoff = 5.0f * std::pow(0.02f / 5.0f, amt); // 5 -> 0.02 Hz
  const float beta = 1.0f + 4.0f * amt;                        // 1 -> 5

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
      return Onnx::ROI::mediapipeRect(det, W, H, Onnx::ROI::faceParams());
    case Onnx::ModelKind::MobileFaceNet:
      return Onnx::ROI::mediapipeRect(det, W, H, Onnx::ROI::mobileFaceParams());
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
  Onnx::ROI::Rect r{}; // w==0 means "invalid" (caller falls back to detecting)
  if(kps.empty())
    return r;

  auto kp = [&](int i) -> QPointF {
    return (i >= 0 && i < (int)kps.size())
               ? QPointF(kps[i].x * W, kps[i].y * H)
               : QPointF(0, 0);
  };
  auto conf = [&](int i) {
    return (i >= 0 && i < (int)kps.size()) ? kps[i].confidence : 0.f;
  };
  auto mid = [&](int a, int b) {
    return QPointF((kp(a).x() + kp(b).x()) * 0.5, (kp(a).y() + kp(b).y()) * 0.5);
  };

  // bbox of confident keypoints (image px) + count
  int nconf = 0;
  float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
  for(const auto& k : kps)
  {
    if(k.confidence < 0.2f)
      continue;
    ++nconf;
    const float x = k.x * W, y = k.y * H;
    minx = std::min(minx, x); maxx = std::max(maxx, x);
    miny = std::min(miny, y); maxy = std::max(maxy, y);
  }
  if(nconf < 3 || maxx <= minx)
    return r; // too sparse/degenerate -> invalid -> re-detect
  const float bbox_cx = 0.5f * (minx + maxx);
  const float bbox_cy = 0.5f * (miny + maxy);
  const float bw = maxx - minx, bh = maxy - miny;

  switch(draw)
  {
    case PoseWorkflow::BlazePose:
    {
      // Match the DETECTOR's alignment rect: hip-centered, size = 2 x radius to
      // the farthest body point, rotated so hips->shoulders points "up". Using
      // the keypoint bbox center/size instead re-frames the crop and makes the
      // landmark model drift (and can collapse to image center).
      const QPointF hip = mid(23, 24), sh = mid(11, 12);
      float R = 0.f;
      for(int i = 0; i < (int)kps.size(); ++i)
        if(conf(i) >= 0.2f)
          R = std::max(
              R, float(std::hypot(kp(i).x() - hip.x(), kp(i).y() - hip.y())));
      const float size = 2.0f * R * 1.15f;
      const float target = float(M_PI) / 2.0f;
      r.cx = hip.x();
      r.cy = hip.y();
      r.w = r.h = size;
      r.angle = target
                - std::atan2(
                    -(float(sh.y() - hip.y())), float(sh.x() - hip.x()));
      break;
    }
    case PoseWorkflow::MediaPipeHands:
    {
      const QPointF wrist = kp(0), mcp = kp(9);
      const float size = std::max(bw, bh) * 2.0f;
      r.cx = bbox_cx; r.cy = bbox_cy; r.w = r.h = size;
      r.angle = float(M_PI) / 2.0f
                - std::atan2(
                    -(float(mcp.y() - wrist.y())), float(mcp.x() - wrist.x()));
      break;
    }
    case PoseWorkflow::FaceMesh:
    case PoseWorkflow::MobileFaceNet:
    {
      const int e0 = (draw == PoseWorkflow::FaceMesh) ? 33 : 36;
      const int e1 = (draw == PoseWorkflow::FaceMesh) ? 263 : 45;
      const float size = std::max(bw, bh) * 1.5f;
      r.cx = bbox_cx; r.cy = bbox_cy; r.w = r.h = size;
      r.angle = -std::atan2(
          -(float(kp(e1).y() - kp(e0).y())), float(kp(e1).x() - kp(e0).x()));
      break;
    }
    default:
    {
      // top-down (RTMPose/ViTPose/YOLO): axis-aligned bbox, aspect-fixed.
      // Recompute the bbox from only WELL-confident points: wholebody (133)
      // emits many low-confidence face/hand points that, if included, blow up
      // the bbox and send the crop off the subject. Require >=0.35 here.
      float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
      int n = 0;
      for(const auto& k : kps)
      {
        if(k.confidence < 0.35f)
          continue;
        ++n;
        const float x = k.x * W, y = k.y * H;
        bx0 = std::min(bx0, x); bx1 = std::max(bx1, x);
        by0 = std::min(by0, y); by1 = std::max(by1, y);
      }
      if(n < 3 || bx1 <= bx0)
        return Onnx::ROI::Rect{}; // invalid -> re-detect
      float sw = (bx1 - bx0) * 1.25f, sh = (by1 - by0) * 1.25f;
      const float a = float(model_w) / float(model_h);
      if(sw > sh * a) sh = sw / a; else sw = sh * a;
      r.cx = 0.5f * (bx0 + bx1); r.cy = 0.5f * (by0 + by1);
      r.w = sw; r.h = sh; r.angle = 0.f;
      break;
    }
  }
  return r;
}

namespace
{
// A tracking ROI must be finite, non-tiny, and centered inside the frame.
bool rectValid(const Onnx::ROI::Rect& r, int W, int H)
{
  if(!std::isfinite(r.cx) || !std::isfinite(r.cy) || !std::isfinite(r.w)
     || !std::isfinite(r.h) || !std::isfinite(r.angle))
    return false;
  if(r.w < 0.04f * W || r.h < 0.04f * H)
    return false;
  if(r.cx < 0 || r.cx > W || r.cy < 0 || r.cy > H)
    return false;
  return true;
}
// Reject a tracked ROI that teleported or changed size implausibly vs the
// previous one (prevents drift/collapse from compounding — re-detect instead).
bool rectPlausible(const Onnx::ROI::Rect& c, const Onnx::ROI::Rect& p)
{
  const float move = std::hypot(c.cx - p.cx, c.cy - p.cy);
  if(move > 0.6f * std::max(p.w, p.h))
    return false;
  const float ratio = c.w / std::max(1.0f, p.w);
  if(ratio < 0.66f || ratio > 1.5f) // implausible per-frame size jump -> re-detect
    return false;
  return true;
}
// Deadband: a near-identical ROI is treated as unchanged, so a STILL subject
// yields the exact same crop every frame and the keypoints stop oscillating
// (the tracking ROI is a feedback loop; without this it shakes on a static
// image).
bool rectClose(const Onnx::ROI::Rect& c, const Onnx::ROI::Rect& p)
{
  const float sz = std::max(p.w, p.h);
  if(std::hypot(c.cx - p.cx, c.cy - p.cy) > 0.015f * sz)
    return false;
  if(std::fabs(c.w - p.w) > 0.03f * p.w || std::fabs(c.h - p.h) > 0.03f * p.h)
    return false;
  if(std::fabs(c.angle - p.angle) > 0.02f)
    return false;
  return true;
}
} // namespace

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
PoseDetector::runDetector(
    const Onnx::ModelRole& role, const QImage& src, Onnx::ModelDomain target,
    int keep_class)
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

    // -2 (domain default) -> person(0); else the requested class (-1 = all).
    const int keep = (keep_class == -2) ? 0 : keep_class;
    auto dets = Onnx::Detection::decodeEnd2End(
        std::span<Ort::Value>(outs, n_out), model, keep, 0.3f);
    removeLetterbox(dets, lb);
    return dets;
  }

  // --- PINTO multi-class detector ([N,7] batchno,classid,score,xyxy): raw BGR,
  // top-left letterbox. Used as a body/person detector (class 0). ---
  if(role.kind == Onnx::ModelKind::MultiClassDetector)
  {
    const int mw = model;
    const int mh = role.input_h > 0 ? role.input_h : model;
    auto lb = Onnx::ROI::letterbox(src, mw, mh, /*center=*/false);
    Ort::Value input_value{nullptr};
    {
      auto t = nchwBgrDetectorTensor(
          spec.inputs[0], lb.img, mw, mh, det_storage, {0, 0, 0}, {1, 1, 1});
      input_value = std::move(t.value);
      std::swap(det_storage, t.storage);
    }
    Ort::Value ins[1] = {std::move(input_value)};
    Ort::Value outs[2]{Ort::Value{nullptr}, Ort::Value{nullptr}};
    const size_t n_out = std::min<size_t>(2, spec.output_names_char.size());
    dctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));

    const bool sbb = !spec.outputs.empty()
                     && spec.outputs[0].name.contains(
                         "score_x", Qt::CaseInsensitive); // score before box
    const int keep = (keep_class == -2) ? 0 : keep_class;
    auto dets = Onnx::Detection::decodeMultiClass(
        std::span<Ort::Value>(outs, n_out), mw, mh, keep, 0.4f, sbb);
    // remove top-left letterbox (non-square aware)
    for(auto& dd : dets)
    {
      auto fix = [&](float nx, float ny, float& ox, float& oy) {
        ox = ((nx * mw - lb.pad_x) / lb.scale) / iw;
        oy = ((ny * mh - lb.pad_y) / lb.scale) / ih;
      };
      fix(dd.xc, dd.yc, dd.xc, dd.yc);
      dd.w = (dd.w * mw / lb.scale) / iw;
      dd.h = (dd.h * mh / lb.scale) / ih;
    }
    return dets;
  }

  // --- Raw YOLOX COCO grid: BGR no-norm, top-left letterbox, class filter.
  // person=0 for body; COCO animals 14..23 for animal pose. ---
  if(role.kind == Onnx::ModelKind::YoloxDetector)
  {
    const int mw = model;
    const int mh = role.input_h > 0 ? role.input_h : model;
    int cls_lo = 0, cls_hi = 0;            // person
    if(target == Onnx::ModelDomain::Animal) { cls_lo = 14; cls_hi = 23; }
    if(keep_class == -1) { cls_lo = 0; cls_hi = 100000; }      // all classes
    else if(keep_class >= 0) { cls_lo = cls_hi = keep_class; } // one class
    auto lb = Onnx::ROI::letterbox(src, mw, mh, /*center=*/false);
    Ort::Value input_value{nullptr};
    {
      // This PINTO YOLOX-COCO export wants BGR [0,1] (raw 0-255 gives garbage).
      auto t = nchwBgrDetectorTensor(
          spec.inputs[0], lb.img, mw, mh, det_storage, {0, 0, 0},
          {255.f, 255.f, 255.f});
      input_value = std::move(t.value);
      std::swap(det_storage, t.storage);
    }
    Ort::Value ins[1] = {std::move(input_value)};
    Ort::Value outs[2]{Ort::Value{nullptr}, Ort::Value{nullptr}};
    const size_t n_out = std::min<size_t>(2, spec.output_names_char.size());
    dctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
    auto dets = Onnx::Detection::decodeYoloxGrid(
        std::span<Ort::Value>(outs, n_out), mw, mh, cls_lo, cls_hi, 0.3f, 0.45f);
    for(auto& dd : dets)
    {
      auto fix = [&](float nx, float ny, float& ox, float& oy) {
        ox = ((nx * mw - lb.pad_x) / lb.scale) / iw;
        oy = ((ny * mh - lb.pad_y) / lb.scale) / ih;
      };
      fix(dd.xc, dd.yc, dd.xc, dd.yc);
      dd.w = (dd.w * mw / lb.scale) / iw;
      dd.h = (dd.h * mh / lb.scale) / ih;
    }
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

namespace
{
// One decoded landmark in MODEL-PIXEL space (x,y in [0,mw]x[0,mh]).
struct LandmarkKp
{
  float x, y, z, conf;
};

// Build the landmark input tensor for one crop (layout + normalization by role).
Onnx::FloatTensor makeLandmarkInput(
    const Onnx::ModelRole& role, Onnx::ModelSpec::Port& port, const QImage& crop,
    int mw, int mh, boost::container::vector<float>& scratch)
{
  if(role.nhwc)
    return Onnx::nhwc_rgb_tensorFromRGBA(
        port, crop.constBits(), mw, mh, mw, mh, scratch);
  if(role.kind == Onnx::ModelKind::MobileFaceNet)
    return Onnx::nchw_tensorFromRGBA(
        port, crop.constBits(), mw, mh, mw, mh, scratch,
        {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f},
        {0.229f * 255.f, 0.224f * 255.f, 0.225f * 255.f});
  if(role.kind == Onnx::ModelKind::FaceMeshLandmark
     || role.kind == Onnx::ModelKind::HandLandmark
     || role.kind == Onnx::ModelKind::BlazePoseLandmark)
    // MediaPipe landmark models want [0,1] regardless of layout.
    return Onnx::nchw_tensorFromRGBA(
        port, crop.constBits(), mw, mh, mw, mh, scratch, {0.f, 0.f, 0.f},
        {255.f, 255.f, 255.f});
  return Onnx::nchw_tensorFromRGBA(
      port, crop.constBits(), mw, mh, mw, mh, scratch,
      {123.675f, 116.28f, 103.53f}, {58.395f, 57.12f, 57.375f});
}

// Decode ONE instance's landmark outputs (a [1,...] outspan) into MODEL-PIXEL
// keypoints. Shared by the single-crop and batched-slice paths.
void decodeLandmark(
    const Onnx::ModelRole& role, const Onnx::ModelSpec& spec,
    std::span<Ort::Value> outspan, int mw, int mh, float min_conf,
    std::vector<LandmarkKp>& kps)
{
  kps.clear();
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
      Onnx::MediaPipeHands::processOutput(spec, outspan, r, min_conf);
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
          spec, outspan, Onnx::FaceMesh::NUM_LANDMARKS, r, min_conf);
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
        auto info = outspan[0].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        if(shape.size() == 4)
        {
          const int hh = static_cast<int>(shape[2]);
          const int hw = static_cast<int>(shape[3]);
          // hw==0 would divide by zero (max_idx % hw); a declared K larger than
          // the real buffer would over-read -> clamp to actual element count.
          if(hh <= 0 || hw <= 0)
            break;
          const int64_t total = static_cast<int64_t>(info.GetElementCount());
          const int K = static_cast<int>(std::min<int64_t>(
              shape[1] > 0 ? shape[1] : 0, total / (static_cast<int64_t>(hh) * hw)));
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

            // DARK sub-pixel refinement (mmpose's heatmap decode).
            float ox = 0.f, oy = 0.f;
            if(hx >= 1 && hx < hw - 1 && hy >= 1 && hy < hh - 1)
            {
              auto L = [&](int x, int y) {
                return std::log(std::max(hm[y * hw + x], 1e-10f));
              };
              const float dx = 0.5f * (L(hx + 1, hy) - L(hx - 1, hy));
              const float dy = 0.5f * (L(hx, hy + 1) - L(hx, hy - 1));
              const float dxx = L(hx + 1, hy) - 2.f * L(hx, hy) + L(hx - 1, hy);
              const float dyy = L(hx, hy + 1) - 2.f * L(hx, hy) + L(hx, hy - 1);
              const float dxy = 0.25f
                                * (L(hx + 1, hy + 1) - L(hx + 1, hy - 1)
                                   - L(hx - 1, hy + 1) + L(hx - 1, hy - 1));
              const float det = dxx * dyy - dxy * dxy;
              if(std::fabs(det) > 1e-9f)
              {
                ox = std::clamp(-(dyy * dx - dxy * dy) / det, -1.f, 1.f);
                oy = std::clamp(-(dxx * dy - dxy * dx) / det, -1.f, 1.f);
              }
            }
            const float mx = (hx + ox + 0.5f) * mw / hw;
            const float my = (hy + oy + 0.5f) * mh / hh;
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
}
} // namespace

float PoseDetector::landmarkKeypoints(
    const Onnx::ModelRole& role, const QImage& src, const QTransform& M,
    std::vector<PoseKeypoint>& out)
{
  out.clear();
  auto& lctx = *this->ctx;
  auto spec = lctx.readModelSpec();
  if(spec.inputs.empty())
    return -1.f;

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
  Onnx::FloatTensor t
      = makeLandmarkInput(role, spec.inputs[0], crop, mw, mh, storage);

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
    Ort::Value ins[2] = {std::move(t.value), std::move(bbox_tensor)};
    lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
  }
  else
  {
    Ort::Value ins[1] = {std::move(t.value)};
    lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
  }

  auto outspan = std::span<Ort::Value>(outs, n_out);

  std::vector<LandmarkKp> kps;
  decodeLandmark(
      role, spec, outspan, mw, mh, static_cast<float>(inputs.min_confidence),
      kps);

  std::swap(storage, t.storage);

  if(kps.empty())
    return -1.f;

  // Map model-pixel keypoints back through M -> image-normalized [0,1].
  const float iw = src.width(), ih = src.height();
  out.reserve(kps.size());
  float sum_conf = 0.0f;
  for(const auto& k : kps)
  {
    const QPointF p = M.map(QPointF(k.x, k.y));
    out.push_back(
        {static_cast<float>(p.x() / iw), static_cast<float>(p.y() / ih), k.z,
         k.conf});
    sum_conf += k.conf;
  }
  return sum_conf / out.size();
}

// Landmark every ROI, filling m_instances. Batches all crops into ONE
// [N,C,H,W] inference when the model's batch dim is dynamic/>=N (the common
// case); otherwise falls back to one inference per crop. Output decode is
// identical either way (each instance decoded from its own [1,...] slice).
void PoseDetector::runLandmarkBatch(
    const Onnx::ModelRole& role, PoseWorkflow draw, const QImage& src,
    const std::vector<Onnx::ROI::Rect>& rois)
{
  (void)draw;
  m_instances.clear();
  if(rois.empty())
    return;
  auto& lctx = *this->ctx;
  auto spec = lctx.readModelSpec();
  if(spec.inputs.empty())
    return;

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

  const int N = static_cast<int>(rois.size());
  const int64_t batch_dim
      = (spec.inputs[0].shape.size() == 4) ? spec.inputs[0].shape[0] : 1;
  const bool can_batch = N >= 2 && (batch_dim < 0 || batch_dim >= N);
  const float iw = src.width(), ih = src.height();

  auto pushFromKps = [&](const std::vector<LandmarkKp>& kps,
                         const QTransform& M) {
    if(kps.empty())
      return;
    DetectedPose pose;
    pose.keypoints.reserve(kps.size());
    float sum = 0.f;
    for(const auto& k : kps)
    {
      const QPointF p = M.map(QPointF(k.x, k.y));
      pose.keypoints.push_back(
          {static_cast<float>(p.x() / iw), static_cast<float>(p.y() / ih), k.z,
           k.conf});
      sum += k.conf;
    }
    pose.mean_confidence = sum / pose.keypoints.size();
    m_instances.push_back(std::move(pose));
  };

  // Fallback: one inference per ROI (fixed batch dim, or single instance).
  if(!can_batch)
  {
    for(const auto& r : rois)
    {
      const QTransform M = Onnx::ROI::rectToTransform(r, mw, mh);
      const float mc = landmarkKeypoints(role, src, M, m_kp_scratch);
      if(mc < 0.f || m_kp_scratch.empty())
        continue;
      DetectedPose pose;
      pose.keypoints = m_kp_scratch;
      pose.mean_confidence = mc;
      m_instances.push_back(std::move(pose));
    }
    return;
  }

  // --- Batched: pack N crops into one [N,C,H,W] input buffer. ---
  const int CHW = 3 * mw * mh;
  m_batch_storage.resize(
      static_cast<size_t>(N) * CHW, boost::container::default_init);
  for(int b = 0; b < N; ++b)
  {
    const QTransform M = Onnx::ROI::rectToTransform(rois[b], mw, mh);
    QImage crop = Onnx::ROI::warpCrop(src, M, mw, mh);
    Onnx::FloatTensor ft
        = makeLandmarkInput(role, spec.inputs[0], crop, mw, mh, m_tmp_storage);
    std::memcpy(
        m_batch_storage.data() + static_cast<size_t>(b) * CHW,
        ft.storage.data(), static_cast<size_t>(CHW) * sizeof(float));
    std::swap(m_tmp_storage, ft.storage); // reclaim capacity for next crop
  }

  std::vector<int64_t> in_shape = spec.inputs[0].shape;
  if(in_shape.size() == 4)
    in_shape[0] = N;
  else
    in_shape = {N, 3, mh, mw};
  Ort::Value in0 = Onnx::vec_to_tensor<float>(m_batch_storage, in_shape);

  Ort::Value outs[5]{
      Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
      Ort::Value{nullptr}, Ort::Value{nullptr}};
  const size_t n_out = std::min<size_t>(5, spec.output_names_char.size());

  if(spec.inputs.size() >= 2)
  {
    // SimCC's second [w,h] input, batched to [N,2].
    m_bbox.resize(static_cast<size_t>(N) * 2);
    for(int b = 0; b < N; ++b)
    {
      m_bbox[2 * b] = mw;
      m_bbox[2 * b + 1] = mh;
    }
    std::array<int64_t, 2> bshape{N, 2};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto bbox_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, m_bbox.data(), m_bbox.size(), bshape.data(), bshape.size());
    Ort::Value ins[2] = {std::move(in0), std::move(bbox_tensor)};
    lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
  }
  else
  {
    Ort::Value ins[1] = {std::move(in0)};
    lctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
  }

  // Decode each instance from its [1,...] slice of the batched outputs. The
  // decoders see exactly the single-instance shapes they already handle.
  Ort::AllocatorWithDefaultOptions alloc;
  std::vector<LandmarkKp> kps;
  for(int b = 0; b < N; ++b)
  {
    Ort::Value sl[5]{
        Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
        Ort::Value{nullptr}, Ort::Value{nullptr}};
    for(size_t j = 0; j < n_out; ++j)
    {
      auto info = outs[j].GetTensorTypeAndShapeInfo();
      auto shp = info.GetShape();
      const size_t total = info.GetElementCount();
      // Treat as batched only if the model's DECLARED leading dim is the batch
      // axis (dynamic, or == N), it resolved to N, and the buffer divides evenly.
      // This avoids mis-slicing an output whose fixed leading dim coincidentally
      // equals the instance count (e.g. a constant [num_keypoints, ...]).
      const int64_t decl0
          = (j < spec.outputs.size() && !spec.outputs[j].shape.empty())
                ? spec.outputs[j].shape[0]
                : -1;
      const bool batched = !shp.empty() && shp[0] == N && (decl0 <= 0 || decl0 == N)
                           && (total % static_cast<size_t>(N)) == 0;
      const size_t per = batched ? total / static_cast<size_t>(N) : total;
      if(batched)
        shp[0] = 1;
      sl[j] = Ort::Value::CreateTensor<float>(
          alloc, shp.data(), shp.size());
      std::memcpy(
          sl[j].GetTensorMutableData<float>(),
          outs[j].GetTensorData<float>()
              + (batched ? static_cast<size_t>(b) * per : 0),
          per * sizeof(float));
    }
    kps.clear();
    decodeLandmark(
        role, spec, std::span<Ort::Value>(sl, n_out), mw, mh,
        static_cast<float>(inputs.min_confidence), kps);
    pushFromKps(kps, Onnx::ROI::rectToTransform(rois[b], mw, mh));
  }
}

void PoseDetector::runLandmark(
    const Onnx::ModelRole& role, PoseWorkflow draw, const QImage& src,
    const QTransform& M, int track_id)
{
  const float mean_conf = landmarkKeypoints(role, src, M, m_kp_scratch);
  if(mean_conf < 0.f || m_kp_scratch.empty())
  {
    passthrough(src);
    return;
  }

  DetectedPose detected;
  detected.keypoints = m_kp_scratch;
  detected.mean_confidence = mean_conf;
  detected.track_id = track_id; // set BEFORE draw so the id-color applies
  applySmoothing(detected);
  fillBoxFromKeypoints(detected);
  outputs.detection.value = std::move(detected);

  drawSkeleton(*outputs.detection.value, draw);
  generateGeometryOutput(*outputs.detection.value, draw);
}

// Multi-instance two-stage path (Track IDs on): detect top-K (or reuse per-track
// ROIs on detector-skip frames), landmark each into m_instances, then emit.
void PoseDetector::runMultiInstance(
    const Onnx::ModelRole& role, PoseWorkflow draw, const QImage& src)
{
  const int W = inputs.image.texture.width, H = inputs.image.texture.height;
  const int mw = role.input_w > 0 ? role.input_w : 256;
  const int mh = role.input_h > 0 ? role.input_h : 256;
  const int max_inst = std::clamp(static_cast<int>(inputs.max_instances.value), 1, 16);

  // Per-track ROI feedback (detector-skip) is only steady for the
  // MediaPipe-rotated landmark kinds; top-down (SimCC/heatmap) re-detects.
  const bool roi_trackable
      = role.kind == Onnx::ModelKind::BlazePoseLandmark
        || role.kind == Onnx::ModelKind::HandLandmark
        || role.kind == Onnx::ModelKind::FaceMeshLandmark;
  const int detect_cadence
      = std::max(1, static_cast<int>(inputs.detector_cadence.value));

  // --- Gather this frame's ROIs (image px), then batch-landmark them. ---
  m_rois.clear();

  const bool use_track_rois = inputs.track_roi.value && roi_trackable
                              && !m_tracker.tracks().empty()
                              && m_frames_since_detect < detect_cadence;
  if(use_track_rois)
  {
    ++m_frames_since_detect;
    for(const auto& tk : m_tracker.tracks())
    {
      if(tk.time_since_update != 0 || tk.kpts_smooth.empty())
        continue;
      m_kp_scratch.clear();
      m_kp_scratch.reserve(tk.kpts_smooth.size());
      for(const auto& k : tk.kpts_smooth)
        m_kp_scratch.push_back({k.x, k.y, k.z, k.score});
      const Onnx::ROI::Rect r
          = roiRectFromKeypoints(draw, m_kp_scratch, W, H, mw, mh);
      if(rectValid(r, W, H))
        m_rois.push_back(r);
    }
    if(m_rois.empty())
      m_frames_since_detect = detect_cadence; // force re-detect next frame
  }

  if(m_rois.empty())
  {
    m_frames_since_detect = 0;
    m_dets = runDetector(m_detector_role, src, role.domain);
    if(m_dets.empty())
    {
      passthrough(src);
      return;
    }
    if(static_cast<int>(m_dets.size()) > max_inst)
    {
      std::partial_sort(
          m_dets.begin(), m_dets.begin() + max_inst, m_dets.end(),
          [](const auto& a, const auto& b) { return a.score > b.score; });
      m_dets.resize(max_inst);
    }
    for(const auto& d : m_dets)
    {
      const Onnx::ROI::Rect r = detectionRect(role, d, W, H);
      if(rectValid(r, W, H))
        m_rois.push_back(r);
    }
  }

  if(m_rois.empty())
  {
    passthrough(src);
    return;
  }

  runLandmarkBatch(role, draw, src, m_rois); // batched if the model allows

  if(m_instances.empty())
  {
    passthrough(src);
    return;
  }
  emitInstances(draw, /*do_track=*/true);
}

// Track m_instances, assign ids + per-id smoothed keypoints + colors, draw all,
// and publish every output port (poses/detection/geometry/poses_geometry/count).
void PoseDetector::emitInstances(PoseWorkflow draw, bool do_track)
{
  m_lost_frames = 0;

  // Every instance carries its bbox in metadata (detector box for box-only
  // detections, else the confident-keypoint bbox).
  for(auto& pose : m_instances)
    fillBoxFromKeypoints(pose);

  if(do_track)
  {
    // The tracker owns per-id One-Euro smoothing (no cross-identity bleed).
    Onnx::Track::Config cfg = m_tracker.config();
    cfg.smooth = inputs.smoothing.value;
    if(inputs.smoothing.value)
    {
      const float amt = std::clamp(
          static_cast<float>(inputs.smoothing_amount.value), 0.f, 1.f);
      cfg.smooth_min_cutoff = 5.0f * std::pow(0.02f / 5.0f, amt);
      cfg.smooth_beta = 1.0f + 4.0f * amt;
    }
    cfg.use_reid = inputs.reid.value && reid_ctx && m_reid_spec.valid;
    cfg.w_emb = std::clamp(static_cast<float>(inputs.reid_weight.value), 0.f, 1.f);

    // Plausibility gates (anti-jitter). Each is independent so methods can be
    // A/B-compared; defaults reproduce the pre-gate baseline.
    cfg.motion_gate = inputs.motion_gate.value;
    cfg.max_speed = std::max(0.01f, static_cast<float>(inputs.max_speed.value));
    cfg.birth_gate = inputs.birth_gate.value;
    cfg.strict_confirm = inputs.strict_confirm.value;
    m_tracker.configure(cfg);

    m_track_in.clear();
    m_track_in.reserve(m_instances.size());
    for(const auto& pose : m_instances)
    {
      Onnx::Track::Detection td;
      // Tracker box (center form) from the pose bbox; keypoints (if any) feed
      // the OKS cue, otherwise the tracker leans on IoU + Re-ID. A non-finite
      // box (a NaN model coordinate reaching here) would poison the Kalman state
      // permanently, so neutralize it: degenerate box + zero score (won't birth
      // or match). Index alignment with m_instances must be preserved, so we
      // sanitize rather than skip.
      const auto& pb = pose.box;
      const bool box_ok = finitef(pb.x) && finitef(pb.y) && finitef(pb.w)
                          && finitef(pb.h) && pb.w > 0.f && pb.h > 0.f;
      if(box_ok)
        td.box = {pb.x + pb.w * 0.5f, pb.y + pb.h * 0.5f, pb.w, pb.h};
      else
        td.box = {0.5f, 0.5f, 1e-3f, 1e-3f};
      td.score = box_ok ? pose.mean_confidence : 0.f;
      td.keypoints.reserve(pose.keypoints.size());
      for(const auto& k : pose.keypoints)
      {
        if(finitef(k.x) && finitef(k.y))
          td.keypoints.push_back({k.x, k.y, k.z, k.confidence});
        else // neutralize a NaN keypoint without breaking OKS's equal-size need
          td.keypoints.push_back({td.box.cx, td.box.cy, 0.f, 0.f});
      }
      m_track_in.push_back(std::move(td));
    }

    if(cfg.use_reid)
      embedInstances(); // crop + Re-ID -> m_track_in[i].embedding

    const auto ids = m_tracker.update(m_track_in);
    for(size_t i = 0; i < m_instances.size(); ++i)
    {
      const int id = (i < ids.size()) ? ids[i] : -1;
      m_instances[i].track_id = id;
      if(id < 0 || !inputs.smoothing.value)
        continue;
      for(const auto& tk : m_tracker.tracks())
      {
        if(tk.id != id)
          continue;
        if(tk.kpts_smooth.size() == m_instances[i].keypoints.size())
          for(size_t k = 0; k < tk.kpts_smooth.size(); ++k)
          {
            m_instances[i].keypoints[k].x = tk.kpts_smooth[k].x;
            m_instances[i].keypoints[k].y = tk.kpts_smooth[k].y;
          }
        // Smooth the bbox too: the tracker's Kalman box state is temporally
        // filtered, so emit it instead of the raw per-frame detection box.
        const auto bb = tk.box();
        if(bb.w > 0.f && bb.h > 0.f)
          m_instances[i].box
              = {bb.cx - bb.w * 0.5f, bb.cy - bb.h * 0.5f, bb.w, bb.h};
        break;
      }
    }
  }

  // primary = highest-confidence instance (back-compat single-pose ports)
  int primary = -1;
  float best = -1.f;
  for(size_t i = 0; i < m_instances.size(); ++i)
    if(m_instances[i].mean_confidence > best)
    {
      best = m_instances[i].mean_confidence;
      primary = static_cast<int>(i);
    }

  drawAllSkeletons(draw);

  outputs.poses.value = m_instances;
  if(primary >= 0)
  {
    outputs.detection.value = m_instances[primary];
    generateGeometryOutput(m_instances[primary], draw); // fills outputs.geometry
  }
  else
  {
    outputs.detection.value.reset();
    outputs.geometry.value.clear();
  }

  // Fixed-stride multi geometry: max_inst slots of
  //   [track_id, class_id, box_x, box_y, box_w, box_h, (x,y,z,conf)*K],
  // zero-padded. Constant layout regardless of Data Format (GPU-friendly).
  // K is 0 for box-only detections (header carries the box).
  constexpr int HEADER = 6;
  const int max_inst = std::clamp(
      static_cast<int>(inputs.max_instances.value),
      1, 16);
  auto& pg = outputs.poses_geometry.value;
  pg.clear();
  if(!m_instances.empty())
  {
    // Stride from the MAX keypoint count across instances so a mixed-K frame
    // (e.g. body + hand) never truncates the richer instances; shorter ones
    // zero-pad. (Today all instances share K, but don't bake that in.)
    int nkpt = 0;
    for(const auto& p : m_instances)
      nkpt = std::max(nkpt, static_cast<int>(p.keypoints.size()));
    const int stride = HEADER + nkpt * 4;
    pg.assign(static_cast<size_t>(max_inst) * stride, 0.f);
    int slot = 0;
    for(const auto& pose : m_instances)
    {
      if(slot >= max_inst)
        break;
      float* s = pg.data() + static_cast<size_t>(slot) * stride;
      s[0] = static_cast<float>(pose.track_id);
      s[1] = static_cast<float>(pose.class_id);
      s[2] = pose.box.x;
      s[3] = pose.box.y;
      s[4] = pose.box.w;
      s[5] = pose.box.h;
      const int K = std::min(nkpt, static_cast<int>(pose.keypoints.size()));
      for(int k = 0; k < K; ++k)
      {
        s[HEADER + k * 4 + 0] = pose.keypoints[k].x;
        s[HEADER + k * 4 + 1] = pose.keypoints[k].y;
        s[HEADER + k * 4 + 2] = pose.keypoints[k].z;
        s[HEADER + k * 4 + 3] = pose.keypoints[k].confidence;
      }
      ++slot;
    }
  }
  outputs.count.value = static_cast<int>(m_instances.size());
}

// Crop each instance's person box, run the Re-ID model (batched if it allows),
// L2-normalize the feature vector, and store it in m_track_in[i].embedding.
void PoseDetector::embedInstances()
{
  if(!reid_ctx || !m_reid_spec.valid || m_track_in.empty())
    return;
  auto& rctx = *reid_ctx;
  auto spec = rctx.readModelSpec();
  if(spec.inputs.empty() || spec.outputs.empty())
    return;

  const int W = inputs.image.texture.width, H = inputs.image.texture.height;
  QImage src(
      reinterpret_cast<const uchar*>(inputs.image.texture.bytes), W, H,
      QImage::Format_RGBA8888);
  const int mw = m_reid_spec.in_w, mh = m_reid_spec.in_h;
  const int D = m_reid_spec.embed_dim;
  const int oidx = m_reid_spec.out_index;
  const int N = static_cast<int>(m_track_in.size());

  // Resolve preprocessing (channel order + mean/std), Auto by input size.
  ReidPreprocess pp = inputs.reid_preprocess.value;
  if(pp == ReidPreprocess::Auto)
    pp = (mw == 112 && mh == 112) ? ReidPreprocess::ArcFaceRGB
         : (mw <= 128 && mh <= 128) ? ReidPreprocess::RawBGR
                                    : ReidPreprocess::ImageNetRGB;
  bool bgr = false;
  std::array<float, 3> mean{0, 0, 0}, stdv{255, 255, 255};
  switch(pp)
  {
    case ReidPreprocess::ImageNetRGB:
      mean = {0.485f * 255, 0.456f * 255, 0.406f * 255};
      stdv = {0.229f * 255, 0.224f * 255, 0.225f * 255};
      break;
    case ReidPreprocess::RawBGR:
      bgr = true; mean = {0, 0, 0}; stdv = {1, 1, 1};
      break;
    case ReidPreprocess::RawRGB:
      mean = {0, 0, 0}; stdv = {1, 1, 1};
      break;
    case ReidPreprocess::ZeroOneRGB:
      mean = {0, 0, 0}; stdv = {255, 255, 255};
      break;
    case ReidPreprocess::ArcFaceRGB:
      mean = {127.5f, 127.5f, 127.5f}; stdv = {128, 128, 128};
      break;
    default:
      break;
  }

  // One crop -> normalized input floats (NCHW RGB/BGR, or NHWC RGB fallback).
  auto buildInput = [&](const QImage& crop, boost::container::vector<float>& scr)
      -> Onnx::FloatTensor {
    if(m_reid_spec.nhwc)
      return Onnx::nhwc_rgb_tensorFromRGBA(
          spec.inputs[0], crop.constBits(), mw, mh, mw, mh, scr);
    if(bgr)
      return nchwBgrDetectorTensor(
          spec.inputs[0], crop, mw, mh, scr, mean, stdv);
    return Onnx::nchw_tensorFromRGBA(
        spec.inputs[0], crop.constBits(), mw, mh, mw, mh, scr, mean, stdv);
  };

  auto cropFor = [&](int i) -> QImage {
    const auto& b = m_track_in[i].box; // normalized center form
    const QRectF box_px(
        (b.cx - b.w * 0.5f) * W, (b.cy - b.h * 0.5f) * H, b.w * W, b.h * H);
    const Onnx::ROI::Rect r = Onnx::ROI::topdownRect(box_px, mw, mh, 1.1f);
    return Onnx::ROI::warpCrop(src, Onnx::ROI::rectToTransform(r, mw, mh), mw, mh);
  };

  auto storeRow = [&](int i, const float* row) {
    auto& e = m_track_in[i].embedding;
    e.resize(D);
    float norm = 0.f;
    for(int k = 0; k < D; ++k) { e[k] = row[k]; norm += row[k] * row[k]; }
    norm = std::sqrt(norm);
    if(norm > 1e-9f) for(auto& v : e) v /= norm;
  };

  const bool can_batch = m_reid_spec.batchable && N >= 2;
  if(can_batch)
  {
    const int CHW = 3 * mw * mh;
    m_reid_batch.resize(static_cast<size_t>(N) * CHW, boost::container::default_init);
    for(int i = 0; i < N; ++i)
    {
      Onnx::FloatTensor ft = buildInput(cropFor(i), m_reid_tmp);
      std::memcpy(
          m_reid_batch.data() + static_cast<size_t>(i) * CHW, ft.storage.data(),
          static_cast<size_t>(CHW) * sizeof(float));
      std::swap(m_reid_tmp, ft.storage);
    }
    std::vector<int64_t> in_shape = spec.inputs[0].shape;
    if(in_shape.size() == 4) in_shape[0] = N;
    else in_shape = {N, 3, mh, mw};
    Ort::Value in0 = Onnx::vec_to_tensor<float>(m_reid_batch, in_shape);
    Ort::Value outs[4]{
        Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
        Ort::Value{nullptr}};
    const size_t n_out = std::min<size_t>(4, spec.output_names_char.size());
    Ort::Value ins[1] = {std::move(in0)};
    rctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
    if(oidx < static_cast<int>(n_out) && outs[oidx].IsTensor())
    {
      const float* data = outs[oidx].GetTensorData<float>();
      for(int i = 0; i < N; ++i)
        storeRow(i, data + static_cast<size_t>(i) * D);
    }
  }
  else
  {
    for(int i = 0; i < N; ++i)
    {
      Onnx::FloatTensor ft = buildInput(cropFor(i), m_reid_tmp);
      Ort::Value outs[4]{
          Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
          Ort::Value{nullptr}};
      const size_t n_out = std::min<size_t>(4, spec.output_names_char.size());
      Ort::Value ins[1] = {std::move(ft.value)};
      rctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
      std::swap(m_reid_tmp, ft.storage);
      if(oidx < static_cast<int>(n_out) && outs[oidx].IsTensor())
        storeRow(i, outs[oidx].GetTensorData<float>());
    }
  }
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
  detected.class_id = d.class_id;
  detected.box
      = {static_cast<float>(d.box().x()), static_cast<float>(d.box().y()),
         d.w, d.h};
  applySmoothing(detected);
  fillBoxFromKeypoints(detected); // no-op if box already set above
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

  // Multi-instance: YOLO-pose is single-stage but already finds every person.
  if(inputs.track_ids.value)
  {
    const int max_inst = std::clamp(
        static_cast<int>(
            inputs.max_instances.value),
        1, 16);
    if(static_cast<int>(poses.size()) > max_inst)
      std::partial_sort(
          poses.begin(), poses.begin() + max_inst, poses.end(),
          [](const auto& a, const auto& b) { return a.confidence > b.confidence; });
    const int np = std::min(static_cast<int>(poses.size()), max_inst);
    m_instances.clear();
    for(int pi = 0; pi < np; ++pi)
    {
      const auto& pp = poses[pi];
      DetectedPose dp;
      dp.keypoints.assign(17, PoseKeypoint{0.f, 0.f, 0.f, 0.f});
      for(const auto& kp : pp.keypoints)
        if(kp.kp >= 0 && kp.kp < 17)
        {
          const QPointF p = M.map(QPointF(kp.x, kp.y));
          dp.keypoints[kp.kp]
              = {static_cast<float>(p.x() / iw), static_cast<float>(p.y() / ih),
                 0.0f, 1.0f};
        }
      dp.mean_confidence = pp.confidence;
      m_instances.push_back(std::move(dp));
    }
    std::swap(storage, t.storage);
    if(m_instances.empty())
    {
      passthrough(src);
      return;
    }
    emitInstances(PoseWorkflow::YOLOPose, /*do_track=*/true);
    return;
  }

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
  fillBoxFromKeypoints(detected);
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

  // Multi-instance: RTMO is NMS-free and returns every person already.
  if(inputs.track_ids.value)
  {
    const float iw = src.width(), ih = src.height();
    const float thr = std::max(0.3f, static_cast<float>(inputs.min_confidence));
    const int max_inst = std::clamp(
        static_cast<int>(
            inputs.max_instances.value),
        1, 16);
    std::vector<std::pair<float, int>> sel;
    sel.reserve(N);
    for(int i = 0; i < N; ++i)
      if(dets[i * 5 + 4] > thr)
        sel.push_back({dets[i * 5 + 4], i});
    if(static_cast<int>(sel.size()) > max_inst)
      std::partial_sort(
          sel.begin(), sel.begin() + max_inst, sel.end(),
          [](const auto& a, const auto& b) { return a.first > b.first; });
    const int ns = std::min(static_cast<int>(sel.size()), max_inst);
    m_instances.clear();
    for(int si = 0; si < ns; ++si)
    {
      const int idx = sel[si].second;
      DetectedPose dp;
      dp.keypoints.reserve(K);
      for(int k = 0; k < K; ++k)
      {
        const float* p = kpt + (static_cast<size_t>(idx) * K + k) * 3;
        dp.keypoints.push_back(
            {((p[0] - lb.pad_x) / lb.scale) / iw,
             ((p[1] - lb.pad_y) / lb.scale) / ih, 0.0f, p[2]});
      }
      dp.mean_confidence = sel[si].first;
      m_instances.push_back(std::move(dp));
    }
    if(m_instances.empty())
    {
      passthrough(src);
      return;
    }
    emitInstances(PoseWorkflow::YOLOPose, /*do_track=*/true);
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
  fillBoxFromKeypoints(detected);
  outputs.detection.value = std::move(detected);

  drawSkeleton(*outputs.detection.value, PoseWorkflow::YOLOPose);
  generateGeometryOutput(*outputs.detection.value, PoseWorkflow::YOLOPose);
}

void PoseDetector::runBoxDetection(
    const Onnx::ModelRole& detRole, const QImage& src)
{
  // detection_class: -1 = all classes, >=0 = that class id.
  const int class_sel = static_cast<int>(inputs.detection_class.value);
  m_dets = runDetector(detRole, src, Onnx::ModelDomain::Unknown, class_sel);
  if(m_dets.empty())
  {
    passthrough(src);
    return;
  }

  const int max_inst
      = std::clamp(static_cast<int>(inputs.max_instances.value), 1, 16);
  if(static_cast<int>(m_dets.size()) > max_inst)
  {
    std::partial_sort(
        m_dets.begin(), m_dets.begin() + max_inst, m_dets.end(),
        [](const auto& a, const auto& b) { return a.score > b.score; });
    m_dets.resize(max_inst);
  }

  m_instances.clear();
  m_instances.reserve(m_dets.size());
  for(const auto& d : m_dets)
  {
    DetectedPose p;
    p.mean_confidence = d.score;
    p.class_id = d.class_id;
    const auto b = d.box();
    p.box = {static_cast<float>(b.x()), static_cast<float>(b.y()), d.w, d.h};
    m_instances.push_back(std::move(p));
  }

  // Reuse the multi-instance back-end: tracking (box-only), Re-ID, per-id color,
  // and all the poses / detection / poses_geometry / count outputs.
  emitInstances(PoseWorkflow::BoxDetection, inputs.track_ids.value);
}

void PoseDetector::operator()()
try
{
  if(!available)
    return;

  auto& in_tex = inputs.image.texture;
  if(!in_tex.changed)
    return;
  if(!in_tex.bytes)
    return;

  const bool have_landmark = !inputs.model.current_model_invalid
                             && inputs.model.file.bytes.size() >= 32;
  const bool have_det = !inputs.det_model.current_model_invalid
                        && inputs.det_model.file.bytes.size() >= 32;
  const bool have_reid = inputs.reid.value
                         && !inputs.reid_model.current_model_invalid
                         && inputs.reid_model.file.bytes.size() >= 32;

  // Reset contexts on workflow / model change.
  const PoseWorkflow wf = inputs.workflow.value;

  // Box Detection runs the Detection Model with no landmark stage: explicit
  // workflow, or Auto with a Detection Model and no Landmark Model loaded.
  const bool box_only
      = have_det
        && (wf == PoseWorkflow::BoxDetection
            || (wf == PoseWorkflow::Auto && !have_landmark));
  if(!box_only && !have_landmark)
    return;

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
  if(inputs.reid_model.file.filename != m_last_reid_model)
  {
    reid_ctx.reset();
    m_reid_spec = {};
    m_last_reid_model = std::string(inputs.reid_model.file.filename);
    reinit = true;
  }
  // A model/workflow change invalidates the temporal tracking/smoothing state.
  if(reinit)
  {
    m_tracking = false;
    m_last_keypoints.clear();
    m_roi_smoother.reset();
    m_smoother.reset();
    m_tracker.reset();
    m_lost_frames = 0;
    m_frames_since_detect = 0;
  }

  // Model construction is the only failure that should permanently invalidate
  // the node; a per-frame inference/decode exception must NOT (it would kill the
  // node forever after a single transient throw — see the function catch below).
  try
  {
    if(have_landmark && !this->ctx)
    {
      this->ctx = std::make_unique<Onnx::OnnxRunContext>(
          this->inputs.model.file.bytes);
      m_landmark_role = Onnx::classify(toModelIO(this->ctx->readModelSpec()));
    }
    if(have_det && !this->det_ctx)
    {
      this->det_ctx = std::make_unique<Onnx::OnnxRunContext>(
          this->inputs.det_model.file.bytes);
      m_detector_role
          = Onnx::classify(toModelIO(this->det_ctx->readModelSpec()));
    }
    if(have_reid && !this->reid_ctx)
    {
      this->reid_ctx = std::make_unique<Onnx::OnnxRunContext>(
          this->inputs.reid_model.file.bytes);
      m_reid_spec
          = Onnx::classifyReid(toModelIO(this->reid_ctx->readModelSpec()));
    }
  }
  catch(...)
  {
    // Invalidate whichever model we were actually trying to construct.
    if(box_only)
      inputs.det_model.current_model_invalid = true;
    else
      inputs.model.current_model_invalid = true;
    ctx.reset();
    det_ctx.reset();
    reid_ctx.reset();
    return;
  }

  QImage src(
      reinterpret_cast<const uchar*>(in_tex.bytes), in_tex.width,
      in_tex.height, QImage::Format_RGBA8888);

  // --- Box Detection: run the Detection Model, emit boxes (no landmark) ---
  if(box_only)
  {
    runBoxDetection(m_detector_role, src);
    return;
  }

  const Onnx::ModelRole role
      = (wf == PoseWorkflow::Auto) ? m_landmark_role : roleForWorkflow(wf);
  const PoseWorkflow draw
      = (wf == PoseWorkflow::Auto) ? workflowForRole(role) : wf;

  // --- Two-stage: detector + landmark ---
  if(have_det && role.stage == Onnx::ModelStage::Landmark)
  {
    // Track IDs on -> multi-instance pipeline (all people, ids, per-id color).
    if(inputs.track_ids.value)
    {
      runMultiInstance(role, draw, src);
      return;
    }

    const int mw = role.input_w > 0 ? role.input_w : 256;
    const int mh = role.input_h > 0 ? role.input_h : 256;

    // --- ROI: tracking loop (skip detector) vs fresh detection -------------
    // The tracking ROI is derived from the model's own output, i.e. a feedback
    // loop. It is only stable for the MediaPipe-rotated landmark models, which
    // produce a well-localized rotated ROI (BlazePose/Hand/FaceMesh). For
    // top-down models (SimCC/heatmap) it is bbox->bbox feedback that jitters,
    // and MobileFaceNet (fill-the-crop) explodes — those re-detect every frame.
    const bool can_track
        = inputs.track_roi.value
          && (role.kind == Onnx::ModelKind::BlazePoseLandmark
              || role.kind == Onnx::ModelKind::HandLandmark
              || role.kind == Onnx::ModelKind::FaceMeshLandmark);
    Onnx::ROI::Rect rect;
    bool from_tracking = false;
    if(can_track && m_tracking && !m_last_keypoints.empty())
    {
      // Derive the ROI from last frame's landmarks — no detector this frame.
      Onnx::ROI::Rect cand = roiRectFromKeypoints(
          draw, m_last_keypoints, in_tex.width, in_tex.height, mw, mh);
      // Only trust it if it's well-formed and didn't teleport/shrink vs the
      // previous ROI — otherwise re-detect (prevents drift/center-collapse).
      if(rectValid(cand, in_tex.width, in_tex.height)
         && (!m_have_prev_roi || rectPlausible(cand, m_prev_roi)))
      {
        rect = cand;
        from_tracking = true;
      }
    }
    if(!from_tracking)
    {
      auto dets = runDetector(m_detector_role, src, role.domain);
      if(dets.empty())
      {
        passthrough(src);
        return;
      }
      rect = detectionRect(role, dets.front(), in_tex.width, in_tex.height);
      m_roi_smoother.reset(); // fresh acquisition: don't blend across the gap
      m_have_prev_roi = false;
    }

    // Stabilize the crop. Deadband first: if the tracked ROI barely changed,
    // reuse the previous one verbatim so a static subject gives a static crop
    // (kills the feedback shake); otherwise smooth toward the new ROI.
    if(from_tracking && m_have_prev_roi && rectClose(rect, m_prev_roi))
      rect = m_prev_roi;
    else
      rect = smoothRoi(rect);
    m_prev_roi = rect;
    m_have_prev_roi = true;
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
  // Transient per-frame failure (a bad crop, an odd output shape on one frame,
  // an ORT hiccup). Skip this frame and retry next one — do NOT permanently
  // invalidate the model (construction failures are handled separately above).
}

} // namespace OnnxModels
