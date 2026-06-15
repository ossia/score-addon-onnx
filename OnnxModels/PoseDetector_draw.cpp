#include "PoseDetector_internal.hpp"

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

// controlnet_aux OpenPose palette, keyed by COCO-18 keypoint index (verified
// vs open_pose/util.py `colors`). This drives an OpenPose ControlNet for Stable
// Diffusion. NOTE: distinct from CMU's COCO palette below (rotated by one); use
// THIS one for ControlNet.
static Rgba getOpenPoseColor18(int idx)
{
  static constexpr unsigned char c[18][3] = {
      {255, 0, 0},   {255, 85, 0},   {255, 170, 0}, {255, 255, 0},
      {170, 255, 0}, {85, 255, 0},   {0, 255, 0},   {0, 255, 85},
      {0, 255, 170}, {0, 255, 255},  {0, 170, 255}, {0, 85, 255},
      {0, 0, 255},   {85, 0, 255},   {170, 0, 255}, {255, 0, 255},
      {255, 0, 170}, {255, 0, 85}};
  if(idx < 0 || idx >= 18)
    return Colors::torso;
  return Onnx::rgb8(c[idx][0], c[idx][1], c[idx][2]);
}

// Official CMU OpenPose BODY_25 palette, keyed by BODY_25 keypoint index
// (verified vs poseParametersRender.hpp POSE_BODY_25_COLORS_RENDER_GPU).
static Rgba getOpenPoseColor25(int idx)
{
  static constexpr unsigned char c[25][3] = {
      {255, 0, 85},  {255, 0, 0},   {255, 85, 0},  {255, 170, 0}, {255, 255, 0},
      {170, 255, 0}, {85, 255, 0},  {0, 255, 0},   {255, 0, 0},   {0, 255, 85},
      {0, 255, 170}, {0, 255, 255}, {0, 170, 255}, {0, 85, 255},  {0, 0, 255},
      {255, 0, 170}, {170, 0, 255}, {255, 0, 255}, {85, 0, 255},  {0, 0, 255},
      {0, 0, 255},   {0, 0, 255},   {0, 255, 255}, {0, 255, 255}, {0, 255, 255}};
  if(idx < 0 || idx >= 25)
    return Colors::torso;
  return Onnx::rgb8(c[idx][0], c[idx][1], c[idx][2]);
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
  auto toPoint = [&](int idx) -> Onnx::Vec2 {
    return {kps[idx].x * w, kps[idx].y * h};
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

  // OpenPose targets: render with the official palette, opaque, thicker sticks,
  // so the overlay can drive an OpenPose ControlNet (use Skeleton-only output
  // for the required black background). Per-id track color overrides it.
  const bool openpose_render
      = m_remap_active && !use_track_color
        && (m_active_target == Onnx::Skel::TargetSkeleton::OpenPoseCoco18
            || m_active_target == Onnx::Skel::TargetSkeleton::OpenPoseBody25);

  // Select skeleton connections and color function based on workflow
  auto getColor = [&](int idx) -> Rgba {
    if(use_track_color)
      return track_color;
    if(m_remap_active) // remapped indices don't match the native palettes
    {
      if(m_active_target == Onnx::Skel::TargetSkeleton::OpenPoseCoco18)
        return getOpenPoseColor18(idx);
      if(m_active_target == Onnx::Skel::TargetSkeleton::OpenPoseBody25)
        return getOpenPoseColor25(idx);
      return Onnx::hsv(
          static_cast<float>(idx) / std::max(1, num_kps), 0.6f, 1.0f);
    }
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

        // Skip edges touching a missing/zero-confidence joint (e.g. (0,0,0)
        // unmappable joints from a remap) even when min_conf == 0.
        if(kps[from].confidence <= 0.f || kps[to].confidence <= 0.f)
          continue;
        float conf = std::min(kps[from].confidence, kps[to].confidence);
        if(conf < min_conf)
          continue;

        Rgba color = openpose_render
                         ? getColor(from)
                         : Onnx::withAlpha(getColor(from), safeAlpha(conf));
        ov.lineWidth(openpose_render ? 4.f : 2.f);
        ov.color(color);
        ov.line(toPoint(from), toPoint(to));
      }
    };

    if(m_remap_active
       && m_active_target == Onnx::Skel::TargetSkeleton::OpenPoseCoco18)
    {
      // Exact OpenPose/ControlNet limbs: limb i uses palette color i at 60%
      // intensity (controlnet_aux), keypoint dots full intensity (below). The
      // edge list is in limbSeq order so index i lines up with the palette.
      const auto bones = Onnx::Skel::edgesFor(m_active_target);
      ov.lineWidth(4.f);
      for(int i = 0; i < static_cast<int>(bones.size()); ++i)
      {
        const auto& b = bones[i];
        if(b.a < 0 || b.b < 0 || b.a >= num_kps || b.b >= num_kps)
          continue;
        if(kps[b.a].confidence <= 0.f || kps[b.b].confidence <= 0.f)
          continue;
        if(std::min(kps[b.a].confidence, kps[b.b].confidence) < min_conf)
          continue;
        const Rgba k = getOpenPoseColor18(i);
        ov.color(Rgba{k.r * 0.6f, k.g * 0.6f, k.b * 0.6f, 1.f});
        ov.line(toPoint(b.a), toPoint(b.b));
      }
    }
    else if(m_remap_active) // other remapped layouts: endpoint color
    {
      drawConnections(Onnx::Skel::edgesFor(m_active_target), num_kps);
    }
    else
    switch(workflow)
    {
      case PoseWorkflow::BlazePose:
        drawConnections(Skeletons::blazepose, 33);
        break;
      case PoseWorkflow::RTMPose_COCO:
        // Generic heatmap/SimCC route: pick the skeleton by keypoint count.
        // 21 = hand, 68 = dlib face (e.g. 2DFAN-4), 17 = COCO body. An unknown
        // count (e.g. 13-pt garment keypoints) gets NO skeleton — just the dots
        // below — instead of a COCO body scribbled over unrelated points.
        if(num_kps == 21)
          drawConnections(Skeletons::hands, 21);
        else if(num_kps == 68)
          drawConnections(Skeletons::dlib68, 68);
        else if(num_kps == 17)
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
        // dlib-68 connections only make sense on an ACTUAL 68-point dlib/300W
        // layout. A 98-point WFLW model (Peppa-Pig) or a 106-point one has a
        // different order; drawing dlib68 on it makes the "weird edges / holes /
        // strange nose" the user saw. For non-68 face landmarks, dots only.
        if(num_kps == 68)
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
        // Only draw a body skeleton if the count actually matches one; a 68-pt
        // face gets the face skeleton, any other unrecognized count draws dots
        // only (no spurious COCO lines all over the points).
        if(num_kps == 17)
          drawConnections(Skeletons::coco17, 17);
        else if(num_kps == 68)
          drawConnections(Skeletons::dlib68, 68);
        break;
    }
  }

  // Draw keypoints (landmark dots), gated independently of the skeleton lines.
  for(int i = 0; inputs.draw_landmarks.value && i < num_kps; ++i)
  {
    float conf = kps[i].confidence;
    if(conf < min_conf)
      continue;

    Rgba color = openpose_render
                     ? getColor(i)
                     : Onnx::withAlpha(getColor(i), safeAlpha(conf));
    ov.color(color);

    // Size based on keypoint importance
    int radius = 4;
    if(openpose_render)
      radius = 4; // uniform OpenPose dots
    else if(workflow == PoseWorkflow::RTMPose_Whole)
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
    const Onnx::Rect r{
        pose.box.x * w, pose.box.y * h, pose.box.w * w, pose.box.h * h};
    ov.strokeRect(r);

    char lbl[96];
    int n = 0;
    // snprintf returns the length it WOULD have written; clamp after each append
    // so a long class name (from a user class_file) can't push n past the buffer
    // and underflow the size_t `sizeof(lbl) - n` of the next call.
    const auto clampN = [&] {
      if(n < 0 || n >= (int)sizeof(lbl))
        n = (int)sizeof(lbl) - 1;
    };
    if(pose.track_id >= 0)
      n += std::snprintf(lbl + n, sizeof(lbl) - n, "#%d ", pose.track_id);
    clampN();
    if(pose.class_id >= 0)
    {
      if(const char* nm = className(pose.class_id))
        n += std::snprintf(lbl + n, sizeof(lbl) - n, "%s ", nm);
      else
        n += std::snprintf(lbl + n, sizeof(lbl) - n, "c%d ", pose.class_id);
    }
    clampN();
    std::snprintf(lbl + n, sizeof(lbl) - n, "%.2f", pose.mean_confidence);
    ov.text(14.f, Onnx::Vec2{r.x + 2.f, std::max(10.f, r.y - 3.f)}, lbl);
  }
}

// Native keypoint layout the emitted pose carries, by workflow (+ keypoint
// count to disambiguate the COCO-body/hand share). nullopt => no remap source.
static std::optional<Onnx::Skel::SourceSkeleton>
sourceFor(PoseWorkflow wf, int num_kps)
{
  using S = Onnx::Skel::SourceSkeleton;
  switch(wf)
  {
    case PoseWorkflow::BlazePose: return S::BlazePose33;
    case PoseWorkflow::MediaPipeHands: return S::Hands21;
    case PoseWorkflow::FaceMesh: return S::FaceMesh468;
    case PoseWorkflow::MobileFaceNet: return S::Dlib68;
    case PoseWorkflow::RTMPose_COCO:
    case PoseWorkflow::RTMPose_Whole:
    case PoseWorkflow::ViTPose:
    case PoseWorkflow::YOLOPose:
      return (num_kps == 21) ? S::Hands21 : S::Coco17;
    default: return std::nullopt; // BlazeFace/RTMPoseFace/Animal/Box: no remap
  }
}

void PoseDetector::setRemapState(PoseWorkflow wf, int num_kps)
{
  const auto tgt = inputs.skeleton_type.value;
  m_active_target = tgt;
  const auto src = sourceFor(wf, num_kps);
  m_remap_active = (tgt != Onnx::Skel::TargetSkeleton::Native) && src.has_value()
                   && !Onnx::Skel::mappingFor(*src, tgt).empty();
  if(src)
    m_remap_src = *src;
}

void PoseDetector::remapPose(DetectedPose& pose)
{
  if(Onnx::Skel::remap<PoseKeypoint>(
         m_remap_src, m_active_target,
         std::span<const PoseKeypoint>(
             pose.keypoints.data(), pose.keypoints.size()),
         m_remap_scratch))
    std::swap(pose.keypoints, m_remap_scratch); // reuse old buffer next call
}

void PoseDetector::finalizeSingle(PoseWorkflow wf)
{
  if(!outputs.detection.value)
    return;
  setRemapState(
      wf, static_cast<int>(outputs.detection.value->keypoints.size()));
  if(m_remap_active)
    remapPose(*outputs.detection.value);
  drawSkeleton(*outputs.detection.value, wf);
  generateGeometryOutput(*outputs.detection.value, wf);
}

void PoseDetector::drawSkeleton(const DetectedPose& pose, PoseWorkflow workflow)
{
  ONNX_PROF_SCOPE(Draw);
  auto& in_tex = inputs.image.texture;
  const int w = in_tex.width, h = in_tex.height;
  const bool skeleton_only
      = (inputs.output_mode.value == PoseRenderMode::SkeletonOnly);

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
  ONNX_PROF_SCOPE(Draw);
  auto& in_tex = inputs.image.texture;
  const int w = in_tex.width, h = in_tex.height;
  const bool skeleton_only
      = (inputs.output_mode.value == PoseRenderMode::SkeletonOnly);

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

    case KeypointOutputFormat::WorldXYZArray:
    {
      // World-space xyz (meters, hip-origin) when the model provides it
      // (BlazePose family); screen xyz otherwise so the output stays usable
      // with any landmark model.
      const auto& src_kps = pose.world.empty() ? kps : pose.world;
      out.reserve(src_kps.size() * 3);
      for (const auto& kp : src_kps)
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

      // Remapped layout -> target edge table (addLine already skips edges
      // touching a missing/low-confidence joint).
      if(m_remap_active)
      {
        const auto bones = Onnx::Skel::edgesFor(m_active_target);
        out.reserve(bones.size() * 6);
        for(const auto& b : bones)
          addLine(b.a, b.b);
        break;
      }

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

} // namespace OnnxModels
