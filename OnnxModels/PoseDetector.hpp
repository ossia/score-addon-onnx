#pragma once
#include <OnnxModels/Utils.hpp>

#include <Onnx/helpers/Detection.hpp>
#include <Onnx/helpers/MeshOps.hpp>
#include <Onnx/helpers/MhrModel.hpp>
#include <Onnx/helpers/ModelRole.hpp>
#include <Onnx/helpers/OnnxBase.hpp>
#include <Onnx/helpers/OneEuro.hpp>
#include <Onnx/helpers/PoseTracker.hpp>
#include <Onnx/helpers/ROI.hpp>
#include <Onnx/helpers/SkeletonFormats.hpp>

#include <halp/controls.hpp>
#include <halp/file_port.hpp>
#include <halp/geometry.hpp>
#include <halp/layout.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>


namespace Onnx
{
struct OnnxRunContext;
}

namespace OnnxModels
{
struct Overlay; // ctx-backed software overlay renderer (CtxOverlay.hpp)

// Unified keypoint structure for all pose models
struct PoseKeypoint
{
  float x, y, z;       // z is 0 for 2D-only models
  float confidence;

  halp_field_names(x, y, z, confidence);
};

// Axis-aligned bounding box, normalized [0,1], top-left form.
struct BoundingBox
{
  float x{}, y{}, w{}, h{};
  halp_field_names(x, y, w, h);
};

struct DetectedPose
{
  std::vector<PoseKeypoint> keypoints; // empty for box-only detections
  // World-space 3D keypoints (meters, hip-origin) for models that emit them
  // (BlazePose full-body landmark's [1,117] world output; InstantHMR's
  // joints_3d re-centred on the pelvis = mid-hips, X right, Y down, Z
  // forward). Same joint order and confidences as `keypoints` (remapped with
  // them by the Skeleton control); empty when the model has no world output.
  // Crop-independent: never affine-mapped. One-Euro smoothed in metres when
  // Smoothing is on.
  std::vector<PoseKeypoint> world;
  float mean_confidence;
  int track_id = -1; // persistent ID across frames (-1 = untracked)
  BoundingBox box;   // always filled: detector box, or the keypoint bbox
  int class_id = -1; // detector class (-1 = n/a, e.g. landmark-only output)
  // --- 3D payload of human-mesh-recovery models (appended: field order of the
  //     reflected struct above is unchanged). Empty for every other model. ---
  // Camera-space position of the pose's world origin (metres, 3 floats: x
  // right, y down, z forward), so world[k] + translation is joint k in camera
  // space. Empty if the model has none.
  std::vector<float> translation;
  // Parametric body model parameters (e.g. MHR: 204 pose + 45 shape), in the
  // model's own order. Empty otherwise.
  std::vector<float> body_params;
  // Where the world origin (the pelvis) sits in the parametric body model's
  // own rig frame (metres, 3 floats, same axes as world). The body model's
  // rig origin is at translation - rig_offset in camera space (InstantHMR:
  // exactly its raw cam_trans), so a mesh evaluated from body_params, which
  // is rig-local, lands in camera space at mesh + translation - rig_offset.
  // Carried and smoothed as its own signal, so the mesh and the 3D keypoints
  // share the same smoothed translation and stay together under motion.
  // Empty when translation or body_params is.
  std::vector<float> rig_offset;

  halp_field_names(
      keypoints, world, mean_confidence, track_id, box, class_id, translation,
      body_params, rig_offset);
};

// Available pose estimation workflows
enum class PoseWorkflow
{
  Auto, // Automatic detection from model structure

  // Body pose
  BlazePose,     // MediaPipe BlazePose (33 keypoints, NHWC, direct landmarks)
  RTMPose_COCO,  // RTMPose COCO format (17 keypoints, NCHW, SimCC)
  RTMPose_Whole, // RTMPose WholeBody (133 keypoints, NCHW, SimCC)
  ViTPose,       // ViTPose (17 keypoints, NCHW, heatmaps)
  YOLOPose,      // YOLO Pose (17 keypoints, NCHW, direct output)
  AnimalPose,    // AP10K/APT36K quadruped (17 keypoints, RTMPose/ViTPose)

  // Hand
  MediaPipeHands, // MediaPipe Hands (21 keypoints, NHWC, direct landmarks)

  // Face
  FaceMesh,      // MediaPipe FaceMesh (468 keypoints, NHWC, direct landmarks)
  BlazeFace,     // BlazeFace detection (6 keypoints, NHWC, anchor-based)
  MobileFaceNet, // MobileFaceNet (68 dlib landmarks, NCHW)
  RTMPoseFace,   // RTMPose-Face (106 LaPa landmarks, NCHW square, SimCC)

  // Detection-only
  BoxDetection,  // Bounding boxes from the Detection Model (no landmark stage)

  // --- Appended: the combobox index is the preset value, keep it stable. ---
  // Human mesh recovery: InstantHMR (MHR70, 70 keypoints, top-down on a person
  // box) with a metric 3D payload: pelvis-relative world joints, the camera
  // translation and the MHR body parameters (204 pose + 45 shape).
  InstantHMR,
};

// ReID input preprocessing (the part not inferable from the ONNX graph).
// Auto picks by input size: 112->ArcFace, 128->RawBGR, else ImageNet-RGB.
enum class ReidPreprocess : uint8_t
{
  Auto,
  ImageNetRGB,  // (x/255 - mean)/std, RGB  (OSNet, FastReID)
  RawBGR,       // x in [0,255], BGR        (OpenVINO person/face-reid)
  RawRGB,       // x in [0,255], RGB
  ZeroOneRGB,   // x/255, RGB
  ArcFaceRGB,   // (x-127.5)/128, RGB       (ArcFace)
};

// Output visualization mode. Uniquely named (not "OutputMode") so it can't
// clash with the image/video nodes' OutputMode in the OnnxModels namespace.
enum class PoseRenderMode
{
  SkeletonOnImage, // Draw skeleton on top of input image
  SkeletonOnly,    // Draw skeleton on black background
};

// Keypoint data output format
enum class KeypointOutputFormat
{
  Raw,       // Raw structured output (x, y, z, confidence per keypoint)
  XYArray,   // Flat xyxy array
  XYZArray,  // Flat xyzxyz array
  LineArray, // Flat xyz pairs for e.g. GL_LINES (x1,y1,z1,x2,y2,z2,...)
  WorldXYZArray, // Flat world-space xyz (meters, hip-origin) from models with a
                 // world-3D output (BlazePose family); falls back to the screen
                 // xyz when the model has none.
  // Bounding-box formats: normalized [0,1] image coordinates, 4 floats per
  // detection — the primary one on Geometry, every live one on Poses Geometry,
  // exactly like the keypoint formats. Note that a keypoint-less detection (Box
  // Detection workflow) falls back to the XYWH box in the keypoint formats too —
  // the box is then the only geometry the instance has.
  BoxXYWH,     // x, y, w, h    (top-left corner + size)
  BoxX1Y1X2Y2, // x1, y1, x2, y2 (top-left + bottom-right corners)
  // Full per-instance record: [track_id, class_id, box_x, box_y, box_w, box_h,
  // (x, y, z, confidence) * K]. Metadata included, no confidence filtering, and
  // the 6-float header alone for a keypoint-less (box-only) detection. This is
  // the layout Poses Geometry always had before the outlet became Data
  // Format-driven, so selecting it there reproduces that buffer exactly.
  Flattened,
  // --- Appended after Flattened: preset enum values stay stable. ---
  // Flat camera-space xyz (metres): world + translation per joint, so several
  // people land at their true relative placement. Filtered by Min Confidence
  // exactly like WorldXYZArray, and falls back to WorldXYZArray (world, else
  // screen xyz) when the model has no camera translation.
  CameraXYZArray,
  // Parametric body record per instance: the body model's rig origin in
  // camera space (3 floats, metres: translation - rig_offset, i.e. InstantHMR's
  // cam_trans; the translation itself when there is no rig_offset, zero if
  // absent) then the body model parameters (e.g. MHR 204 pose + 45 shape). A
  // rig-local mesh evaluated from those parameters lands at mesh + origin,
  // where the node's own mesh formats put it. Unfiltered and fixed-length per
  // model; nothing at all for a model without body params.
  BodyParams,
  // --- Appended after BodyParams: the body mesh (needs a Body Model file and
  //     a model with body parameters, i.e. InstantHMR; nothing otherwise).
  //     Metres, in the Mesh Space control's axes, camera-space placement (so
  //     several people keep their true relative positions), faces
  //     counter-clockwise seen from outside in either space. Unfiltered and
  //     fixed-length per Body Model file (its LOD). ---
  // Unindexed triangle soup, 9 floats per face: the xyz of its 3 corners.
  // Directly renderable as a plain vertex array (3 vertices per triangle).
  MeshTriangles,
  // The unique vertices, 3 floats each (the topology is fixed per LOD: the
  // faces index them in the Body Model file's order).
  MeshVertices,
  // No triangles + smooth normals format: score's mesh objects recompute
  // normals from the soup.
};

// True for the formats whose Geometry payload is a bounding box, not keypoints.
inline constexpr bool isBoxFormat(KeypointOutputFormat f) noexcept
{
  return f == KeypointOutputFormat::BoxXYWH
         || f == KeypointOutputFormat::BoxX1Y1X2Y2;
}

// True for the body-mesh formats (their payload is a mesh, not keypoints).
inline constexpr bool isMeshFormat(KeypointOutputFormat f) noexcept
{
  return f == KeypointOutputFormat::MeshTriangles
         || f == KeypointOutputFormat::MeshVertices;
}

// Axes of the mesh Data Formats. Both are metres in camera space (camera at
// the origin); OpenGL is the Camera frame rotated 180 deg about X, so the
// winding stays counter-clockwise from outside.
enum class MeshSpace
{
  OpenGL, // X right, Y up, Z backward (toward the viewer): score's 3D objects
  Camera, // OpenCV / InstantHMR: X right, Y down, Z forward (into the scene)
};

struct PoseDetector : OnnxObject
{
public:
  halp_meta(name, "Pose Detector");
  halp_meta(c_name, "pose_detector");
  halp_meta(category, "AI/Computer Vision");
  halp_meta(author, "MediaPipe, MMPose, InstantHMR (MHR), ONNX Runtime");
  halp_meta(
      description,
      "Unified keypoint detection for body pose, hands, and face landmarks, "
      "plus bounding-box object detection and multi-object tracking. "
      "InstantHMR adds 3D human mesh recovery: MHR70 keypoints, metric "
      "pelvis-relative joints, their camera-space placement (perspective "
      "camera, principal point at the frame centre, focal = the frame "
      "diagonal in pixels) and the MHR body parameters");
  halp_meta(uuid, "f8e7d6c5-b4a3-4291-8c0d-1e2f3a4b5c6d");
  halp_meta(manual_url, "https://ossia.io/score-docs/processes/ai-recognition.html")

  struct ins
  {
    halp::texture_input<"In"> image;
    ModelPort<"Landmark Model"> model;

    struct : halp::combobox_t<"Workflow", PoseWorkflow>
    {
      halp_meta(description, "Pose estimation model type");

    } workflow;

    // Shown as "Preview Mode": it only affects the "Out" preview image. Presets
    // address ports by index, so the rename keeps them working.
    struct : halp::enum_t<PoseRenderMode, "Preview Mode">
    {
      halp_meta(
          description,
          "Background of the preview image: the input frame, or black (Skeleton "
          "Only, e.g. for an OpenPose ControlNet).");
    } output_mode;

    struct : halp::hslider_f32<"Min Confidence", halp::range{0., 1., 0.3}>
    {
      halp_meta(
          description,
          "Detection / keypoint acceptance threshold: instances and joints below "
          "this score are dropped. Also gates which instances get tracked.");
    } min_confidence;

    struct : halp::toggle<"Draw Skeleton">
    {
      halp_meta(
          description, "Draw the skeleton bone lines connecting the keypoints.");
      bool value = true;
    } draw_skeleton;

    struct : halp::enum_t<KeypointOutputFormat, "Data Format">
    {
      halp_meta(
          description,
          "Layout of the Geometry outlets, for GPU rendering: flat keypoint "
          "arrays, bone lines, the detection bounding box (XYWH / X1Y1X2Y2, "
          "normalized [0,1]), Flattened — the full per-instance record "
          "(ids + box + keypoints) — or, for 3D models, World XYZ (metres, "
          "hip-origin), Camera XYZ (metres, camera space: world + the model's "
          "camera translation) and Body Params (the body model's rig origin in "
          "camera space, then the parametric body parameters: a mesh evaluated "
          "from them lands at mesh + origin), and with a Body Model file the body mesh (triangle "
          "soup or unique vertices, metres in the Mesh Space axes). Geometry carries the primary detection, Poses Geometry "
          "one slot per tracked instance.");
    } data_format;


    ModelPort<"Detection Model"> det_model;

    struct : halp::toggle<"Track ROI">
    {
      halp_meta(
          description,
          "Two-stage only: derive the ROI from the previous frame's landmarks "
          "and skip the detector (faster + steadier, like MediaPipe). "
          "Experimental — feedback can drift on some models; off by default.");
      bool value = false;
    } track_roi;

    struct : halp::toggle<"Smoothing">
    {
      halp_meta(description, "Temporal One-Euro smoothing of keypoints");
      bool value = true;
    } smoothing;

    struct : halp::hslider_f32<"Smoothing Amount", halp::range{0., 1., 0.5}>
    {
      halp_meta(description, "0 = responsive, 1 = very smooth");
    } smoothing_amount;

    struct : halp::toggle<"Track IDs">
    {
      halp_meta(
          description,
          "Track every person/hand/face across frames and emit them all with a "
          "persistent ID + stable per-ID color (ByteTrack: Kalman + two-stage + "
          "OKS). Enables multi-instance output. Requires a Detection Model.");
      bool value = false;
    } track_ids;

    struct : halp::spinbox_i32<"Max Instances", halp::range{1, 16, 5}>
    {
      halp_meta(
          description,
          "Max simultaneous tracked instances (top-K by score). Track IDs only.");
    } max_instances;

    struct : halp::spinbox_i32<"Detector Cadence", halp::range{1, 30, 4}>
    {
      halp_meta(
          description,
          "Track ROI: re-run the detector every N frames; reuse per-track ROIs "
          "in between (1 = detect every frame). Track IDs + Track ROI only.");
    } detector_cadence;

    ModelPort<"Re-ID Model"> reid_model;

    struct : halp::toggle<"Re-ID">
    {
      halp_meta(
          description,
          "Blend an appearance embedding (any ReID model in the Re-ID Model "
          "port) into tracking so IDs survive long occlusion / re-entry. "
          "Track IDs only.");
      bool value = false;
    } reid;

    struct : halp::hslider_f32<"Re-ID Weight", halp::range{0., 1., 0.25}>
    {
      halp_meta(description, "How strongly appearance influences association.");
    } reid_weight;

    struct : halp::enum_t<ReidPreprocess, "Re-ID Preprocess">
    {
      halp_meta(
          description,
          "Input normalization for the Re-ID model (not inferable from the "
          "graph). Auto guesses from input size.");
    } reid_preprocess;

    // Appended after reid_preprocess to keep existing presets' port indices
    // (1..17) stable; these two are 18 and 19.
    struct : halp::toggle<"Draw Boxes">
    {
      halp_meta(
          description,
          "Draw each instance's bounding box (always on in Box Detection).");
      bool value = false;
    } draw_boxes;

    struct : halp::spinbox_i32<"Detection Class", halp::range{-1, 90, -1}>
    {
      halp_meta(
          description,
          "Keep only this class id of a multi-class detector; ignored by "
          "single-class detectors. Box Detection: -1 = all classes. Two-stage: "
          "the class to crop for the landmark model; -1 = by its domain "
          "(body 0, then hand 2 / face 3 on body-head-hand(-face) detectors).");
    } detection_class;

    struct : halp::toggle<"Draw Landmarks">
    {
      halp_meta(
          description,
          "Draw the keypoint dots (independent of the skeleton lines).");
      bool value = true;
    } draw_landmarks;

    // --- Tracking plausibility gates (anti-jitter; Track IDs only) ---
    struct : halp::combobox_t<"Motion Gate", Onnx::Track::MotionGate>
    {
      halp_meta(
          description,
          "Reject an id->detection match that is an implausible jump: None, "
          "MaxSpeed (an id can't cross the frame in one step), or Mahalanobis "
          "(Kalman gating distance, gap-aware). An appearance/ReID match can "
          "still re-acquire across the gate.");
    } motion_gate;

    struct : halp::hslider_f32<"Max Speed", halp::range{0.25, 6., 2.}>
    {
      halp_meta(
          description,
          "MaxSpeed gate budget: max per-frame center move, in units of the "
          "track's box size (scaled by how long it was lost). Lower = stricter.");
    } max_speed;

    struct : halp::toggle<"Birth Gate">
    {
      halp_meta(
          description,
          "Suppress spurious new ids whose box sits mostly inside an existing "
          "tracked person (e.g. a raised arm read as a second person).");
      bool value = true;
    } birth_gate;

    struct : halp::toggle<"Strict Confirmation">
    {
      halp_meta(
          description,
          "Delete a tentative id the first frame it fails to re-appear "
          "(DeepSORT n_init), so a one-frame detection never persists.");
      bool value = false;
    } strict_confirm;

    struct : halp::combobox_t<"Skeleton", Onnx::Skel::TargetSkeleton>
    {
      halp_meta(
          description,
          "Remap output keypoints (overlay + ports) to a standard skeleton "
          "layout. Native = the model's own layout; unsupported (e.g. animal "
          "-> human) falls back to Native.");
    } skeleton_type;

    // Appended last to keep existing presets' port indices stable.
    struct : halp::file_port<"Class Names File">
    {
      halp_meta(
          description,
          "Box Detection: a text file with one class name per line, used to "
          "label boxes instead of the numeric id. Empty = built-in COCO-80.");
    } class_file;

    struct : halp::spinbox_i32<"Track Memory", halp::range{1, 300, 30}>
    {
      halp_meta(
          description,
          "Track IDs: how many frames a lost track is kept alive (Kalman-coasted) "
          "before it is dropped (~ frames-per-second = ~1s at 30). Higher = an id "
          "survives longer occlusions but may stick to a stale position.");
    } track_memory;

    struct : halp::spinbox_i32<"Re-ID Memory", halp::range{0, 18000, 1800}>
    {
      halp_meta(
          description,
          "Re-ID: how many frames a DEPARTED person's appearance is remembered "
          "so they re-acquire their id when they return to frame (~30-60s at 30 "
          "fps). 0 = no long-term gallery. Needs Re-ID on.");
    } reid_memory;

    struct : halp::hslider_f32<"Re-ID Margin", halp::range{0., 0.5, 0.1}>
    {
      halp_meta(
          description,
          "Re-ID ambiguity guard: the best remembered identity must beat the "
          "second-best by at least this appearance margin before a returning "
          "person re-acquires its id. Higher = stricter, avoids swapping "
          "near-identical people (same uniform); 0 = off.");
    } reid_margin;

    // Appended last to keep existing presets' port indices stable (id 30).
    struct : halp::spinbox_i32<"Detection Hold", halp::range{0, 60, 6}>
    {
      halp_meta(
          description,
          "Keep showing the last detection for up to this many frames after it "
          "is momentarily lost (a detector miss or a low-confidence frame) "
          "instead of blanking immediately. Bridges 1-2 frame dropouts so "
          "poses/boxes stop blinking. Applies to single detections and (as the "
          "coast window) to tracked instances. 0 = blank immediately.");
    } hold_frames;

    // --- Body mesh (appended last to keep existing presets' port indices
    //     stable). Only models with body parameters use them (InstantHMR). ---
    // Binary (mmap view): a text view would mangle it, and the fine LODs are large.
    struct : halp::file_port<"Body Model", halp::mmap_file_view>
    {
      halp_meta(extensions, "*.mhrbin");
      halp_meta(
          description,
          "An MHR body model (.mhrbin, made from Meta's MHR assets with "
          "tools/mhr_export.py; the LOD is the file's). Turns InstantHMR's "
          "body parameters into each person's mesh, for the Mesh Data Formats, "
          "Draw Mesh and Mesh Keypoints. Empty = no mesh.");
    } body_model;

    struct : halp::toggle<"Mesh Keypoints">
    {
      halp_meta(
          description,
          "With a Body Model: replace the 3D keypoints (and the 2D ones, by "
          "projecting through the model's camera) with the MHR70 keypoints of "
          "the mesh, so they sit exactly on it. Off = the model's own "
          "keypoint heads (a few cm apart from the mesh).");
      bool value = false;
    } mesh_keypoints;

    struct : halp::enum_t<MeshSpace, "Mesh Space">
    {
      halp_meta(
          description,
          "Axes of the mesh Data Formats: OpenGL (Y up, Z toward the viewer, "
          "like score's 3D scenes) or Camera (OpenCV: Y down, Z forward, like "
          "Camera XYZ). Metres, camera at the origin, in both.");
    } mesh_space;

    struct : halp::toggle<"Draw Mesh">
    {
      halp_meta(
          description,
          "With a Body Model: draw each person's mesh (shaded, in its track "
          "colour) on the output image, under the skeleton.");
      bool value = false;
    } draw_mesh;

  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

    struct
    {
      halp_meta(name, "Detection");
      std::optional<DetectedPose> value;
    } detection;

    struct
    {
      halp_meta(name, "Geometry");
      std::vector<float> value;
    } geometry;

    // --- Multi-instance outputs (populated when Track IDs is on) ---
    struct
    {
      halp_meta(name, "Poses");
      std::vector<DetectedPose> value; // every tracked instance, each w/ track_id
    } poses;

    struct
    {
      halp_meta(name, "Poses Geometry");
      // Fixed-stride, zero-padded: Max Instances slots, each holding one
      // instance's geometry in the current Data Format — the same payload the
      // Geometry outlet emits for a single pose. Purely geometric (track ids and
      // classes live on the Poses outlet) EXCEPT in the Flattened format, which
      // is the legacy [track_id, class_id, box, keypoints] record, and
      // BodyParams (rig origin + parametric body parameters). The stride is
      // the longest payload of the frame, so slot i always starts at
      // i * (size / Max Instances), and count gives the number of live slots.
      std::vector<float> value;
    } poses_geometry;

    struct
    {
      halp_meta(name, "Count");
      int value{}; // number of live tracked instances this frame
    } count;
  } outputs;

  // Inspector layout: group the controls into meaningful tabs.
  struct ui
  {
    halp_meta(name, "Pose Detector")
    halp_meta(layout, halp::layouts::tabs)
    halp_meta(background, halp::colors::background_mid)

    struct
    {
      halp_meta(name, "Models")
      halp_meta(layout, halp::layouts::vbox)
      halp::item<&ins::model> model;
      halp::item<&ins::det_model> det_model;
      halp::item<&ins::workflow> workflow;
      halp::item<&ins::detection_class> detection_class;
      halp::item<&ins::class_file> class_file;
      halp::item<&ins::body_model> body_model;
    } models_tab;

    // Tracking tab, split into three columns: core tracking, ROI feedback, and
    // the plausibility gates.
    struct
    {
      halp_meta(name, "Tracking")
      halp_meta(layout, halp::layouts::hbox)

      struct
      {
        halp_meta(name, "Tracking")
        halp_meta(layout, halp::layouts::vbox)
        // Min Confidence belongs with tracking: it is the detection-acceptance
        // threshold that gates which instances are tracked.
        halp::item<&ins::min_confidence> min_confidence;
        halp::item<&ins::track_ids> track_ids;
        halp::item<&ins::max_instances> max_instances;
        halp::item<&ins::track_memory> track_memory;
        halp::item<&ins::hold_frames> hold_frames;
      } core;

      halp::spacing sp1{.width = 12, .height = 1};

      struct
      {
        halp_meta(name, "ROI")
        halp_meta(layout, halp::layouts::vbox)
        halp::item<&ins::track_roi> track_roi;
        halp::item<&ins::detector_cadence> detector_cadence;
      } roi;

      halp::spacing sp2{.width = 12, .height = 1};

      struct
      {
        halp_meta(name, "Gates")
        halp_meta(layout, halp::layouts::vbox)
        halp::item<&ins::motion_gate> motion_gate;
        halp::item<&ins::max_speed> max_speed;
        halp::item<&ins::birth_gate> birth_gate;
        halp::item<&ins::strict_confirm> strict_confirm;
      } gates;
    } tracking_tab;

    struct
    {
      halp_meta(name, "Re-ID")
      halp_meta(layout, halp::layouts::vbox)
      halp::item<&ins::reid_model> reid_model;
      halp::item<&ins::reid> reid;
      halp::item<&ins::reid_weight> reid_weight;
      halp::item<&ins::reid_preprocess> reid_preprocess;
      halp::item<&ins::reid_memory> reid_memory;
      halp::item<&ins::reid_margin> reid_margin;
    } reid_tab;

    struct
    {
      halp_meta(name, "Smoothing")
      halp_meta(layout, halp::layouts::vbox)
      halp::item<&ins::smoothing> smoothing;
      halp::item<&ins::smoothing_amount> smoothing_amount;
    } smoothing_tab;

    // Preview: everything that only changes the "Out" image (what is drawn and
    // on what background). None of it touches the data outlets.
    struct
    {
      halp_meta(name, "Preview")
      halp_meta(layout, halp::layouts::vbox)
      halp::item<&ins::output_mode> output_mode;
      halp::item<&ins::draw_skeleton> draw_skeleton;
      halp::item<&ins::draw_landmarks> draw_landmarks;
      halp::item<&ins::draw_boxes> draw_boxes;
      halp::item<&ins::draw_mesh> draw_mesh;
    } preview_tab;

    // Output is the LAST tab: what the data outlets carry (layout, skeleton,
    // mesh space and keypoint source), set once and forgotten. Skeleton lives
    // here because it defines the emitted joint layout; the preview follows it.
    struct
    {
      halp_meta(name, "Output")
      halp_meta(layout, halp::layouts::vbox)
      halp::item<&ins::data_format> data_format;
      halp::item<&ins::skeleton_type> skeleton_type;
      halp::item<&ins::mesh_space> mesh_space;
      halp::item<&ins::mesh_keypoints> mesh_keypoints;
    } output_tab;
  };

  PoseDetector() noexcept;
  ~PoseDetector();

  void operator()();

private:
  // --- Two-stage building blocks ---
  // Stage 1: run the detector on the full frame, return detections in
  // image-normalized [0,1] coordinates (letterbox removed).
  // keep_class: -2 = use the target-domain default class filter (person/animal),
  // -1 = keep all classes, >=0 = keep only that class id.
  // ctx_override runs the detector on a specific context instead of det_ctx
  // (used when a detector model sits in the LANDMARK port, e.g. RetinaFace alone
  // drawn as a 5-keypoint pose — its session is `ctx`, not `det_ctx`).
  // score_thr < 0 keeps each detector family's built-in quality floor; >= 0
  // raises the accept threshold to max(floor, score_thr) — how Min Confidence
  // (with hysteresis, see detThreshold) reaches the stage-1 detector.
  std::vector<Onnx::Detection::Detection> runDetector(
      const Onnx::ModelRole& role, const Onnx::ImageView& src,
      Onnx::ModelDomain target = Onnx::ModelDomain::Body, int keep_class = -2,
      Onnx::OnnxRunContext* ctx_override = nullptr, float score_thr = -1.f);

  // Stage 2: run the landmark/pose model on the crop defined by M
  // (crop-pixels -> image-pixels), then map keypoints back through M.
  // crop_padded: the ROI is a padded detector crop (two-stage / multi-instance),
  // so a keypoint pegged to the crop edge is a SimCC/heatmap decode artifact to
  // drop. False for whole-frame, where edge keypoints can be real.
  void runLandmark(
      const Onnx::ModelRole& role, PoseWorkflow draw, const Onnx::ImageView& src,
      const Onnx::Affine& M, int track_id = -1, bool crop_padded = false);

  // Stage 2 core: run the landmark model on one crop and decode keypoints into
  // `out` (image-normalized [0,1]); no smoothing/drawing/output. Returns the
  // mean confidence, or -1 on decode failure. Shared by single + multi paths.
  // `out_world` (when non-null) receives the model's world-space 3D keypoints
  // (meters, hip-origin; BlazePose family) — empty if the model has none.
  // det_score: the detector score of the instance the crop came from (1 for
  // the whole frame). Models without per-joint confidence (InstantHMR) use it
  // as their confidence; the others ignore it. A model with a camera
  // translation / body parameters leaves them in m_trans_scratch /
  // m_params_scratch (cleared otherwise).
  float landmarkKeypoints(
      const Onnx::ModelRole& role, const Onnx::ImageView& src, const Onnx::Affine& M,
      std::vector<PoseKeypoint>& out, std::vector<PoseKeypoint>* out_world = nullptr,
      bool crop_padded = false, float det_score = 1.f);

  // --- Multi-instance back-end (Track IDs path) ---
  // Run the detector (top-K) or per-track ROIs, landmark each, track, per-id
  // smooth, draw all, and fill the poses / primary / geometry / count outputs.
  void runMultiInstance(
      const Onnx::ModelRole& role, PoseWorkflow draw, const Onnx::ImageView& src);
  // Landmark every ROI into m_instances, batching all crops into one inference
  // when the model's batch dim allows it (else one inference per crop).
  // m_roi_scores[i] is the detector score of rois[i] (see landmarkKeypoints).
  void runLandmarkBatch(
      const Onnx::ModelRole& role, PoseWorkflow draw, const Onnx::ImageView& src,
      const std::vector<Onnx::ROI::Rect>& rois);
  // Track m_instances, assign ids + per-id smoothed keypoints + colors, then
  // draw all and publish every output port.
  void emitInstances(PoseWorkflow draw, bool do_track);
  // Crop each entry of m_track_in, run the Re-ID model (batched if it allows),
  // and write an L2-normalized embedding into each m_track_in[i].embedding.
  void embedInstances();

  // Draw a detector's own keypoints/box directly (detector used standalone).
  void runDetectorAsPose(const Onnx::ModelRole& role, const Onnx::ImageView& src);

  // Single-stage YOLO-pose on the full frame.
  void runYOLOPose(const Onnx::ImageView& src, const Onnx::Affine& M);

  // Single-stage RTMO on the full frame (dets + keypoints, NMS-free).
  void runRTMO(const Onnx::ImageView& src);

  // Detection-only: run the Detection Model, emit boxes as keypoint-less poses
  // (optionally tracked). Drawn as rectangles; box lives in each pose's metadata.
  void runBoxDetection(const Onnx::ModelRole& detRole, const Onnx::ImageView& src);

  // Map a manual workflow selection onto a concrete model role.
  Onnx::ModelRole roleForWorkflow(PoseWorkflow w) const;
  // InstantHMR per-model setup (output indices, CLIFF form, image size, body
  // params smoothing kinds); clears it for any other landmark model.
  void loadInstantHmr(Onnx::OnnxRunContext& lctx, Onnx::ModelRole& role);
  // Smoothing kind of each MHR pose parameter (m_params_angle_mask[0:204]):
  // from the Body Model's parameter transform when one is loaded, else the
  // same derivation hardcoded. No-op unless an InstantHMR model is loaded.
  void setHmrParamKinds();

public:
  // The body_params smoothing kinds (Onnx::ParamKind per entry), for tests.
  std::span<const std::uint8_t> paramKindsForTest() const noexcept;

private:

  // Common visualization. mesh_slot: the pose's evaluated body mesh (an index
  // into m_mesh_slots, see PoseDetector_mesh.cpp), -1 = none.
  void drawSkeleton(
      const DetectedPose& pose, PoseWorkflow workflow, int mesh_slot = -1);
  // Draw all of m_instances onto one output image (multi-instance path).
  void drawAllSkeletons(PoseWorkflow workflow);
  // Draw one pose's connections + points into an open ctx overlay.
  void drawOnePose(
      Overlay& ov, const DetectedPose& pose, PoseWorkflow workflow, int w,
      int h);
  void generateGeometryOutput(
      const DetectedPose& pose, PoseWorkflow workflow, int mesh_slot = -1);

  // Skeleton remap: set m_remap_* from the control + workflow, remap a pose's
  // keypoints in place (native -> target), and finalize a single-instance pose
  // (remap + draw + geometry). Remap drives both overlay and output ports.
  void setRemapState(PoseWorkflow workflow, int num_kps);
  void remapPose(DetectedPose& pose);
  void finalizeSingle(PoseWorkflow workflow);
  // Append one pose's flattened geometry (current Data Format) to `out`.
  void appendGeometry(
      std::vector<float>& out, const DetectedPose& pose, PoseWorkflow workflow,
      int mesh_slot = -1);
  void passthrough(const Onnx::ImageView& src);
  // Class id -> display name (custom file, else built-in COCO-80, else null).
  const char* className(int id) const;
  // Death-gate helpers (Track IDs path): age the tracker on a no-detection
  // frame, re-emit still-live coasted poses, or fall back to passthrough.
  void ageTracker();
  bool reEmitCoasted(PoseWorkflow draw);
  void coastOrPassthrough(PoseWorkflow draw, const Onnx::ImageView& src);
  // Single-instance equivalent of coastOrPassthrough: re-emit the last good pose
  // for up to Detection-Hold frames on a transient miss, else passthrough.
  void holdOrPassthrough(const Onnx::ImageView& src);

  // Detector accept score with enter/exit hysteresis: the user's Min Confidence
  // is the enter threshold; once a detection is live we hold marginal ones down
  // to a lower exit threshold so a subject hovering near the cut stops blinking.
  float detThreshold() const noexcept
  {
    const float enter
        = std::clamp(static_cast<float>(inputs.min_confidence.value), 0.05f, 0.95f);
    return m_had_detection ? 0.66f * enter : enter;
  }

  // The class the two-stage detector keeps: the Detection Class port when set,
  // else -2, the landmark model's domain default.
  int detectorClass() const noexcept
  {
    const int c = static_cast<int>(inputs.detection_class.value);
    return c >= 0 ? c : -2;
  }

  // Temporal One-Euro smoothing of the detected keypoints (in place), and of
  // the 3D payload (world / translation / body_params, via smooth3D).
  // conf_hold: also blend each joint toward its previous position by (1 -
  // confidence). Off for models whose confidence is one instance-wide score
  // (InstantHMR: the detector's), where it would only be a uniform extra lag.
  void applySmoothing(DetectedPose& pose, bool conf_hold = true);
  // Single-instance One-Euro of the 3D payload in metres/radians (see
  // smoothing3D); the multi-instance path smooths per track id instead.
  void smooth3D(DetectedPose& pose);
  // Drop the single-instance temporal filter state (2D + 3D), e.g. on a model
  // change or a sustained loss. Alloc-free: the vectors keep their capacity.
  void resetSmoothers();

  // ROI as a rect (image px) from a detection, by landmark kind.
  Onnx::ROI::Rect
  detectionRect(const Onnx::ModelRole& role,
                const Onnx::Detection::Detection& det, int W, int H);
  // ROI rect derived from the previous frame's landmarks (tracking loop).
  Onnx::ROI::Rect roiRectFromKeypoints(
      PoseWorkflow draw, const std::vector<PoseKeypoint>& kps, int W, int H,
      int model_w, int model_h);
  // Temporally smooth the ROI rect so the crop stays steady.
  Onnx::ROI::Rect smoothRoi(Onnx::ROI::Rect r);

  // --- Body mesh (PoseDetector_mesh.cpp) ---
  // Parse the Body Model file when its name changes (allocates; never per
  // frame). An unreadable file logs once and leaves the mesh off.
  void loadBodyModel();
  // What this frame's outputs need from the mesh: vertices (a mesh Data
  // Format, Draw Mesh) and/or MHR70 keypoints (Mesh Keypoints). All false =
  // no body model, or nothing asks for it: the mesh costs nothing.
  struct MeshNeeds
  {
    bool verts = false, keypoints = false;
    bool any() const noexcept { return verts || keypoints; }
  };
  MeshNeeds meshNeeds() const noexcept;
  // Evaluate one pose's mesh into m_mesh_slots[slot] and, with Mesh
  // Keypoints, move its world / 2D keypoints onto the mesh (native joint
  // order: call before the Skeleton remap). False = no mesh for this pose.
  bool evaluateMesh(DetectedPose& pose, int slot, const MeshNeeds& needs);
  // Multi-instance: a slot per m_instances entry, reused by track id, then
  // evaluate; fills m_inst_mesh. Single path: slot 0 -> m_single_mesh.
  void evaluateInstanceMeshes();
  void evaluateSingleMesh(DetectedPose& pose);
  int claimMeshSlot(int track_id);
  // Rasterize the given slots' meshes onto the output image (Draw Mesh).
  void drawMeshes(unsigned char* dst, int w, int h, std::span<const int> slots);

  std::unique_ptr<Onnx::OnnxRunContext> ctx;     // main / landmark model
  std::unique_ptr<Onnx::OnnxRunContext> det_ctx; // optional stage-1 detector
  std::unique_ptr<Onnx::OnnxRunContext> reid_ctx; // optional appearance ReID
  boost::container::vector<float> storage;
  boost::container::vector<float> det_storage;

  // Cache for avoiding re-initialization
  PoseWorkflow m_last_workflow{PoseWorkflow::Auto};
  Onnx::ModelRole m_landmark_role;
  Onnx::ModelRole m_detector_role;
  Onnx::PoseSmoother m_smoother;
  // Single-instance 3D payload smoothers (the tracker owns the per-id ones).
  Onnx::PoseSmoother m_world_smoother;  // 3 filters per world joint
  Onnx::PoseSmoother m_trans_smoother;  // translation (3)
  Onnx::PoseSmoother m_params_smoother; // body_params
  Onnx::PoseSmoother m_offset_smoother; // rig_offset (3), world's tuning
  // One Onnx::ParamKind per DetectedPose::body_params entry: Angle (rad,
  // smoothed with unwrapping), Static (identity: a slow pure low-pass), else
  // Linear (Onnx::smoothParams). Model-specific, so set by the family that
  // fills body_params when its model loads; empty = all linear. Its size must
  // match body_params for the mask to apply.
  std::vector<std::uint8_t> m_params_angle_mask;
  // beta anchor (per unit/s at Smoothing Amount 0.5) of the body_params
  // One-Euro, set with the mask (see smoothing3D): 0 = pure low-pass.
  float m_params_beta{0.f};

  // --- InstantHMR (human mesh recovery), resolved once at model load ---
  Onnx::HmrOutputs m_hmr_out;      // output indices of the loaded graph
  bool m_hmr_cliff_focal{false};   // metadata cliff_focal: angular CLIFF form
  // Detector score of the single-path crop, stashed before runLandmark (1 for
  // the whole frame / a tracking ROI): InstantHMR's joint confidence.
  float m_landmark_score{1.f};

  // Multi-instance persistent-ID tracker (ByteTrack-style; opt-in via Track IDs).
  Onnx::Track::PoseTracker m_tracker;

  // Tracking-loop state (two-stage path)
  bool m_tracking{false};               // valid ROI carried from prev frame?
  Onnx::RectSmoother m_roi_smoother;    // temporal ROI stabilization
  std::vector<PoseKeypoint> m_last_keypoints; // image-normalized, prev frame
  // Pre-remap (native-skeleton) snapshot of the last single-pose keypoints. The
  // tracking-ROI loop indexes NATIVE joint ids (e.g. BlazePose hips 23/24), so
  // it must not see keypoints that finalizeSingle() remapped to another layout.
  std::vector<PoseKeypoint> m_native_keypoints;
  // Previous frame's smoothed keypoints (single-instance), for the
  // confidence-weighted hold that tames low-confidence joint teleports.
  std::vector<PoseKeypoint> m_prev_smoothed_kps;
  int m_lost_frames{0};                 // consecutive frames without a pose
  Onnx::ROI::Rect m_prev_roi{};         // last smoothed ROI (plausibility check)
  bool m_have_prev_roi{false};

  // Single-instance visible hold + box stabilization (the multi-instance path
  // coasts via the tracker; these are the single-instance equivalents).
  std::optional<DetectedPose> m_last_single_pose; // last good pose, for re-emit
  PoseWorkflow m_last_single_draw{PoseWorkflow::Auto};
  int m_hold_frames{0};                 // consecutive re-emitted (held) frames
  bool m_had_detection{false};          // a detection is live (hysteresis)
  Onnx::RectSmoother m_box_smoother;    // temporal One-Euro on the output box

  Onnx::ReidSpec m_reid_spec;
  boost::container::vector<float> m_reid_batch; // packed [N,3,H,W] reid input
  boost::container::vector<float> m_reid_tmp;   // per-crop reid build scratch

  std::string m_last_model;
  std::string m_last_det_model;
  // YoloxDetector input range, probed per detection model (see runDetector).
  enum class YoloxRange : uint8_t { Unknown, Unit, Raw } m_yolox_range{};
  std::string m_last_reid_model;
  std::string m_last_cfg_log; // de-dup for the debug config trace

  // Custom box class names, loaded once when the Class Names File path changes.
  std::vector<std::string> m_class_names;
  std::string m_last_class_file;

  // --- Cached scratch reused every frame (zero steady-state allocation) ---
  std::vector<DetectedPose> m_instances;          // this frame's instances
  std::vector<PoseKeypoint> m_kp_scratch;         // landmark decode -> keypoints
  std::vector<PoseKeypoint> m_world_scratch;      // landmark decode -> world 3D
  std::vector<Onnx::Track::Detection> m_track_in; // tracker input
  std::vector<Onnx::Detection::Detection> m_dets; // detector output (top-K)
  std::vector<std::pair<float, int>> m_box_sel;   // (score,idx) top-K box select
  std::vector<Onnx::ROI::Rect> m_rois;            // ROIs to landmark this frame
  std::vector<float> m_roi_scores;                // detector score per m_rois
  std::vector<float> m_trans_scratch;             // landmark decode -> translation
  std::vector<float> m_params_scratch;            // landmark decode -> body params
  std::vector<float> m_geom_scratch;  // packed per-instance geometry payloads
  std::vector<int> m_geom_ends;       // end offset of each payload in the above
  boost::container::vector<float> m_batch_storage; // packed [N,C,H,W] input
  std::vector<int64_t> m_bbox;                     // batched SimCC [N,2] bbox
  std::vector<float> m_cliff;                      // InstantHMR [N,3] CLIFF input
  // The landmark model declared a dynamic batch but failed a batched run:
  // per-crop from then on (reset when a landmark model is loaded).
  bool m_landmark_no_batch{false};
  // GPU provider: pad the landmark batch to the largest person count seen so
  // far (m_landmark_batch_cap), so a changing count doesn't re-plan the graph
  // every time (see runLandmarkBatch). Reset when a landmark model is loaded.
  bool m_landmark_pad_batch{false};
  int m_landmark_batch_cap{0};
  int m_frames_since_detect{0};                   // detector-cadence counter

  // Detector anchor/prior caches: the geometry only depends on the model, so
  // regenerating it per frame would heap-allocate tens of KB at frame rate.
  std::vector<Onnx::Vec2> m_ssd_anchors;
  Onnx::ModelKind m_ssd_kind{};
  int m_ssd_input{0}, m_ssd_num_boxes{-1};
  std::vector<Onnx::Detection::PriorBox> m_prior_cache;
  Onnx::ModelKind m_prior_kind{};
  int m_prior_w{0}, m_prior_h{0};

  // Skeleton remap state (recomputed per emit from the Skeleton control).
  bool m_remap_active{false};
  Onnx::Skel::SourceSkeleton m_remap_src{Onnx::Skel::SourceSkeleton::Coco17};
  Onnx::Skel::TargetSkeleton m_active_target{Onnx::Skel::TargetSkeleton::Native};
  std::vector<PoseKeypoint> m_remap_scratch; // reused remap output buffer
  std::vector<float> m_offset_scratch; // landmark decode -> rig_offset

  // --- Body mesh (MHR) state, see PoseDetector_mesh.cpp ---
  // One per simultaneously evaluated person: the MHR workspace (with its
  // cached identity blend, so a person keeps its slot across frames by
  // track id) and that person's evaluated mesh. Sized lazily up to Max
  // Instances, then reused: no allocation in the steady state.
  struct MeshSlot
  {
    Onnx::Mhr::Workspace ws;
    std::vector<float> verts;   // V*3, camera space (OpenCV axes), metres
    std::vector<float> kp;      // 70*3 rig-local MHR70 (Mesh Keypoints only)
    int track_id = -1;          // the person whose identity ws caches
    int claimed = -1;           // m_mesh_frame of the last use (LRU)
    bool has_verts = false;
  };
  // Heap-held: every Workspace keeps a pointer to the Model it was sized for.
  std::unique_ptr<Onnx::Mhr::Model> m_mhr;
  std::string m_last_body_model;
  const char* m_last_body_data{}; // the mapping it was loaded from
  size_t m_last_body_size{};
  std::vector<MeshSlot> m_mesh_slots;
  std::vector<int> m_inst_mesh; // per m_instances entry: its slot, or -1
  int m_single_mesh{-1};        // slot of the single-path pose (held too)
  int m_mesh_frame{0};
  Onnx::MeshOps::Raster m_mesh_raster; // Draw Mesh depth + colour layer
};

} // namespace OnnxModels
