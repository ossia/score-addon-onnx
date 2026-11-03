#pragma once
#include <Onnx/helpers/Detection.hpp>
#include <Onnx/helpers/ImageOps.hpp>

#include <algorithm>
#include <cmath>

// Region-of-interest construction shared by every two-stage pose pipeline.
//
// All ROIs collapse to a single affine that maps a point in the landmark
// model's input space (pixels, [0,model_w]x[0,model_h]) to the original image
// (pixels). The same affine:
//   - warps the source image into the model crop (warpAffine samples src at it),
//   - maps decoded keypoints back to the image (applyAffine).
// See TWO_STAGE_ARCHITECTURE.md §3.
namespace Onnx::ROI
{

// Rotated MediaPipe-style ROI (DetectionsToRects + RectTransformation).
struct MediapipeParams
{
  int kp_start = 0;             // rotation vector start keypoint index
  int kp_end = 1;              // rotation vector end keypoint index
  float target_angle_deg = 90.0f; // pose/hand: 90, face: 0
  float scale = 1.25f;         // expansion factor (square_long)
  float shift_x = 0.0f;
  float shift_y = 0.0f;
  bool alignment_points = false; // pose: size = 2*dist(start,end); else bbox
};

// pose_detection -> pose_landmark ROI
inline MediapipeParams poseParams()
{
  return MediapipeParams{
      .kp_start = 0, .kp_end = 1, .target_angle_deg = 90.0f, .scale = 1.25f,
      .alignment_points = true};
}
// palm_detection -> hand_landmark ROI
inline MediapipeParams handParams()
{
  return MediapipeParams{
      .kp_start = 0, .kp_end = 2, .target_angle_deg = 90.0f, .scale = 2.6f,
      .shift_y = -0.5f, .alignment_points = false};
}
// face_detection -> face_landmark ROI (FaceMesh: a bit of margin around the box)
inline MediapipeParams faceParams()
{
  return MediapipeParams{
      .kp_start = 0, .kp_end = 1, .target_angle_deg = 0.0f, .scale = 1.5f,
      .alignment_points = false};
}
// face_detection -> MobileFaceNet (dlib-68) ROI. MobileFaceNet is trained on a
// TIGHT aligned face crop, so a loose 1.5x ROI makes its landmarks overshoot/
// offset the real face. Keep the crop close to the detection box.
inline MediapipeParams mobileFaceParams()
{
  return MediapipeParams{
      .kp_start = 0, .kp_end = 1, .target_angle_deg = 0.0f, .scale = 1.1f,
      .alignment_points = false};
}

// A region of interest in image pixels: center, full width/height, rotation.
// This is the smoothable intermediate — temporally filter these 5 scalars, then
// rebuild the affine, so the crop the landmark model sees stays stable.
struct Rect
{
  float cx{}, cy{}, w{}, h{}, angle{}; // angle in radians
};

// Rect (image px) -> crop(model px)->image(px) affine (dst->src; identical to
// the old QTransform, so it also maps decoded keypoints back via applyAffine).
inline Affine rectToAffine(const Rect& r, int model_w, int model_h)
{
  return affineFromRoi(r.cx, r.cy, r.w, r.h, r.angle, model_w, model_h);
}

// Apply an affine (model px -> image px) to a point.
inline Vec2 applyAffine(const Affine& a, float x, float y)
{
  return {a.m0 * x + a.m1 * y + a.m2, a.m3 * x + a.m4 * y + a.m5};
}

// Rotated MediaPipe-style ROI rect from a normalized detection.
inline Rect mediapipeRect(
    const Detection::Detection& det, int srcW, int srcH,
    const MediapipeParams& p)
{
  const auto& kps = det.keypoints;
  const int ke = std::min<int>(p.kp_end, static_cast<int>(kps.size()) - 1);
  const int ks = std::min<int>(p.kp_start, static_cast<int>(kps.size()) - 1);

  float cx, cy, size;
  float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  if(ks >= 0 && ke >= 0)
  {
    x0 = kps[ks].x * srcW;
    y0 = kps[ks].y * srcH;
    x1 = kps[ke].x * srcW;
    y1 = kps[ke].y * srcH;
  }

  const float target = p.target_angle_deg * float(M_PI) / 180.0f;
  const float angle = (ks >= 0 && ke >= 0)
                          ? target - std::atan2(-(y1 - y0), (x1 - x0))
                          : 0.0f;

  if(p.alignment_points && ks >= 0 && ke >= 0)
  {
    cx = x0;
    cy = y0;
    const float dx = x1 - x0, dy = y1 - y0;
    size = 2.0f * std::sqrt(dx * dx + dy * dy);
  }
  else
  {
    cx = det.xc * srcW;
    cy = det.yc * srcH;
    size = std::max(det.w * srcW, det.h * srcH); // square_long
  }

  const float ca = std::cos(angle), sa = std::sin(angle);
  cx += ca * (p.shift_x * size) - sa * (p.shift_y * size);
  cy += sa * (p.shift_x * size) + ca * (p.shift_y * size);

  return Rect{cx, cy, size * p.scale, size * p.scale, angle};
}

// hand_landmark -> next hand_landmark ROI (MediaPipe HandLandmarksToRect +
// RectTransformation scale 2.0, shift_y -0.1, square_long). `kp(i)` gives
// landmark i of the 21 in image pixels. The rotation runs from the wrist to
// the mean of the middle MCP and the index/ring MCP midpoint; the size is the
// bounding box of the palm and lower-finger joints in that rotated frame, so
// a rotated hand does not get the larger axis-aligned box.
template <typename KpFn>
inline Rect handRectFromLandmarks(KpFn&& kp)
{
  static constexpr int partial[] = {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18};
  const Vec2 w = kp(0), i5 = kp(5), m9 = kp(9), r13 = kp(13);
  const float x1 = ((i5.x + r13.x) * 0.5f + m9.x) * 0.5f;
  const float y1 = ((i5.y + r13.y) * 0.5f + m9.y) * 0.5f;
  const float angle
      = float(M_PI) * 0.5f - std::atan2(-(y1 - w.y), x1 - w.x);

  float minx = 1e30f, miny = 1e30f, maxx = -1e30f, maxy = -1e30f;
  for(int i : partial)
  {
    const Vec2 p = kp(i);
    minx = std::min(minx, p.x);
    maxx = std::max(maxx, p.x);
    miny = std::min(miny, p.y);
    maxy = std::max(maxy, p.y);
  }
  const float acx = 0.5f * (minx + maxx), acy = 0.5f * (miny + maxy);

  // Bounds of the landmarks rotated by -angle about the axis-aligned centre.
  const float rc = std::cos(-angle), rs = std::sin(-angle);
  float pminx = 1e30f, pminy = 1e30f, pmaxx = -1e30f, pmaxy = -1e30f;
  for(int i : partial)
  {
    const Vec2 p = kp(i);
    const float ox = p.x - acx, oy = p.y - acy;
    const float px = ox * rc - oy * rs, py = ox * rs + oy * rc;
    pminx = std::min(pminx, px);
    pmaxx = std::max(pmaxx, px);
    pminy = std::min(pminy, py);
    pmaxy = std::max(pmaxy, py);
  }
  const float pcx = 0.5f * (pminx + pmaxx), pcy = 0.5f * (pminy + pmaxy);
  const float c = std::cos(angle), s = std::sin(angle);
  float cx = pcx * c - pcy * s + acx;
  float cy = pcx * s + pcy * c + acy;
  const float bw = pmaxx - pminx, bh = pmaxy - pminy;

  constexpr float shift_y = -0.1f, scale = 2.0f;
  cx += -s * (shift_y * bh);
  cy += c * (shift_y * bh);
  const float size = std::max(bw, bh) * scale;
  return Rect{cx, cy, size, size, angle};
}

// A whole-hand box (RTMDet-hand, a body-head-hand detector) gives no
// orientation: the MediaPipe hand landmark model gets an upright square crop
// this much larger than the box's long side, then one more pass on the
// landmark-derived rect (handRectFromLandmarks) to rotate and fit it.
inline constexpr float kHandBoxCropScale = 1.6f;
// A person box (no alignment points) -> BlazePose landmark crop: the square
// around the box, scaled like the pose detector's ROI.
inline constexpr float kPoseBoxCropScale = 1.25f;

// Axis-aligned top-down (mmpose) ROI rect from a source-pixel bbox (top-left
// x,y,w,h).
inline Rect topdownRect(
    const Onnx::Rect& bbox_px, int model_w, int model_h, float pad)
{
  const float cx = bbox_px.x + bbox_px.w * 0.5f;
  const float cy = bbox_px.y + bbox_px.h * 0.5f;
  float sw = bbox_px.w * pad;
  float sh = bbox_px.h * pad;
  const float a = float(model_w) / float(model_h);
  if(sw > sh * a)
    sh = sw / a;
  else
    sw = sh * a;
  return Rect{cx, cy, sw, sh, 0.0f};
}

inline Affine mediapipeAffine(
    const Detection::Detection& det, int srcW, int srcH, int model_w,
    int model_h, const MediapipeParams& p)
{
  return rectToAffine(mediapipeRect(det, srcW, srcH, p), model_w, model_h);
}

inline Affine topdownAffine(
    const Onnx::Rect& bbox_px, int model_w, int model_h, float pad = 1.25f)
{
  return rectToAffine(topdownRect(bbox_px, model_w, model_h, pad), model_w, model_h);
}

// Whole-frame affine matching ImageOps cover-resize (KeepAspectRatioByExpanding
// + center-crop) — used for keypoint mapback when no detector is present.
inline Affine wholeFrameAffine(int srcW, int srcH, int model_w, int model_h)
{
  const float scale = std::max(float(model_w) / srcW, float(model_h) / srcH);
  const float scaled_w = srcW * scale, scaled_h = srcH * scale;
  const float crop_x = (scaled_w - model_w) * 0.5f;
  const float crop_y = (scaled_h - model_h) * 0.5f;
  // img_px = (m_px + crop) / scale
  return Affine{
      1.0f / scale, 0.f, crop_x / scale, 0.f, 1.0f / scale, crop_y / scale};
}

// MoveNet's crop around a person (upstream's determine_crop_region): a square
// centred on the hips, large enough for the torso (1.9x its extent) and every
// confident joint (1.2x); none when the torso is not visible or the square
// would exceed the frame, where upstream falls back to the whole frame.
// `kp(i)` is COCO-17 joint i in image pixels with its score.
struct ScoredPoint
{
  float x, y, score;
};
template <typename KpFn>
inline bool moveNetCropRect(KpFn&& kp, int W, int H, Rect& out)
{
  constexpr float min_score = 0.2f;
  auto ok = [&](int i) { return kp(i).score > min_score; };
  if(!((ok(11) || ok(12)) && (ok(5) || ok(6))))
    return false;
  const float cx = (kp(11).x + kp(12).x) * 0.5f;
  const float cy = (kp(11).y + kp(12).y) * 0.5f;
  float tx = 0.f, ty = 0.f, bx = 0.f, by = 0.f;
  for(int i : {5, 6, 11, 12})
  {
    tx = std::max(tx, std::abs(cx - kp(i).x));
    ty = std::max(ty, std::abs(cy - kp(i).y));
  }
  for(int i = 0; i < 17; i++)
    if(ok(i))
    {
      bx = std::max(bx, std::abs(cx - kp(i).x));
      by = std::max(by, std::abs(cy - kp(i).y));
    }
  float half = std::max({tx * 1.9f, ty * 1.9f, bx * 1.2f, by * 1.2f});
  half = std::min(half, std::max({cx, W - cx, cy, H - cy}));
  if(!(half > 0.f) || half > std::max(W, H) * 0.5f)
    return false;
  out = Rect{cx, cy, 2.f * half, 2.f * half, 0.f};
  return true;
}

// Whole-frame affine of a centred letterbox: the whole frame fits in the
// model input, padded on the short side, as Ultralytics' letterbox and
// MoveNet's resize_with_pad feed it. Nothing of the frame is cropped away.
inline Affine wholeFrameFitAffine(int srcW, int srcH, int model_w, int model_h)
{
  const float scale = std::min(float(model_w) / srcW, float(model_h) / srcH);
  const float pad_x = (model_w - srcW * scale) * 0.5f;
  const float pad_y = (model_h - srcH * scale) * 0.5f;
  // img_px = (m_px - pad) / scale
  return Affine{
      1.0f / scale, 0.f, -pad_x / scale, 0.f, 1.0f / scale, -pad_y / scale};
}

// Render the source image into dst (a model-sized RGBA buffer) through M
// (model px -> image px), bilinear. Replaces the old QPainter warpCrop.
inline void warpCrop(
    const ImageView& src, const Affine& M, const MutableImageView& dst)
{
  warpAffine(src, dst, M);
}

} // namespace Onnx::ROI
