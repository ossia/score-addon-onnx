#pragma once
#include <Onnx/helpers/Detection.hpp>

#include <QImage>
#include <QPainter>
#include <QTransform>

#include <algorithm>
#include <cmath>

// Region-of-interest construction shared by every two-stage pose pipeline.
//
// All ROIs collapse to a single affine `QTransform M` that maps a point in the
// landmark model's input space (pixels, [0,model_w]x[0,model_h]) to the
// original image (pixels). The same M:
//   - warps the source image into the model crop (via M.inverted()),
//   - maps decoded keypoints back to the image (via M.map()).
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
// face_detection -> face_landmark ROI
inline MediapipeParams faceParams()
{
  return MediapipeParams{
      .kp_start = 0, .kp_end = 1, .target_angle_deg = 0.0f, .scale = 1.5f,
      .alignment_points = false};
}

// Build the rotated crop->image transform from a normalized detection.
inline QTransform mediapipeTransform(
    const Detection::Detection& det, int srcW, int srcH, int model_w,
    int model_h, const MediapipeParams& p)
{
  const auto& kps = det.keypoints;
  const int ke = std::min<int>(p.kp_end, static_cast<int>(kps.size()) - 1);
  const int ks = std::min<int>(p.kp_start, static_cast<int>(kps.size()) - 1);

  float cx, cy, size;
  float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  if(ks >= 0 && ke >= 0)
  {
    x0 = kps[ks].x() * srcW;
    y0 = kps[ks].y() * srcH;
    x1 = kps[ke].x() * srcW;
    y1 = kps[ke].y() * srcH;
  }

  // image Y points down -> negate the y delta for a math-convention angle.
  // No usable alignment keypoints (e.g. a plain person/bbox detector) -> 0.
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

  // shift along the rotated axes (fraction of pre-scale size)
  cx += ca * (p.shift_x * size) - sa * (p.shift_y * size);
  cy += sa * (p.shift_x * size) + ca * (p.shift_y * size);

  const float w = size * p.scale;
  const float h = size * p.scale;

  // M maps (mx,my) in [0,model]^2 to image px:
  //   img = center + R(angle) * ( (m/model - 0.5) * size )
  // QTransform(m11,m12,m21,m22,dx,dy): x' = m11*x + m21*y + dx
  const float m11 = w * ca / model_w;
  const float m12 = w * sa / model_w;
  const float m21 = -h * sa / model_h;
  const float m22 = h * ca / model_h;
  const float dx = cx - 0.5f * w * ca + 0.5f * h * sa;
  const float dy = cy - 0.5f * w * sa - 0.5f * h * ca;
  return QTransform(m11, m12, m21, m22, dx, dy);
}

// Axis-aligned top-down (mmpose) crop->image transform from a source-pixel bbox.
inline QTransform topdownTransform(
    QRectF bbox_px, int model_w, int model_h, float pad = 1.25f)
{
  const float cx = bbox_px.center().x();
  const float cy = bbox_px.center().y();
  float sw = bbox_px.width() * pad;
  float sh = bbox_px.height() * pad;
  const float a = float(model_w) / float(model_h);
  if(sw > sh * a)
    sh = sw / a;
  else
    sw = sh * a;
  return QTransform(
      sw / model_w, 0, 0, sh / model_h, cx - 0.5f * sw, cy - 0.5f * sh);
}

// Whole-frame transform matching Images.hpp's KeepAspectRatioByExpanding +
// center-crop preprocessing (used when no detector is present).
inline QTransform
wholeFrameTransform(int srcW, int srcH, int model_w, int model_h)
{
  const float scale
      = std::max(float(model_w) / srcW, float(model_h) / srcH);
  const float scaled_w = srcW * scale, scaled_h = srcH * scale;
  const float crop_x = (scaled_w - model_w) * 0.5f;
  const float crop_y = (scaled_h - model_h) * 0.5f;
  // img_px = (m_px + crop) / scale
  return QTransform(
      1.0f / scale, 0, 0, 1.0f / scale, crop_x / scale, crop_y / scale);
}

// Aspect-preserving resize into a model_w x model_h square with padding.
// center=true (MediaPipe): pad split both sides. center=false (YOLOX/RTMDet):
// paste top-left, pad bottom-right.
struct LetterboxResult
{
  QImage img;
  float scale{1};
  float pad_x{0};
  float pad_y{0};
};

inline LetterboxResult letterbox(
    const QImage& src, int model_w, int model_h, bool center = true)
{
  LetterboxResult lb;
  lb.scale = std::min(
      float(model_w) / src.width(), float(model_h) / src.height());
  const int nw = static_cast<int>(src.width() * lb.scale);
  const int nh = static_cast<int>(src.height() * lb.scale);
  lb.pad_x = center ? (model_w - nw) * 0.5f : 0.0f;
  lb.pad_y = center ? (model_h - nh) * 0.5f : 0.0f;

  lb.img = QImage(model_w, model_h, QImage::Format_RGBA8888);
  lb.img.fill(Qt::black);
  QPainter p(&lb.img);
  p.setRenderHint(QPainter::SmoothPixmapTransform);
  p.drawImage(QRectF(lb.pad_x, lb.pad_y, nw, nh), src);
  return lb;
}

// Render the source image into a model_w x model_h crop defined by M.
inline QImage
warpCrop(const QImage& src, const QTransform& M, int model_w, int model_h)
{
  QImage crop(model_w, model_h, QImage::Format_RGBA8888);
  crop.fill(Qt::black);
  bool ok = false;
  const QTransform inv = M.inverted(&ok); // image -> crop
  if(ok)
  {
    QPainter p(&crop);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    p.setTransform(inv);
    p.drawImage(0, 0, src);
  }
  return crop;
}

} // namespace Onnx::ROI
